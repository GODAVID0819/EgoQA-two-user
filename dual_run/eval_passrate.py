"""基座与LoRA在固定验证输入上各生成一题，三个冻结Judge逐项按P(pass)>0.5算通过率。

启动方式与dual_launch一致：每张卡一个保护进程和一个冻结Judge服务，dual_evaluate在同卡上
先跑基座再跑adapter（同输入、同种子、同Judge实例）。通过标准与Judge SFT评估一致：p_pass>0.5。
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid
from dual_runtime import validation_partition

KEYS = ('formality', 'groundedness', 'all_six')


def pass_rates(rows):
    """无效生成(未被Judge打分)对每个Judge都算不通过，并单独报告有效率。"""
    n = len(rows)
    passed = {k: [r['status'] == 'scored' and r['probabilities'][k] > .5 for r in rows] for k in KEYS}
    all_three = [all(passed[k][i] for k in KEYS) for i in range(n)]
    scored = [r for r in rows if r['status'] == 'scored']
    return {'questions': n, 'valid_rate': len(scored) / n,
            'pass_rate': {**{k: sum(v) / n for k, v in passed.items()}, 'all_three': sum(all_three) / n},
            'pass_count': {**{k: sum(v) for k, v in passed.items()}, 'all_three': sum(all_three)},
            'mean_p_pass_scored_only': {k: sum(r['probabilities'][k] for r in scored) / len(scored) if scored else None
                                        for k in KEYS},
            'mean_reward': sum(r['reward'] for r in rows) / n}


def merge(output, world, count):
    result = {}
    for label in ('baseline', 'policy'):
        rows = []
        for rank in range(world):
            record = json.loads((output / f'rank{rank}' / f'validation_{label}.json').read_text())
            if record['status'] != 'completed':
                raise ValueError(f'rank{rank} {label} 未完成')
            rows += record['rows']
        if len(rows) != count or len({r['evidence_id'] for r in rows}) != count:
            raise ValueError(f'{label} 题数应为 {count}，实际 {len(rows)}')
        result[label] = {'adapter': record['adapter'], **pass_rates(rows)}
    b = {r['evidence_id']: r for rank in range(world)
         for r in json.loads((output / f'rank{rank}' / 'validation_baseline.json').read_text())['rows']}
    p = {r['evidence_id']: r for rank in range(world)
         for r in json.loads((output / f'rank{rank}' / 'validation_policy.json').read_text())['rows']}
    if set(b) != set(p) or any(b[e]['seed'] != p[e]['seed'] for e in b):
        raise ValueError('基座与adapter的题目不能按输入和种子配对')
    result['criterion'] = 'pass = P(verdict=pass) > 0.5 for each judge (same as judge SFT eval); unscored = fail'
    (output / 'passrate_summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--output-root', type=Path, required=True)
    a = parser.parse_args()
    c = json.loads(a.config.read_text())
    from training.grpo_v3.six_user_binary.launch import scratch_environment, role_environment, select_master_port, cleanup_processes
    from training.torch_storage_preflight import validate_storage_environment
    from training.grpo_v3.six_user_binary.service import JudgeClient, raise_startup_error
    job = os.environ['SLURM_JOB_ID']
    allocation = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    world = len(allocation)
    root = Path(c['project_root'])
    package = root / 'multi-user'
    output = a.output_root / ('eval_' + job)
    output.mkdir(parents=True, exist_ok=False)
    scratch, env = scratch_environment(c, job)
    if validate_storage_environment(allowed_root=scratch, environ=env)['status'] != 'passed':
        raise RuntimeError('scratch封闭或可写性失败')
    val_rows = [json.loads(s) for s in Path(c['val_dataset']).read_text().splitlines() if s.strip()][:a.count]
    if len(val_rows) != a.count:
        raise ValueError('验证输入不足')
    run = {'job_id': job, 'config': str(a.config), 'adapter': str(a.adapter), 'count': a.count, 'world': world,
           'val_dataset': c['val_dataset'], 'num_generations_eval': 1, 'status': 'running'}
    (output / 'eval_run.json').write_text(json.dumps(run, indent=2))
    children, streams, services, clients, records = [], [], [], [], []

    def child(command, path, child_env):
        stream = path.open('w')
        streams.append(stream)
        p = subprocess.Popen(command, cwd=package, env=child_env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(p)
        return p

    def interrupted(signum, frame):
        raise RuntimeError('Slurm终止信号' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        train_env = role_environment(c['train_python'], env, allocator_config=c.get('policy_allocator_config'))
        train_env['PYTHONPATH'] += os.pathsep + str(root)
        if c.get('acceleration_packages'):
            train_env['PYTHONPATH'] += os.pathsep + c['acceleration_packages']
        if c.get('policy_cuda_home'):
            train_env.update(CUDA_HOME=c['policy_cuda_home'], CUDA_PATH=c['policy_cuda_home'])
        guard_json = json.dumps(c['utilization_guard'])
        for rank, gpu in enumerate(allocation):
            p = output / ('rank' + str(rank))
            p.mkdir()
            child(['nvidia-smi', '--id=' + gpu, '--query-gpu=timestamp,index,uuid,name,memory.total,memory.used,utilization.gpu',
                   '--format=csv', '-l', '1'], p / 'gpu_metrics.csv', env)
            guard_env = {**train_env, 'CUDA_VISIBLE_DEVICES': gpu, 'EGOQA_GUARD_CUDA_DEVICE': '0',
                         'EGOQA_UTILIZATION_METRICS': str(p / 'gpu_metrics.csv')}
            child([c['train_python'], '-m', 'training.grpo_v3.six_user_binary.utilization_runtime', '--config', guard_json,
                   '--log', str(p / 'utilization_startup.jsonl'), '--handoff', str(p / 'utilization_handoff.signal')],
                  p / 'utilization_startup.log', guard_env)
            config = json.loads(Path(c['judge_config']).read_text())
            config.update(shared_gpu=True, tensor_parallel_size=1)
            (p / 'judge_config.json').write_text(json.dumps(config, indent=2))
            port = select_master_port()
            while any(v['port'] == port for v in records):
                port = select_master_port()
            records.append({'rank': rank, 'gpu': gpu, 'port': port, 'instance': uuid.uuid4().hex})
            jenv = {**role_environment(c['judge_python'], env), 'CUDA_VISIBLE_DEVICES': gpu}
            services.append(child([c['judge_python'], '-m', 'training.grpo_v3.six_user_binary.service', '--config',
                                   str(p / 'judge_config.json'), '--port', str(port), '--instance-id', records[-1]['instance']],
                                  p / 'judge_service.log', jenv))
            clients.append(JudgeClient('http://127.0.0.1:' + str(port), timeout_seconds=2, expected_instance=records[-1]['instance']))
        deadline = time.monotonic() + c.get('judge_startup_timeout_seconds', 1800)
        for rank, client in enumerate(clients):
            while True:
                raise_startup_error(output / ('rank' + str(rank)) / 'judge_config.json')
                if services[rank].poll() is not None:
                    raise RuntimeError('rank' + str(rank) + ' Judge启动退出')
                try:
                    health = client.health()
                    break
                except (OSError, RuntimeError):
                    if time.monotonic() > deadline:
                        raise RuntimeError('Judge启动超时')
                    time.sleep(2)
            (output / ('rank' + str(rank)) / 'judge_health.json').write_text(json.dumps(health, indent=2))
        train_env.update(EGOQA_SHARED_GPU='1', EGOQA_SIX_USER_REWARD_MODE=c['reward_mode'],
                         EGOQA_POLICY_IMAGE_CACHE_GB=str(c['policy_image_cache_gb']), EGOQA_QWEN_PARTIAL_PACKED_LORA='1',
                         OMP_NUM_THREADS='8')
        evals = []
        for rank, gpu in enumerate(allocation):
            p = output / ('rank' + str(rank))
            partition = p / 'validation.jsonl'
            partition.write_text(''.join(json.dumps({**val_rows[i], 'global_validation_index': i}) + '\n'
                                         for i in validation_partition(a.count, rank, world)))
            vc = {**c, 'val_dataset': str(partition), 'world_size': 1, 'num_generations_eval': 1,
                  'policy_gpu_indices': [0], 'judge_gpu_indices': [0]}
            (p / 'validation_config.json').write_text(json.dumps(vc, indent=2))
            eenv = {**train_env, 'CUDA_VISIBLE_DEVICES': gpu, 'EGOQA_GUARD_CUDA_DEVICE': '0',
                    'EGOQA_UTILIZATION_HANDOFF': str(p / 'utilization_handoff.signal'),
                    'EGOQA_UTILIZATION_LOG': str(p / 'utilization_validation.jsonl'),
                    'EGOQA_UTILIZATION_METRICS': str(p / 'gpu_metrics.csv'),
                    'EGOQA_SHARED_GPU_TRACE': str(p / 'validation_phase.jsonl')}
            evals.append(child([c['train_python'], str(root / 'dual_evaluate.py'), '--config', str(p / 'validation_config.json'),
                                '--output', str(p), '--adapter', str(a.adapter), '--url', clients[rank].base_url,
                                '--instance', records[rank]['instance']], p / 'paired_validation.log', eenv))
        for process in evals:
            if process.wait():
                raise RuntimeError('基座/adapter生成评分失败')
        summary = merge(output, world, a.count)
        run['status'] = 'completed'
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    except BaseException as exc:
        run.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        run['cleanup_errors'] = cleanup_processes(children)
        for f in streams:
            f.close()
        (output / 'eval_run.json').write_text(json.dumps(run, indent=2))


if __name__ == '__main__':
    main()
