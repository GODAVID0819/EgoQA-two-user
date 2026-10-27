"""三轮Judge反馈环评估（基座或LoRA Generator）：每题最多3轮 生成→三个已训练Judge判定→同一Judge写反馈→重生成。

启动方式与eval_passrate一致：每张卡一个保护进程和一个冻结Judge服务（同卡与Generator交替），前 --count 条验证输入
按rank切分，每条输入生成 --samples 道题。逐轮记录每个Judge的P(pass)、是否通过(>0.5)与反馈，汇总每轮结束后的通过率。
--resume-dir 指向已有输出目录时，沿用原切分并从各rank已保存的轮次继续。
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

KEYS = ('formality', 'groundedness', 'combined_answerability')


def validation_partition(count, rank, world):
    """按rank交错切分（lxr dual_runtime.validation_partition 只支持两卡，这里允许 2 或 4 卡）。"""
    if world not in (2, 4) or count % world or not 0 <= rank < world:
        raise ValueError('验证数量必须可均分到 2 或 4 个rank')
    return list(range(rank, count, world))


def item_state(item, round_index):
    """第 round_index 轮结束后的状态：已接受的题保持接受时那一轮的结果。"""
    done = [a for a in item['attempts'] if a['attempt'] <= round_index]
    return done[-1] if done else None


def summarize(items, max_attempts):
    """每轮结束后各Judge的通过率（分母=全部题）；另给本轮实际评审的题内的通过率。"""
    n = len(items)
    rounds = []
    for r in range(1, max_attempts + 1):
        states = [item_state(i, r) for i in items]
        if any(s is None for s in states):
            raise ValueError(f'第{r}轮有题没有任何结果')
        judged = [a for i in items for a in i['attempts'] if a['attempt'] == r]
        rounds.append({
            'round': r,
            'pass_rate_after_round': {**{k: sum(s['pass'][k] for s in states) / n for k in KEYS},
                                      'all_three': sum(s['all_pass'] for s in states) / n},
            'pass_count_after_round': {**{k: sum(s['pass'][k] for s in states) for k in KEYS},
                                       'all_three': sum(s['all_pass'] for s in states)},
            'judged_this_round': len(judged),
            'pass_rate_among_judged_this_round': {**{k: sum(a['pass'][k] for a in judged) / len(judged) for k in KEYS},
                                                  'all_three': sum(a['all_pass'] for a in judged) / len(judged)} if judged else None,
            'valid_rate_among_judged_this_round': sum(a['status'] == 'scored' for a in judged) / len(judged) if judged else None})
    accepted = [i['accepted_attempt'] for i in items]
    return {'questions': n, 'final': rounds[-1]['pass_rate_after_round'], 'final_count': rounds[-1]['pass_count_after_round'],
            'rounds': rounds, 'accepted_at_attempt': {str(a): accepted.count(a) for a in range(1, max_attempts + 1)},
            'not_accepted': accepted.count(None),
            'mean_attempts': sum(len(i['attempts']) for i in items) / n}


def merge(output, world, count, samples, max_attempts):
    items = []
    for rank in range(world):
        record = json.loads((output / f'rank{rank}' / 'feedback_loop.json').read_text())
        if record['status'] != 'completed':
            raise ValueError(f'rank{rank} 未完成')
        items += record['items']
    keys = {(i['evidence_id'], i['sample']) for i in items}
    if len(items) != count * samples or len(keys) != len(items):
        raise ValueError(f'题数应为 {count}x{samples}，实际 {len(items)}')
    if any(len(i['attempts']) != (i['accepted_attempt'] or max_attempts) for i in items):
        raise ValueError('有题没有跑满轮数或接受后仍继续')
    result = {'adapter': record['adapter'], 'settings': record['settings'], **summarize(items, max_attempts),
              'criterion': 'pass = P(verdict=pass) > 0.5 for each trained judge (judge SFT criterion); accept iff all three pass; '
                           'invalid generator output = fail on all three'}
    (output / 'feedback_loop_summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, help='不给则为未训练基座')
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--max-attempts', type=int, default=3)
    parser.add_argument('--resume-dir', type=Path, help='续跑已有输出目录（沿用原切分，不覆盖已完成轮次）')
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
    output = a.resume_dir or a.output_root / ('fbloop_' + job)
    resume = a.resume_dir is not None
    if resume:
        previous = json.loads((output / 'eval_run.json').read_text())
        if (previous['adapter'], previous['count'], previous['world'], previous['samples'], previous['max_attempts']) != (
                str(a.adapter) if a.adapter else None, a.count, world, a.samples, a.max_attempts):
            raise ValueError('续跑参数与原运行不一致')
    else:
        output.mkdir(parents=True, exist_ok=False)
    scratch, env = scratch_environment(c, job)
    if validate_storage_environment(allowed_root=scratch, environ=env)['status'] != 'passed':
        raise RuntimeError('scratch封闭或可写性失败')
    val_rows = [json.loads(s) for s in Path(c['val_dataset']).read_text().splitlines() if s.strip()][:a.count]
    if len(val_rows) != a.count:
        raise ValueError('验证输入不足')
    run = {'job_id': job, 'config': str(a.config), 'adapter': str(a.adapter) if a.adapter else None, 'count': a.count, 'world': world,
           'samples': a.samples, 'max_attempts': a.max_attempts, 'val_dataset': c['val_dataset'], 'status': 'running'}
    if resume:
        run['resumed_from_jobs'] = previous.get('resumed_from_jobs', []) + [previous['job_id']]
    (output / 'eval_run.json').write_text(json.dumps(run, indent=2))
    children, streams, services, clients, records = [], [], [], [], []

    def child(command, path, child_env):
        stream = path.open('a')
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
            p.mkdir(exist_ok=resume)
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
            text = ''.join(json.dumps({**val_rows[i], 'global_validation_index': i}) + '\n'
                           for i in validation_partition(a.count, rank, world))
            if resume and partition.read_text() != text:
                raise ValueError(f'rank{rank} 续跑切分与原运行不一致')
            partition.write_text(text)
            vc = {**c, 'val_dataset': str(partition), 'world_size': 1, 'num_generations_eval': 1,
                  'policy_gpu_indices': [0], 'judge_gpu_indices': [0]}
            (p / 'validation_config.json').write_text(json.dumps(vc, indent=2))
            eenv = {**train_env, 'CUDA_VISIBLE_DEVICES': gpu, 'EGOQA_GUARD_CUDA_DEVICE': '0',
                    'EGOQA_UTILIZATION_HANDOFF': str(p / 'utilization_handoff.signal'),
                    'EGOQA_UTILIZATION_LOG': str(p / 'utilization_validation.jsonl'),
                    'EGOQA_UTILIZATION_METRICS': str(p / 'gpu_metrics.csv'),
                    'EGOQA_SHARED_GPU_TRACE': str(p / 'validation_phase.jsonl')}
            command = [c['train_python'], '-m', 'training.grpo_v3.six_user_binary.feedback_loop',
                       '--config', str(p / 'validation_config.json'), '--output', str(p / 'feedback_loop.json'),
                       '--judge-url', clients[rank].base_url, '--judge-instance', records[rank]['instance'],
                       '--samples', str(a.samples), '--max-attempts', str(a.max_attempts)]
            if a.adapter:
                command += ['--adapter', str(a.adapter)]
            evals.append(child(command, p / 'feedback_loop.log', eenv))
        for process in evals:
            if process.wait():
                raise RuntimeError('反馈环生成评分失败')
        summary = merge(output, world, a.count, a.samples, a.max_attempts)
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
