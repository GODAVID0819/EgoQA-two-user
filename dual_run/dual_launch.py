"""双卡同节点DDP、每卡独立冻结Judge及同步完整G4评分。"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid
from dual_runtime import validate_config, validation_partition, vision_cache_environment


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    c=json.loads(parser.parse_args().config.read_text());validate_config(c)
    from training.grpo_v3.six_user_binary.launch import scratch_environment,role_environment,select_master_port,swift_command,cleanup_processes
    from training.torch_storage_preflight import validate_storage_environment
    from training.grpo_v3.six_user_binary.service import JudgeClient, raise_startup_error
    job=os.environ['SLURM_JOB_ID'];allocation=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(allocation)!=2:raise RuntimeError('必须实际分配两张GPU')
    root=Path(c['project_root']);package=root/'multi-user';output=Path(c['output_root'])/('train_'+job)
    output.mkdir(parents=True,exist_ok=False)
    scratch,env=scratch_environment(c,job)
    storage=validate_storage_environment(allowed_root=scratch,environ=env)
    (output/'storage_preflight.json').write_text(json.dumps(storage,indent=2))
    if storage['status']!='passed':raise RuntimeError('scratch封闭或可写性失败')
    (output/'run_config.json').write_text(json.dumps(c,indent=2))
    children=[];streams=[];services=[];clients=[];records=[];guards=[]
    result={'status':'running','job_id':job,'world_size':2,'config':c}
    def child(command,path,child_env):
        stream=path.open('w');streams.append(stream)
        p=subprocess.Popen(command,cwd=package,env=child_env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(p);return p
    def interrupted(signum,frame):raise RuntimeError('Slurm终止信号'+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        train_env=role_environment(c['train_python'],env,allocator_config=c.get('policy_allocator_config'))
        train_env['PYTHONPATH']+=os.pathsep+str(root)
        if c.get('acceleration_packages'):train_env['PYTHONPATH']+=os.pathsep+c['acceleration_packages']
        if c.get('policy_cuda_home'):train_env.update(CUDA_HOME=c['policy_cuda_home'],CUDA_PATH=c['policy_cuda_home'])
        guard_json=json.dumps(c['utilization_guard'])
        for rank,gpu in enumerate(allocation):
            p=output/('rank'+str(rank));p.mkdir()
            child(['nvidia-smi','--id='+gpu,'--query-gpu=timestamp,index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv','-l','1'],p/'gpu_metrics.csv',env)
            handoff=str(p/'utilization_handoff.signal')
            guard_env={**train_env,'CUDA_VISIBLE_DEVICES':gpu,'EGOQA_GUARD_CUDA_DEVICE':'0',
                       'EGOQA_UTILIZATION_METRICS':str(p/'gpu_metrics.csv')}
            guards.append(child([c['train_python'],'-m','training.grpo_v3.six_user_binary.utilization_runtime',
                '--config',guard_json,'--log',str(p/'utilization_startup.jsonl'),'--handoff',handoff],p/'utilization_startup.log',guard_env))
            config=json.loads(Path(c['judge_config']).read_text());config['shared_gpu']=True
            config['tensor_parallel_size']=1
            (p/'judge_config.json').write_text(json.dumps(config,indent=2))
            port=select_master_port();instance=uuid.uuid4().hex
            while any(v['port']==port for v in records):port=select_master_port()
            rec={'rank':rank,'gpu':gpu,'port':port,'instance':instance};records.append(rec)
            jenv={**role_environment(c['judge_python'],env),'CUDA_VISIBLE_DEVICES':gpu}
            services.append(child([c['judge_python'],'-m','training.grpo_v3.six_user_binary.service',
                '--config',str(p/'judge_config.json'),'--port',str(port),'--instance-id',instance],p/'judge_service.log',jenv))
            clients.append(JudgeClient('http://127.0.0.1:'+str(port),timeout_seconds=2,expected_instance=instance))
        (output/'rank_manifest.json').write_text(json.dumps(records,indent=2))
        from recovery import seed_reward_prefix, validated_resume
        recovery=seed_reward_prefix(c,output)
        deadline=time.monotonic()+c.get('judge_startup_timeout_seconds',1800)
        for rank,client in enumerate(clients):
            while True:
                raise_startup_error(output/('rank'+str(rank))/'judge_config.json')
                if services[rank].poll() is not None or guards[rank].poll() not in (None,0):
                    raise RuntimeError('rank'+str(rank)+'启动Judge/保护退出')
                try:health=client.health();break
                except (OSError,RuntimeError):
                    if time.monotonic()>deadline:raise RuntimeError('Judge启动超时')
                    time.sleep(2)
            (output/('rank'+str(rank))/'judge_health.json').write_text(json.dumps(health,indent=2))
            if rank==0:(output/'judge_health.json').write_text(json.dumps(health,indent=2))
        train_env.update(CUDA_VISIBLE_DEVICES=','.join(allocation),EGOQA_DUAL_OUTPUT=str(output),
            EGOQA_UTILIZATION_GUARD=guard_json,EGOQA_SHARED_GPU='1',EGOQA_SIX_USER_REWARD_MODE=c['reward_mode'],
            EGOQA_GRPO_STOP_AT_STEP=str(c['stop_after_steps']),EGOQA_POLICY_IMAGE_CACHE_GB=str(c['policy_image_cache_gb']),
            EGOQA_QWEN_PARTIAL_PACKED_LORA='1',OMP_NUM_THREADS='8')
        train_env.update(vision_cache_environment(c))
        if recovery:
            train_env['EGOQA_RESUME_METADATA']=json.dumps(validated_resume(c))
        # 原CLI构造器描述单个rank的同卡轮换；全局两卡由本父入口管理。
        cmd=swift_command({**c,'policy_gpu_indices':[0],'judge_gpu_indices':[0]},str(output))[2:]
        cb=cmd.index('--callbacks');cmd[cb+1]='egoqa_stage_stop';cmd.insert(cb+2,'egoqa_dual_audit')
        cmd.extend(['--steps_per_generation','4','--ddp_find_unused_parameters','false'])
        if recovery:
            cmd.extend(['--resume_only_model','false','--ignore_data_skip','false'])
        command=[c['train_python'],'-m','torch.distributed.run','--nproc_per_node=2','--master_addr=127.0.0.1',
                 '--master_port='+str(select_master_port()),str(root/'dual_worker.py'),*cmd]
        (output/'train_command.json').write_text(json.dumps(command,indent=2))
        trainer=child(command,output/'trainer.log',train_env)
        while trainer.poll() is None:
            if any(p.poll() is not None for p in services):raise RuntimeError('训练中Judge退出')
            time.sleep(3)
        if trainer.returncode:raise RuntimeError('双卡训练退出码'+str(trainer.returncode))
        errors=cleanup_processes([trainer]);children.remove(trainer)
        if errors:raise RuntimeError(str(errors))
        with (output/'reward_trace.jsonl').open('w') as f:
            for rank in range(2):f.write((output/('rank'+str(rank))/'reward_trace.jsonl').read_text())
        subprocess.run([c['train_python'],'-m','training.grpo_v3.six_user_binary.validate_run','--output',str(output)],cwd=package,env=env,check=True)
        checked=json.loads((output/'training_result.json').read_text());adapter=checked['checkpoint']
        if c['stop_after_steps']<c['max_steps']:
            # 中间阶段只保存完整断点；120条件固定配对验证只在调度终点做一次。
            result['status']='completed';result['stage_only']=True
            return
        # 两卡并行固定验证，各自基座/adapter始终使用同一Judge实例和输入。
        val_rows=[json.loads(s) for s in Path(c['val_dataset']).read_text().splitlines() if s.strip()]
        eval_processes=[]
        for rank,gpu in enumerate(allocation):
            p=output/('rank'+str(rank));indices=validation_partition(len(val_rows),rank)
            partition=p/'validation.jsonl'
            partition.write_text(''.join(json.dumps({**val_rows[i],'global_validation_index':i})+'\n' for i in indices))
            vc={**c,'val_dataset':str(partition),'world_size':1,'policy_gpu_indices':[0],'judge_gpu_indices':[0]}
            (p/'validation_config.json').write_text(json.dumps(vc,indent=2))
            eenv={**train_env,'CUDA_VISIBLE_DEVICES':gpu,'EGOQA_GUARD_CUDA_DEVICE':'0',
                  'EGOQA_UTILIZATION_HANDOFF':str(p/'utilization_handoff.signal'),
                  'EGOQA_UTILIZATION_LOG':str(p/'utilization_validation.jsonl'),
                  'EGOQA_UTILIZATION_METRICS':str(p/'gpu_metrics.csv'),
                  'EGOQA_SHARED_GPU_TRACE':str(p/'validation_phase.jsonl')}
            for key in ['EGOQA_DUAL_OUTPUT','RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT']:eenv.pop(key,None)
            command=[c['train_python'],str(root/'dual_evaluate.py'),'--config',str(p/'validation_config.json'),
                     '--output',str(p),'--adapter',adapter,'--url',clients[rank].base_url,'--instance',records[rank]['instance']]
            eval_processes.append(child(command,p/'paired_validation.log',eenv))
        for process in eval_processes:
            if process.wait():raise RuntimeError('固定配对验证失败')
        from dual_evaluate import merge_validation
        merge_validation(output)
        result['status']='completed'
    except BaseException as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        result['cleanup_errors']=cleanup_processes(children)
        for f in streams:f.close()
        (output/'run_manifest.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
