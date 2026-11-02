"""异步登录节点验证控制器；每项保存日志，任一失败停止，不提交Slurm。"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from .launch import role_environment


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reuse-passed-from',type=Path)
    args=parser.parse_args()
    c=json.loads(args.config.read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    status={'status':'running','config':str(args.config),'started_epoch':time.time(),'checks':[]}
    def save():(args.output/'status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    save()
    try:
        tasks=[('train','environment'),('judge','environment'),('train','data'),('train','processor'),('judge','processor')]
        if c.get('resume_from_checkpoint'):tasks.append(('train','checkpoint'))
        for role,mode in tasks:
            name=role+'_'+mode
            if args.reuse_passed_from:
                source=args.reuse_passed_from/(name+'.json')
                source_status=json.loads((args.reuse_passed_from/'status.json').read_text())
                if Path(source_status['config']).resolve()!=args.config.resolve():
                    raise ValueError('不可复用另一配置的零GPU检查')
                prior=json.loads(source.read_text()) if source.exists() else {}
                if prior.get('status')=='passed':
                    if prior.get('role')!=role or prior.get('mode')!=mode or prior.get('python')!=c[role+'_python']:
                        raise ValueError('历史检查角色或环境身份不一致')
                    prior['reused_from']=str(source)
                    (args.output/(name+'.json')).write_text(json.dumps(prior,indent=2),encoding='utf-8')
                    status['checks'].append({'name':name,'exit_code':0,'reused_from':str(source)})
                    save()
                    continue
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                PYTHONPATH=str(Path(c['project_root'])/'multi-user'),VLLM_NO_USAGE_STATS='1')
            for key in ('HOME','XDG_CACHE_HOME','HF_HOME','HF_DATASETS_CACHE','MODELSCOPE_CACHE','TORCH_HOME','TRITON_CACHE_DIR',
                        'TORCHINDUCTOR_CACHE_DIR','VLLM_CACHE_ROOT','CUDA_CACHE_PATH','FLASHINFER_WORKSPACE_BASE','TMPDIR','TMP','TEMP'):
                env[key]=str(args.output/role/key.lower());Path(env[key]).mkdir(parents=True,exist_ok=True)
            env['PATH']=str(Path(c['ffmpeg']).parent)+os.pathsep+env['PATH']
            env['LD_LIBRARY_PATH']=str(Path(c['ffmpeg']).parent.parent/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH','')
            env=role_environment(c[role+'_python'],env)
            env.update(c.get('compiler_environment',{}))
            if role=='train' and c.get('policy_cuda_home'):
                env.update(CUDA_HOME=c['policy_cuda_home'],CUDA_PATH=c['policy_cuda_home'])
            if role=='train':env['PYTHONPATH']+=os.pathsep+c['acceleration_packages']
            status['current_check']=name;save()
            with (args.output/(name+'.log')).open('w') as log:
                result=subprocess.run([c[role+'_python'],'-u','-m','training.grpo_v3.six_user_binary.zero_gpu',
                    '--config',str(args.config),'--output',str(args.output/(name+'.json')),'--role',role,'--mode',mode],
                    cwd=Path(c['project_root'])/'multi-user',env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
            status['checks'].append({'name':name,'exit_code':result.returncode});save()
            result.check_returncode()
        status.update(status='passed',finished_epoch=time.time())
    except BaseException as exc:
        status.update(status='failed',error=f'{type(exc).__name__}: {exc}',finished_epoch=time.time())
        raise
    finally:save()


if __name__=='__main__':main()
