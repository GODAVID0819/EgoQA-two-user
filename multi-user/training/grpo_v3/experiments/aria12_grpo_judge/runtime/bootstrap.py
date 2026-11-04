"""在任何模型依赖导入前封闭本作业缓存，再进入CPU检查或双卡训练。"""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['cpu','train'],required=True)
    parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    config=json.loads(args.config.read_text())
    job=os.environ['SLURM_JOB_ID'];root=Path(config['project_root'])
    scratch=Path(config['scratch_root'])/job
    env=os.environ.copy()
    keys=['HOME','XDG_CACHE_HOME','HF_HOME','HF_DATASETS_CACHE','MODELSCOPE_CACHE','TORCH_HOME',
          'TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','VLLM_CACHE_ROOT','CUDA_CACHE_PATH',
          'FLASHINFER_WORKSPACE_BASE','TMPDIR','TMP','TEMP']
    for key in keys:
        path=scratch/key.lower();path.mkdir(parents=True,exist_ok=True);env[key]=str(path)
    media=json.loads((root/'preflight/media_tools.json').read_text())
    env['PATH']=str(Path(media['ffmpeg']).parent)+os.pathsep+str(Path(config['train_python']).parent)+os.pathsep+env.get('PATH','')
    env['LD_LIBRARY_PATH']=media['ffmpeg_lib']+os.pathsep+env.get('LD_LIBRARY_PATH','')
    env.update({k:str(v) for k,v in config.get('compiler_environment',{}).items()})
    if config.get('policy_cuda_home'):
        env.update(CUDA_HOME=config['policy_cuda_home'],CUDA_PATH=config['policy_cuda_home'])
    env.update(PYTHONDONTWRITEBYTECODE='1',VLLM_NO_USAGE_STATS='1',TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8')
    paths=[str(root/'runtime_deps_train'),str(root/'multi-user'),str(root)]
    if config.get('acceleration_packages'):paths.append(config['acceleration_packages'])
    env['PYTHONPATH']=os.pathsep.join(paths)
    if args.mode=='cpu':env['CUDA_VISIBLE_DEVICES']=''
    script='cpu_validate.py' if args.mode=='cpu' else 'dual_launch.py'
    if args.mode=='cpu':
        subprocess.run([config['train_python'],str(root/'prepare_run.py'),'data','--config',str(args.config.resolve())],
                       cwd=root,env=env,check=True)
    command=[config['train_python'],'-u',str(root/script),'--config',str(args.config.resolve())]
    print('ARIA3_BOOTSTRAP',json.dumps({'job_id':job,'mode':args.mode,'command':command,'scratch':str(scratch)}),flush=True)
    os.execvpe(command[0],command,env)


if __name__=='__main__':main()
