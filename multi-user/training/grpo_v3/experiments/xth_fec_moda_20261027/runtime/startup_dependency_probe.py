"""在真实训练/评审角色环境检查启动依赖，禁止初始化CUDA或模型。"""
import argparse
import importlib
import json
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--role',choices=['train','judge'],required=True);args=parser.parse_args()
    root=args.root.resolve()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('此导入检查必须隐藏GPU')
    import torch
    if torch.cuda.is_initialized():raise RuntimeError('CPU检查意外初始化CUDA')
    modules={}
    if args.role=='train':
        import pynvml
        module=importlib.import_module('hpc.shared.cuda_device_identity')
        if not Path(module.__file__).resolve().is_relative_to(root/'hpc'):
            raise RuntimeError('导入了隔离根之外的hpc依赖，不能证明部署已修复')
        assert callable(module.handle_for_cuda_device)
        modules['hpc.shared.cuda_device_identity']=module.__file__
        for name in ['training.grpo_v3.six_user_binary.utilization_runtime','dual_launch','dual_worker']:
            module=importlib.import_module(name);modules[name]=module.__file__
    else:
        from training.grpo_v3.six_user_binary.predictor import validate_config,engine_options
        config=validate_config(json.loads((root/'configs/judge_fec.json').read_text()))
        options=engine_options(config)
        assert options['max_lora_rank']==128 and options['max_loras']==1
        for name in ['training.grpo_v3.six_user_binary.service','training.grpo_v3.six_user_binary.predictor','qwen_vl_utils']:
            module=importlib.import_module(name);modules[name]=module.__file__
    if torch.cuda.is_initialized():raise RuntimeError('入口导入触发CUDA上下文')
    print('STARTUP_ROLE_IMPORT_PASSED',json.dumps({'role':args.role,'modules':modules,
        'GPU_operations':0,'cuda_initialized':False,'model_instantiated':False}),flush=True)


if __name__=='__main__':main()
