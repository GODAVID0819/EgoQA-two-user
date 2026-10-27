"""核对已归档diversity与阶段语义，不加载模型、不访问Torch或提交作业。"""
from pathlib import Path
import os
import sys
import unittest


def main():
    here=Path(__file__).resolve().parent
    runtime=here/'runtime'
    package=next(p for p in here.parents if p.name=='multi-user')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    sys.dont_write_bytecode=True
    sys.path[:0]=[str(runtime),str(package)]
    import training
    import training.grpo_v3
    training.grpo_v3.__path__.insert(0,str(runtime/'multi-user/training/grpo_v3'))
    modules=[
        'test_moda_data','test_moda_history','test_moda_history_stage',
        'test_moda_judge','test_moda_plugin','test_moda_reward',
        'test_moda_training','test_moda_validation',
        'test_stage_capacity_inputs','test_stage_validation','test_stage_resume',
        'test_judge_prompt_assets','test_combined_allocation',
    ]
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in modules)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':
    raise SystemExit(main())
