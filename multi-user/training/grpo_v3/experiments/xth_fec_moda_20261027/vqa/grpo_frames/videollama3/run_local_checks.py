"""本地CPU检查：原GRPO提示词与真实官方图片处理器，无7B权重或GPU。"""
import os
import sys
import time
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['HF_HOME']=str(ROOT/'local_hf_cache')
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'


if __name__=='__main__':
    start=time.time()
    suite=unittest.defaultTestLoader.loadTestsFromNames(['test_grpo','test_model_adapter','test_frame_cache','test_batching','test_resume','test_rebuild'])
    with (ROOT/'local_tests.log').open('w',encoding='utf-8') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    import grpo_inputs as core
    core.write_json(ROOT/'LOCAL_VERIFICATION.json',{'status':'PASS' if result.wasSuccessful() else 'FAIL',
        'tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),
        'gpu_used':False,'model_weights_loaded':False,'elapsed_seconds':time.time()-start,
        'coverage':'131_questions_393_prompts_synthetic_156_images_original_rescale_and_source_prompt'})
    print((ROOT/'LOCAL_VERIFICATION.json').read_text(encoding='utf-8'))
    if not result.wasSuccessful():raise SystemExit(1)
