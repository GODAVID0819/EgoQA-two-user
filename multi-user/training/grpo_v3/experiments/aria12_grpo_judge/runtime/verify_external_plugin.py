"""在CPU作业中执行Swift真实独立文件加载入口，不加载模型。"""
import json,os,sys
from pathlib import Path
config=json.loads(Path(sys.argv[1]).read_text())
from cpu_validate import setup
root,scratch=setup(config)
os.environ['EGOQA_QWEN_PARTIAL_PACKED_LORA']='1'
from swift.utils import import_external_file
from swift.rewards import orms
from swift.callbacks import callbacks_map
plugin=root/'multi-user/training/grpo_v3/six_user_binary/plugin.py'
module=import_external_file(str(plugin))
assert Path(module.__file__).resolve()==plugin.resolve()
assert module.__package__==''
assert orms['egoqa_fec_moda_v1'] is module.SixUserBinaryReward
assert 'egoqa_stage_stop' in callbacks_map
from dual_worker import install_audit
audit=scratch/'audit'
audit.mkdir(parents=True,exist_ok=True)
(audit/'run_config.json').write_text(json.dumps(config))
os.environ['RANK']='0'
install_audit(audit)
assert 'egoqa_dual_audit' in callbacks_map
import torch
assert not torch.cuda.is_initialized() and not torch.distributed.is_initialized()
result={'status':'passed','job_id':os.environ['SLURM_JOB_ID'],'loader':import_external_file.__module__+'.'+import_external_file.__name__,'plugin':str(plugin),'package':module.__package__,'reward_registered':True,'callbacks_registered':['egoqa_stage_stop','egoqa_dual_audit'],'packed_lora_compat_enabled':True,'GPU_models_loaded':False}
(root/'preflight/external_plugin_validation.json').write_text(json.dumps(result,indent=2))
print('ARIA3_REAL_EXTERNAL_PLUGIN_PASS',json.dumps(result),flush=True)
