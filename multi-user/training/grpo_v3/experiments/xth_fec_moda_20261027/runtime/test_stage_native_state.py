"""用真实CPU张量、Adam、scheduler和safetensors检查重载；这是夹具，不是真实CP180。"""
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'multi-user'))


@unittest.skipUnless(importlib.util.find_spec('torch'),'本地无Torch；必须在Torch CPU角色执行')
class NativeCheckpointTests(unittest.TestCase):
    def make(self,root,target,capacity=False):
        import numpy as np
        import torch
        from safetensors.torch import save_file
        from transformers.optimization import get_scheduler
        c={'learning_rate':1e-5,'lr_scheduler_type':'cosine_with_min_lr','warmup_steps':10,
            'max_steps':360,'lr_scheduler_kwargs':{'min_lr_rate':.1},'stop_after_steps':target}
        if capacity:c['capacity_measurement']=True
        p=root/'swift/v0'/f'checkpoint-{target}';p.mkdir(parents=True)
        names=[];parameters=[]
        for layer in range(64):
            for module in (['q_proj','v_proj'] if layer%4==3 else ['in_proj_qkv']):
                for side in ['A','B']:
                    names.append(f'base_model.model.model.layers.{layer}.{module}.lora_{side}.weight')
                    value=0. if side=='B' else .01
                    parameters.append(torch.nn.Parameter(torch.full((64,1) if side=='A' else (1,64),value)))
        optimizer=torch.optim.AdamW(parameters,lr=1e-5)
        scheduler=get_scheduler('cosine_with_min_lr',optimizer,num_warmup_steps=10,num_training_steps=360,
            scheduler_specific_kwargs={'min_lr_rate':.1})
        for _ in range(target):
            for value in parameters:value.grad=torch.ones_like(value)
            optimizer.step();scheduler.step();optimizer.zero_grad()
        save_file({name:value.detach().contiguous() for name,value in zip(names,parameters)},str(p/'adapter_model.safetensors'))
        (p/'adapter_config.json').write_text(json.dumps({'r':64,'lora_alpha':128,'lora_dropout':.05}))
        (p/'trainer_state.json').write_text(json.dumps({'global_step':target,'max_steps':360}))
        torch.save(optimizer.state_dict(),p/'optimizer.pt');torch.save(scheduler.state_dict(),p/'scheduler.pt')
        for rank in range(2):
            torch.save({'python':random.getstate(),'numpy':np.random.get_state(),'cpu':torch.get_rng_state(),
                'cuda':[torch.get_rng_state()]},p/f'rng_state_{rank}.pth')
        return p,c

    def test_real_cpu_optimizer_and_cosine_halfway_reload(self):
        from training.grpo_v3.six_user_binary.validate_run import finite_updated_adapter
        with tempfile.TemporaryDirectory() as directory:
            p,c=self.make(Path(directory),180)
            self.assertTrue(finite_updated_adapter(p/'adapter_model.safetensors',config=c))
            result=json.loads((p/'reload_cpu_audit.json').read_text())
            self.assertEqual(result['optimizer_states'],160)
            self.assertEqual(result['global_step'],180)
            self.assertTrue(result['nonzero_lora_B'])
            self.assertAlmostEqual(result['last_lr'][0],5.701891736297396e-6,places=12)

    def test_capacity_zero_LR_first_step_may_have_zero_B(self):
        from training.grpo_v3.six_user_binary.validate_run import finite_updated_adapter
        with tempfile.TemporaryDirectory() as directory:
            p,c=self.make(Path(directory),1,capacity=True)
            self.assertTrue(finite_updated_adapter(p/'adapter_model.safetensors',config=c))
            self.assertFalse(json.loads((p/'reload_cpu_audit.json').read_text())['nonzero_lora_B'])

    def test_missing_Adam_moment_is_rejected(self):
        import torch
        from training.grpo_v3.six_user_binary.validate_run import finite_updated_adapter
        with tempfile.TemporaryDirectory() as directory:
            p,c=self.make(Path(directory),180)
            optimizer=torch.load(p/'optimizer.pt',map_location='cpu',weights_only=False)
            del optimizer['state'][next(iter(optimizer['state']))]['exp_avg_sq']
            torch.save(optimizer,p/'optimizer.pt')
            with self.assertRaises(ValueError):finite_updated_adapter(p/'adapter_model.safetensors',config=c)


if __name__=='__main__':unittest.main()
