"""由torchrun启动2或4个同步训练rank，按rank选择同卡独立Judge。"""
import json
import os
from pathlib import Path
import sys
from dual_runtime import rank_environment, validate_rank_bindings, frozen_visual_key


def install_audit(output):
    import time
    import torch
    import torch.distributed as dist
    from swift.rlhf_trainers.grpo_trainer import GRPOTrainer
    from swift.callbacks import TrainerCallback, callbacks_map
    rank=dist.get_rank() if dist.is_initialized() else int(os.environ['RANK'])
    cache_controller=None
    step_windows={}
    config=json.loads((Path(output)/'run_config.json').read_text())
    def record(name,values):
        p=Path(output)/('rank'+str(rank))/(name+'.jsonl')
        with p.open('a',encoding='utf-8') as f:f.write(json.dumps(values,allow_nan=False)+'\n')
    original_rng=GRPOTrainer._load_rng_state
    def load_rng(trainer,checkpoint,*args,**kwargs):
        result=original_rng(trainer,checkpoint,*args,**kwargs)
        if config.get('resume_from_checkpoint'):
            record('resume_rng',{'global_step':trainer.state.global_step,'rank':rank,
                'event':'native_rng_restore_returned','checkpoint':str(checkpoint)})
        return result
    GRPOTrainer._load_rng_state=load_rng
    from gradient_sync import install
    install(GRPOTrainer,lambda trainer,stats:record('gradient_sync',{
        'global_step':trainer.state.global_step,'rank':rank,**stats}))
    original=GRPOTrainer._generate_completions
    def generate(trainer,inputs):
        if cache_controller is not None:
            key=frozen_visual_key(inputs)
            cache_controller.set_key(key)
            record('frozen_vision_cache',{'global_step':trainer.state.global_step,'event':'group_binding',
                'eligible':key is not None,'image_count':len(key) if key else 0})
        bindings=[{'source_packet_id':r['source_packet_id'],'asker_index':r['asker_index']} for r in inputs]
        mode='train' if trainer.model.training else 'eval'
        groups=[None]*dist.get_world_size();collective_started=time.monotonic()
        dist.all_gather_object(groups,bindings)
        record('phase_timing',{'global_step':trainer.state.global_step,'phase':'generation_group_collective',
            'seconds':time.monotonic()-collective_started,'epoch':time.time()})
        if mode=='train':
            windows=validate_rank_bindings(groups)
            seen=step_windows.setdefault(trainer.state.global_step,[])
            if set(windows)&set(seen) or len(seen)+len(windows)>4:
                raise RuntimeError(f'第{trainer.state.global_step+1}步窗口重复或超过4个：{seen}+{windows}')
            seen.extend(windows)
        if mode=='train' and trainer.state.global_step==config.get('expected_resume_step') and len(step_windows[trainer.state.global_step])==dist.get_world_size():
            from recovery import expected_resume_bindings
            actual=[(group[0]['source_packet_id'],int(group[0]['asker_index'])) for group in groups]
            if actual!=expected_resume_bindings(config):raise RuntimeError('实际恢复采样位置错误')
        record('group_audit',{'global_step':trainer.state.global_step,'mode':mode,'rank':rank,'groups':groups})
        started=time.monotonic();result=original(trainer,inputs)
        record('phase_timing',{'global_step':trainer.state.global_step,'phase':'policy_generate_completions','seconds':time.monotonic()-started,'epoch':time.time()})
        return result
    GRPOTrainer._generate_completions=generate
    for method in ['_prepare_batch_inputs']:
        base=getattr(GRPOTrainer,method)
        def timed(trainer,*args,_base=base,_method=method,**kwargs):
            started=time.monotonic();result=_base(trainer,*args,**kwargs)
            record('phase_timing',{'global_step':trainer.state.global_step,'phase':_method,'seconds':time.monotonic()-started,'epoch':time.time()})
            return result
        setattr(GRPOTrainer,method,timed)
    class DualAudit(TrainerCallback):
        def on_step_begin(self,args,state,control,**kwargs):
            self.step_started=time.monotonic();self.step_epoch=time.time()
            return control
        def verify(self,state,model):
            equal=True;finite=True;elements=0;count=0;differences=[]
            for name,value in model.named_parameters():
                if not value.requires_grad:continue
                local=value.detach();reference=local.clone()
                dist.broadcast(reference,src=0)
                same=torch.equal(local,reference)
                if not same and len(differences)<8:
                    differences.append({'name':name,'max_abs':float((local.float()-reference.float()).abs().max())})
                equal=equal and same
                finite=finite and bool(torch.isfinite(local).all())
                elements+=local.numel();count+=1
            record('ddp_sync',{'global_step':state.global_step,'rank':rank,'world_size':dist.get_world_size(),
                               'cuda_device':torch.cuda.current_device(),'all_trainable_parameters_equal':equal,
                               'checked_elements':elements,'checked_tensors':count,'finite':finite,'differences':differences})
            if not equal or not finite:raise RuntimeError('多卡adapter参数样本不同步或非有限')
        def on_train_begin(self,args,state,control,**kwargs):
            nonlocal cache_controller
            if config.get('resume_from_checkpoint') and state.global_step!=config['expected_resume_step']:
                raise RuntimeError('实际恢复global_step错误，禁止从0重新训练')
            world=config['world_size']
            if dist.get_world_size()!=world or args.generation_batch_size!=4*world or args.steps_per_generation!=4:
                raise ValueError('实际生成批量必须4×world_size、每rank4微批')
            self.verify(state,kwargs['model'])
            if os.environ.get('EGOQA_FROZEN_VISION_CACHE')=='1':
                from frozen_vision_cache import install
                cache_controller=install(kwargs['model'],max_bytes=int(os.environ['EGOQA_FROZEN_VISION_CACHE_MAX_BYTES']))
                record('frozen_vision_cache',{'global_step':state.global_step,'event':'installed',
                    'enabled':cache_controller.enabled,'reason':cache_controller.reason,'cpu_max_bytes':cache_controller.max_bytes})
            return control
        def on_step_end(self,args,state,control,**kwargs):
            record('phase_timing',{'global_step':state.global_step,'phase':'optimizer_step_wall',
                'seconds':time.monotonic()-self.step_started,'started_epoch':self.step_epoch,'epoch':time.time()})
            if state.global_step<=2 or state.global_step%2==0:self.verify(state,kwargs['model'])
            from training.grpo_v3.six_user_binary.policy_image_cache import policy_cache_metrics
            record('policy_image_cache',{'global_step':state.global_step,'epoch':time.time(),'metrics':policy_cache_metrics()})
            if cache_controller is not None:
                record('frozen_vision_cache',{'global_step':state.global_step,'event':'step_end',
                    **cache_controller.stats,'enabled':cache_controller.enabled,'reason':cache_controller.reason})
            return control
        def on_train_end(self,args,state,control,**kwargs):
            nonlocal cache_controller
            if cache_controller is not None:
                record('frozen_vision_cache',{'global_step':state.global_step,'event':'train_end',**cache_controller.stats})
                cache_controller.close();cache_controller=None
            return control
    callbacks_map['egoqa_dual_audit']=DualAudit


def main():
    output=Path(os.environ['EGOQA_DUAL_OUTPUT'])
    records=json.loads((output/'rank_manifest.json').read_text())
    rank=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=len(records):raise RuntimeError('DDP进程数必须等于rank清单')
    os.environ.update(rank_environment(output,records,rank))
    import torch
    torch.cuda.set_device(rank)
    install_audit(output)
    from swift.pipelines import rlhf_main
    rlhf_main()

if __name__=='__main__':main()
