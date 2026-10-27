"""由torchrun启动两个同步训练rank，按rank选择同卡独立Judge。"""
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
    checkpoint_offload_controller=None
    config=json.loads((Path(output)/'run_config.json').read_text())
    def record(name,values):
        import resource
        values={**values,'memory_allocated_bytes':torch.cuda.memory_allocated(),
            'memory_reserved_bytes':torch.cuda.memory_reserved(),
            'peak_memory_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_memory_reserved_bytes':torch.cuda.max_memory_reserved(),
            'process_peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
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
        groups=[None,None];collective_started=time.monotonic()
        dist.all_gather_object(groups,bindings)
        record('phase_timing',{'global_step':trainer.state.global_step,'phase':'generation_group_collective',
            'seconds':time.monotonic()-collective_started,'epoch':time.time()})
        if mode=='train':validate_rank_bindings(groups)
        if mode=='train' and trainer.state.global_step==config.get('expected_resume_step'):
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
    original_loss=GRPOTrainer._compute_loss_and_metrics
    def audited_loss(trainer,*args,**kwargs):
        try:return original_loss(trainer,*args,**kwargs)
        finally:
            if checkpoint_offload_controller is not None:
                record('checkpoint_input_offload',{'global_step':trainer.state.global_step,
                    'event':'loss_forward_finished_or_failed',**checkpoint_offload_controller.stats})
    GRPOTrainer._compute_loss_and_metrics=audited_loss
    class DualAudit(TrainerCallback):
        def __init__(self,args,trainer):
            super().__init__(args,trainer)
            self.trainer=trainer
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
            if not equal or not finite:raise RuntimeError('双卡adapter参数样本不同步或非有限')
        def on_train_begin(self,args,state,control,**kwargs):
            nonlocal cache_controller,checkpoint_offload_controller
            if config.get('resume_from_checkpoint') and state.global_step!=config['expected_resume_step']:
                raise RuntimeError('实际恢复global_step错误，禁止从0重新训练')
            if dist.get_world_size()!=2 or args.generation_batch_size!=8 or args.steps_per_generation!=4:
                raise ValueError('实际双卡生成批量必须8、每rank4微批')
            expected_rows=2 if config.get('capacity_measurement') else 720
            actual_rows=len(self.trainer.train_dataset)
            truncation=getattr(self.trainer.template,'truncation_strategy',None)
            record('dataset_retention',{'global_step':state.global_step,'expected_rows':expected_rows,
                'actual_rows':actual_rows,'truncation_strategy':truncation})
            if actual_rows!=expected_rows or truncation!='raise':
                raise ValueError('正式数据被删行或没有启用超长即报错')
            self.verify(state,kwargs['model'])
            if config.get('checkpoint_input_cpu_offload') is True:
                from checkpoint_input_offload import install as install_checkpoint_offload
                checkpoint_offload_controller=install_checkpoint_offload(kwargs['model'],
                    layer_stride=config.get('checkpoint_input_cpu_offload_stride',4))
                record('checkpoint_input_offload',{'global_step':state.global_step,'event':'installed',
                    'selected_layers':checkpoint_offload_controller.selected_layers,
                    'storage':'CPU','input_only':True,'original_checkpoint_and_RNG_preserved':True})
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
            if checkpoint_offload_controller is not None:
                record('checkpoint_input_offload',{'global_step':state.global_step,'event':'step_end',
                    **checkpoint_offload_controller.stats})
            from training.grpo_v3.six_user_binary.policy_image_cache import policy_cache_metrics
            record('policy_image_cache',{'global_step':state.global_step,'epoch':time.time(),'metrics':policy_cache_metrics()})
            if cache_controller is not None:
                record('frozen_vision_cache',{'global_step':state.global_step,'event':'step_end',
                    **cache_controller.stats,'enabled':cache_controller.enabled,'reason':cache_controller.reason})
            return control
        def on_train_end(self,args,state,control,**kwargs):
            nonlocal cache_controller,checkpoint_offload_controller
            if checkpoint_offload_controller is not None:
                record('checkpoint_input_offload',{'global_step':state.global_step,'event':'train_end',
                    **checkpoint_offload_controller.stats})
                checkpoint_offload_controller.close();checkpoint_offload_controller=None
            if cache_controller is not None:
                record('frozen_vision_cache',{'global_step':state.global_step,'event':'train_end',**cache_controller.stats})
                cache_controller.close();cache_controller=None
            return control
    callbacks_map['egoqa_dual_audit']=DualAudit


def main():
    output=Path(os.environ['EGOQA_DUAL_OUTPUT'])
    records=json.loads((output/'rank_manifest.json').read_text())
    rank=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=2:raise RuntimeError('只允许两个DDP进程')
    os.environ.update(rank_environment(output,records,rank))
    import torch
    torch.cuda.set_device(rank)
    install_audit(output)
    from swift.pipelines import rlhf_main
    rlhf_main()

if __name__=='__main__':main()
