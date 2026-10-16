"""优化器更新前显式归约LoRA梯度；不广播更新后的权重来掩盖失步。"""
import time
from functools import wraps


def average_gradients(model, *, max_bucket_bytes=8*1024*1024):
    import torch
    import torch.distributed as dist
    if not dist.is_initialized() or dist.get_world_size() not in (2,4):
        raise RuntimeError('本轮梯度均值同步要求2或4个实际分布式进程')
    started=time.monotonic()
    named=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
    if not named:raise RuntimeError('缺少可训练参数')
    device=named[0][1].device
    presence=torch.tensor([p.grad is not None for _,p in named],device=device,dtype=torch.int32)
    dist.all_reduce(presence,op=dist.ReduceOp.SUM)
    counts=presence.tolist()
    groups={};filled=0
    for (name,p),count in zip(named,counts):
        if not count:continue
        if p.grad is None:
            p.grad=torch.zeros_like(p);filled+=1
        if p.grad.is_sparse or not p.grad.is_floating_point():
            raise RuntimeError('本轮LoRA需要稠密实数梯度：'+name)
        groups.setdefault(p.grad.device,[]).append(p.grad)
    maximum_change=0.;elements=0;tensors=0;buckets=0
    def reduce(bucket):
        nonlocal maximum_change,elements,tensors,buckets
        if not bucket:return
        flat=torch.cat([g.detach().reshape(-1).float() for g in bucket])
        before=flat.clone()
        dist.all_reduce(flat,op=dist.ReduceOp.SUM)
        flat.div_(dist.get_world_size())
        if not bool(torch.isfinite(flat).all()):raise RuntimeError('全局梯度含非有限值')
        maximum_change=max(maximum_change,float((before-flat).abs().max()))
        offset=0
        for grad in bucket:
            count=grad.numel()
            grad.copy_(flat[offset:offset+count].view_as(grad).to(grad.dtype));offset+=count
        elements+=flat.numel();tensors+=len(bucket);buckets+=1
    with torch.no_grad():
        for gradients in groups.values():
            bucket=[];size=0
            for grad in gradients:
                amount=grad.numel()*4
                if bucket and size+amount>max_bucket_bytes:
                    reduce(bucket);bucket=[];size=0
                bucket.append(grad);size+=amount
            reduce(bucket)
    return {'world_size':dist.get_world_size(),'tensors':tensors,'elements':elements,
            'filled_local_missing_gradients':filled,'buckets':buckets,
            'max_local_to_mean_change':maximum_change,'finite':True,
            'seconds':time.monotonic()-started,'order':'mean_then_clip_then_optimizer'}


def install(trainer_cls, record):
    original_clip=trainer_cls._clip_grad_norm
    original_norm=trainer_cls._get_grad_norm
    if getattr(original_clip,'_egoqa_gradient_sync',False):
        raise RuntimeError('同一训练器不能重复安装梯度同步')
    def ensure(trainer,model):
        step=trainer.state.global_step
        if getattr(trainer,'_egoqa_grad_sync_step',None)==step:return
        import os,json
        if os.environ.get('EGOQA_RESUME_METADATA') and not getattr(trainer,'_egoqa_resume_checked',False):
            from recovery import check_restored_optimizer
            record(trainer,{'event':'resume_optimizer_verified',**check_restored_optimizer(
                trainer,json.loads(os.environ['EGOQA_RESUME_METADATA']))})
            trainer._egoqa_resume_checked=True
        unwrapped=trainer.accelerator.unwrap_model(model)
        stats=average_gradients(unwrapped)
        stats.update(model_type=type(model).__qualname__,
                     model_wrapped_type=type(getattr(trainer,'model_wrapped',model)).__qualname__,
                     accelerator_distributed_type=str(trainer.accelerator.distributed_type),
                     accelerator_processes=trainer.accelerator.num_processes)
        record(trainer,stats)
        trainer._egoqa_grad_sync_step=step
    @wraps(original_clip)
    def clip(trainer,model):
        ensure(trainer,model)
        return original_clip(trainer,model)
    @wraps(original_norm)
    def norm(trainer,model,grad_norm=None):
        ensure(trainer,model)
        return original_norm(trainer,model,grad_norm=grad_norm)
    clip._egoqa_gradient_sync=True
    trainer_cls._clip_grad_norm=clip
    trainer_cls._get_grad_norm=norm
