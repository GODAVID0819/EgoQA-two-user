"""加载既有LoRA作为新数据阶段初值，不恢复旧优化器、数据位置或调度。"""
import json
from pathlib import Path,PurePosixPath
import re
import struct

RECOVERY_FIELDS=('resume_from_checkpoint','resume_source_output','recovery_from_job','expected_resume_step')

def validate_initialization(c):
    if any(c.get(k) for k in RECOVERY_FIELDS):
        raise ValueError('新数据阶段不恢复旧optimizer、scheduler、RNG或数据游标')
    if any(c.get(k) for k in ('adapters','adapter','policy_adapter','ref_adapters')):
        raise ValueError('请使用明确的initial_adapter/reference_adapter配置')
    mode=c.get('initialization_mode','fresh_base')
    if mode=='fresh_base':
        if c.get('initial_fresh_base') is not True or c.get('initial_adapter') or c.get('reference_adapter'):
            raise ValueError('fresh_base必须新建LoRA，不能混入既有adapter')
        return
    if mode!='adapter_warm_start' or c.get('initial_fresh_base') is not False or c.get('reset_training_state') is not True:
        raise ValueError('adapter_warm_start必须明确新建训练状态')
    path=c.get('initial_adapter')
    if not isinstance(path,str) or not PurePosixPath(path).is_absolute() or not PurePosixPath(path).is_relative_to('/scratch'):
        raise ValueError('initial_adapter必须是scratch上的绝对路径')
    if c.get('reference_adapter')!=path:
        raise ValueError('新阶段KL reference必须绑定同一个初始adapter')
    if type(c.get('initial_checkpoint_step')) is not int or c['initial_checkpoint_step']<=0:
        raise ValueError('必须记录来源checkpoint实际步数')

def validate_adapter_files(path,c):
    path=Path(path)
    cfg=json.loads((path/'adapter_config.json').read_text(encoding='utf-8'))
    state=json.loads((path/'trainer_state.json').read_text(encoding='utf-8'))
    for key,expected in [('r',c['lora_rank']),('lora_alpha',c['lora_alpha']),('lora_dropout',c['lora_dropout'])]:
        if cfg.get(key)!=expected:raise ValueError('初始adapter超参不匹配：'+key)
    if cfg.get('peft_type')!='LORA' or set(cfg.get('target_modules',[]))!=set(c['lora_target_modules']):
        raise ValueError('初始adapter类型或target_modules不匹配')
    if Path(cfg.get('base_model_name_or_path','')).name!=Path(c['policy_model']).name:
        raise ValueError('初始adapter的基座模型不匹配')
    if state.get('global_step')!=c['initial_checkpoint_step']:
        raise ValueError('来源checkpoint实际步数不匹配')
    weights=path/'adapter_model.safetensors';size=weights.stat().st_size
    with weights.open('rb') as f:
        prefix=f.read(8)
        if len(prefix)!=8:raise ValueError('初始权重缺少safetensors头')
        count=struct.unpack('<Q',prefix)[0]
        if not 0<count<=min(size-8,8*1024*1024):raise ValueError('初始权重头长度错误')
        header=json.loads(f.read(count))
    tensors={k:v for k,v in header.items() if k!='__metadata__'}
    layers=set()
    for name,value in tensors.items():
        match=re.search(r'\.layers\.(\d+)\.',name)
        if match:layers.add(int(match.group(1)))
        shape=value['shape'];a,b=value['data_offsets']
        valid=(len(shape)==2 and (('.lora_A.' in name and shape[0]==c['lora_rank'])
            or ('.lora_B.' in name and shape[1]==c['lora_rank'])))
        if not valid or not 0<=a<b<=size-8-count:raise ValueError('LoRA张量维度或文件范围错误：'+name)
        if any(word in name for word in ('visual','aligner')):raise ValueError('初始adapter包含不应训练的视觉参数')
    if len(tensors)!=160 or layers!=set(range(64)):
        raise ValueError('初始adapter应覆盖64语言层、160个LoRA张量')
    return {'status':'passed','checkpoint':str(path),'source_global_step':state['global_step'],
        'tensor_count':len(tensors),'adapter_bytes':size,'rank':cfg['r'],'alpha':cfg['lora_alpha'],
        'target_modules':sorted(cfg['target_modules']),'training_state_restored':False,
        'boundary':'只读元数据及文件范围检查；真实权重一致性在训练启动时验证'}

def check_new_training_state(trainer,state):
    if state.global_step!=0 or len(trainer.optimizer.state)!=0 or trainer.lr_scheduler.last_epoch not in (-1,0):
        raise ValueError('新阶段意外恢复了旧优化器步数或scheduler')
    if trainer.ref_adapter_name!='ref_adapter':raise ValueError('GRPO未绑定冻结的348 reference adapter')

def verify_loaded_initial_adapter(model,c,trainer,state):
    """在首次更新前比较policy及冻结reference的全部LoRA权重。"""
    if c.get('initialization_mode')!='adapter_warm_start':return None
    validate_initialization(c);check_new_training_state(trainer,state)
    report=validate_adapter_files(c['initial_adapter'],c)
    import torch
    from safetensors.torch import load_file
    source=load_file(str(Path(c['initial_adapter'])/'adapter_model.safetensors'),device='cpu')
    found={'policy':{},'reference':{}}
    for name,value in model.named_parameters():
        if '.lora_A.' not in name and '.lora_B.' not in name:continue
        role='reference' if '.ref_adapter.' in name else 'policy'
        if role=='policy' and not value.requires_grad:continue
        canonical=re.sub(r'\.lora_([AB])\.[^.]+\.weight$',r'.lora_\1.weight',name.removeprefix('module.'))
        if role=='reference' and value.requires_grad:raise ValueError('KL reference的LoRA不应可训练')
        found[role][canonical]=value
    for role,values in found.items():
        if set(values)!=set(source):raise ValueError(role+'初始LoRA张量集合不匹配')
        if not all(torch.equal(value.detach().cpu(),source[name].to(dtype=value.dtype)) for name,value in values.items()):
            raise ValueError(role+'未实际加载指定初始adapter权重')
    return {**report,'status':'passed','policy_weights_match':True,'reference_weights_match':True,
        'reference_frozen':True,'global_step':0,'optimizer_state_entries':0,
        'scheduler_last_epoch':trainer.lr_scheduler.last_epoch,'boundary':'真实初值检查；不是训练质量结论'}
