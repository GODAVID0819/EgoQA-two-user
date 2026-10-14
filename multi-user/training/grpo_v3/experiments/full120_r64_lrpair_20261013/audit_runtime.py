"""在CPU作业核对真实TRL重复采样、r64模块覆盖与启动参数。"""
import argparse
import collections
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root.resolve()
    import torch
    from trl.trainer.utils import RepeatSampler
    from trl import GRPOConfig
    from pair_data import make_order
    definition=json.loads((root/'condition_definition.json').read_text())
    ordered=definition['train'];assert ordered==make_order(ordered)
    train_args=GRPOConfig(output_dir=str(root/'preflight/args'),per_device_train_batch_size=1,
        gradient_accumulation_steps=16,num_generations=4,max_steps=300,use_cpu=True,bf16=False,report_to='none',scale_rewards='group')
    assert train_args.steps_per_generation==16 and train_args.generation_batch_size==16
    sampler=RepeatSampler(ordered,mini_repeat_count=4,batch_size=4,repeat_count=16,shuffle=False,seed=42)
    indexes=list(sampler);assert len(indexes)==38400
    visits=collections.Counter()
    for start in range(0,len(indexes),256):
        batch=indexes[start:start+16]
        assert all(indexes[start+j*16:start+(j+1)*16]==batch for j in range(16))
        unique=[batch[j] for j in range(0,16,4)]
        assert len({ordered[i]['source_packet_id'] for i in unique})==4
        assert all(len(set(batch[j:j+4]))==1 for j in range(0,16,4))
        visits.update(unique)
    assert set(visits.values())=={1} and len(visits)==600
    configs=[json.loads((root/'configs'/name).read_text()) for name in ['lr1e5.json','lr5e6.json']]
    from training.grpo_v3.six_user_binary.launch import swift_command
    for config in configs:
        command=swift_command(config,str(root/'preflight/command'))
        def value(key):return command[command.index('--'+key)+1]
        assert value('dataset_shuffle')=='false' and value('gradient_accumulation_steps')=='16'
        assert value('lora_rank')=='64' and value('lora_alpha')=='128' and value('vllm_max_lora_rank')=='64'
        assert value('max_steps')=='300'
    differences={k for k in set(configs[0])|set(configs[1]) if configs[0].get(k)!=configs[1].get(k)}
    assert differences=={'learning_rate','output_root','run_attempt'}
    from training.grpo_v3.six_user_binary.data import read_rows
    from training.grpo_v3.six_user_binary.evaluation import policy_template
    from training.judge_sft.collator import qwen_vision_geometry
    import math
    template=policy_template(configs[0]);geometry=qwen_vision_geometry(template.processor)
    per_image=math.ceil(configs[0]['max_pixels']/geometry['merged_token_pixel_area'])+8
    bounds=[]
    for split in ['train','validation']:
        for row in read_rows(root/'data/trainval'/(split+'.jsonl')):
            content=row['messages'][0]['content'].replace('<image>','')
            text_tokens=len(template.processor.tokenizer.encode(content,add_special_tokens=False))
            bound=text_tokens+len(row['images'])*per_image+256+configs[0]['max_completion_length']
            if bound>configs[0]['max_length']:raise ValueError('保守输入长度超限：'+row['evidence_id']+' '+str(bound))
            bounds.append(bound)
    assert len(bounds)==720
    from transformers import AutoConfig,AutoModelForImageTextToText
    from accelerate import init_empty_weights
    from peft import LoraConfig,get_peft_model,TaskType
    config=AutoConfig.from_pretrained(configs[0]['policy_model'],local_files_only=True)
    with init_empty_weights():
        model=AutoModelForImageTextToText.from_config(config,attn_implementation='sdpa')
        model=get_peft_model(model,LoraConfig(r=64,lora_alpha=128,lora_dropout=.05,
            target_modules=['q_proj','v_proj','in_proj_qkv'],task_type=TaskType.CAUSAL_LM))
    modules=[(name,m) for name,m in model.named_modules() if hasattr(m,'lora_A') and len(m.lora_A)]
    import re
    layers={int(re.search(r'\.layers\.(\d+)\.',name).group(1)) for name,m in modules}
    assert layers==set(range(64)) and len(modules)==80
    assert not any('.visual.' in name for name,m in modules)
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    result={'status':'passed','job_id':os.environ['SLURM_JOB_ID'],'num_generations':4,'windows_per_update':4,
        'generation_batch_size':train_args.generation_batch_size,'steps_per_generation':train_args.steps_per_generation,
        'optimizer_updates_per_epoch':150,'total_updates':300,'visits_per_condition':2,
        'lora_rank':64,'lora_alpha':128,'lora_modules':len(modules),'language_layers':len(layers),'trainable_parameters':trainable,
        'learning_rates':[c['learning_rate'] for c in configs],'allowed_config_differences':sorted(differences),
        'checked_length_bounds':len(bounds),'maximum_input_bound_including_completion':max(bounds),
        'gpu_forward_or_adapter_sync_tested':False}
    (root/'preflight/runtime_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('RUNTIME_AUDIT_PASSED',json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
