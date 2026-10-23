"""合并真实600训练与120原验证条件，构建完整720条件单遍训练。"""
from collections import Counter
from pathlib import Path
import argparse
import json
import math
import os
import sys

from dual_runtime import condition_binding, make_length_pairs


def assemble_720(train_rows, validation_rows, definition):
    if len(train_rows)!=600 or len(validation_rows)!=120:
        raise ValueError('来源应为真实600训练和120原验证条件')
    rows=[]
    for split,source in [('train',train_rows),('validation',validation_rows)]:
        rows.extend({**row,'original_split':split} for row in source)
    bindings=[condition_binding(row) for row in rows]
    if len(set(bindings))!=720: raise ValueError('720条件存在规范化主键重复或缺失')
    windows=Counter(key[0] for key in bindings)
    speakers=Counter(key[1] for key in bindings)
    if len(windows)!=120 or set(windows.values())!={6} or speakers!=Counter({i:120 for i in range(6)}):
        raise ValueError('必须为120窗口各6speaker且每speaker120条件')
    metadata={}
    for split in ['train','validation']:
        for item in definition[split]:
            key=condition_binding(item)
            if key in metadata: raise ValueError('来源定义条件重复')
            metadata[key]={**item,'original_split':split,'split':'train'}
    if set(metadata)!=set(bindings): raise ValueError('实际媒体条件与来源定义未逐条对齐')
    for row in rows:
        key=condition_binding(row)
        if len(row['required_users'])!=6 or metadata[key]['original_split']!=row['original_split']:
            raise ValueError('来源speaker/六视角或原split错位')
        metadata[key].update(source_packet_id=row['source_packet_id'],speaker_index=int(row['asker_index']),window_id=key[0])
    return rows,metadata


def verify_reused_conditions(actual,expected):
    actual_map={condition_binding(row):row for row in actual}
    expected_map={condition_binding(row):row for row in expected}
    if (len(actual_map)!=len(actual) or len(expected_map)!=len(expected)
            or actual_map!=expected_map):
        raise ValueError('复用媒体核验要求全部条件及原始输入字段逐项一致')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='': raise RuntimeError('数据构建必须显式禁用GPU')
    c=json.loads(args.config.read_text());root=Path(c['project_root'])
    sys.path[:0]=[str(root/'multi-user'),str(root)]
    load_rows=lambda path:[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    source_definition=json.loads(Path(c['source_condition_definition']).read_text())
    rows,metadata=assemble_720(load_rows(c['source_train_dataset']),load_rows(c['source_validation_dataset']),source_definition)
    from training.grpo_v3.six_user_binary.data import validate_row
    for index,row in enumerate(rows):
        validate_row(row)
        if index%120==0: print('VALIDATED_CONDITION',index,flush=True)
    from transformers import AutoProcessor
    from training.judge_sft.collator import qwen_vision_geometry
    processor=AutoProcessor.from_pretrained(c['policy_model'],local_files_only=True,trust_remote_code=True)
    geometry=qwen_vision_geometry(processor)
    per_image=math.ceil(c['max_pixels']/geometry['merged_token_pixel_area'])+8
    costs={condition_binding(row):len(processor.tokenizer.encode(row['messages'][0]['content'].replace('<image>',''),
                add_special_tokens=False))+len(row['images'])*per_image+256+c['max_completion_length'] for row in rows}
    ordered=make_length_pairs(rows,lambda row:costs[condition_binding(row)],seed=42)
    assert len(ordered)==720 and {condition_binding(row) for row in ordered}==set(costs)
    assert all(condition_binding(ordered[i])[0]!=condition_binding(ordered[i+1])[0] for i in range(0,720,2))
    definition={'train':[metadata[condition_binding(row)] for row in ordered],'validation':[],
                'scope':'720全部训练，original_split仅保留来源，不再代表held-out'}
    (root/'condition_definition.json').write_text(json.dumps(definition,ensure_ascii=False,indent=2))
    destination=root/'data/train_length_paired.jsonl'
    destination.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in ordered))
    c.update(train_dataset=str(destination),dataset_shuffle=False,val_dataset=None,eval_strategy='no',
             train_conditions=720,validation_conditions=0,max_steps=360,stop_after_steps=360)
    args.config.write_text(json.dumps(c,ensure_ascii=False,indent=2))
    from training.grpo_v3.six_user_binary.launch import swift_command
    command=swift_command({**c,'policy_gpu_indices':[0],'judge_gpu_indices':[0]},str(root/'outputs/lr1e5/CPU_CLI_REVIEW'))
    assert '--val_dataset' not in command and '--resume_from_checkpoint' not in command
    assert command[command.index('--eval_strategy')+1]=='no'
    (root/'preflight/train_command_cpu.json').write_text(json.dumps(command,indent=2))
    report={'status':'passed','validation_job_id':os.environ.get('SLURM_JOB_ID'),'train_conditions':720,'validation_conditions':0,'windows':120,
            'conditions_per_speaker':dict(Counter(row['asker_index'] for row in ordered)),
            'pairs':360,'same_window_pairs':0,'validated_real_rows':720,
            'geometry':geometry,'cost_method':'真实tokenizer及图像数的CPU长度上界，不是GPU实测',
            'rank_token_cost_sums':[sum(costs[condition_binding(row)] for row in ordered[rank::2]) for rank in range(2)]}
    (root/'preflight/data_720_cpu.json').write_text(json.dumps(report,indent=2))
    print('DATA_720_CPU_PASSED',json.dumps(report),flush=True)


if __name__=='__main__': main()
