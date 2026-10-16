"""纯CPU核查、不同窗口长度配对与真实帧并行解码一致性；不提交任务。"""
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path('/scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929/variants/full120-ftjudge-dual-speed-v5-20261016')
SOURCE=ROOT.parent/'full120-ftjudge-dual-r64-lrpair-20261015'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('本脚本必须显式禁用GPU')
    sys.path[:0]=[str(ROOT/'multi-user'),str(ROOT)]
    from transformers import AutoProcessor
    from training.judge_sft.collator import qwen_vision_geometry
    from dual_runtime import make_length_pairs,condition_binding
    c=json.loads((ROOT/'configs/lr1e5.json').read_text())
    rows=[json.loads(s) for s in Path(c['train_dataset']).read_text().splitlines() if s.strip()]
    assert len(rows)==600 and len({condition_binding(r) for r in rows})==600
    processor=AutoProcessor.from_pretrained(c['policy_model'],local_files_only=True,trust_remote_code=True)
    geometry=qwen_vision_geometry(processor)
    per_image=math.ceil(c['max_pixels']/geometry['merged_token_pixel_area'])+8
    costs={}
    for row in rows:
        text=row['messages'][0]['content'].replace('<image>','')
        costs[condition_binding(row)]=len(processor.tokenizer.encode(text,add_special_tokens=False))+len(row['images'])*per_image+256+c['max_completion_length']
    ordered=make_length_pairs(rows,lambda r:costs[condition_binding(r)],seed=42)
    assert {condition_binding(r) for r in ordered}==set(costs)
    assert all(condition_binding(ordered[i])[0]!=condition_binding(ordered[i+1])[0] for i in range(0,600,2))
    audit=json.loads((SOURCE/'preflight/dual_sampler_audit.json').read_text())
    definition=json.loads(Path(audit['definition_path']).read_text())
    by_key={condition_binding(r):r for r in definition['train']}
    assert set(by_key)==set(costs)
    definition['train']=[by_key[condition_binding(row)] for row in ordered]
    (ROOT/'condition_definition.json').write_text(json.dumps(definition,indent=2))
    train=ROOT/'data/train_length_paired.jsonl'
    train.write_text(''.join(json.dumps(row)+'\n' for row in ordered))
    from training.grpo_v3.six_user_binary.launch import swift_command
    for name in ['lr1e5','lr5e6']:
        p=ROOT/'configs'/(name+'.json');config=json.loads(p.read_text())
        config.update(train_dataset=str(train),dataset_shuffle=False,source_train_dataset=c['train_dataset'])
        command=swift_command({**config,'policy_gpu_indices':[0],'judge_gpu_indices':[0]},str(ROOT/'outputs'/name))
        assert command[command.index('--dataset_shuffle')+1]=='false'
        p.write_text(json.dumps(config,indent=2))
    def stats(values):
        pairs=[(costs[condition_binding(values[i])],costs[condition_binding(values[i+1])]) for i in range(0,600,2)]
        return {'sum_pair_max_tokens':sum(max(a,b) for a,b in pairs),'sum_pair_length_gap':sum(abs(a-b) for a,b in pairs),
                'rank_token_sums':[sum(p[i] for p in pairs) for i in (0,1)]}
    before,after=stats(rows),stats(ordered)
    from decode_pipeline import DecodePipeline
    from qwen_vl_utils import process_vision_info
    contents=[{'type':'image','image':path,'min_pixels':3136,'max_pixels':c['max_pixels']} for path in rows[0]['images'][:8]]
    def decode(chunk):return list(process_vision_info([{'role':'user','content':chunk}],image_patch_size=geometry['patch_size'])[0])
    start=time.monotonic();serial=decode(contents);serial_seconds=time.monotonic()-start
    pool=DecodePipeline(workers=4,chunk_size=2)
    try:
        start=time.monotonic();parallel=pool.decode(contents,decode);parallel_seconds=time.monotonic()-start
    finally:pool.close()
    assert [(i.size,i.mode,i.tobytes()) for i in serial]==[(i.size,i.mode,i.tobytes()) for i in parallel]
    report={'status':'passed','gpu_operations':0,'train_conditions':600,'fixed_validation_conditions':120,
            'pairs':300,'same_window_pairs':0,'geometry':geometry,'length_cost_method':'CPU conservative token upper bound, not measured GPU time',
            'before':before,'after':after,'pair_max_token_proxy_reduction':1-after['sum_pair_max_tokens']/before['sum_pair_max_tokens'],
            'decode_pixels_equal_real_frames':len(serial),'serial_cpu_seconds':serial_seconds,'parallel_cpu_seconds':parallel_seconds,
            'decode_timing_boundary':'eight-frame CPU check; warm-cache/order effects mean this is not a GPU speed claim'}
    (ROOT/'preflight/speed_data_cpu.json').write_text(json.dumps(report,indent=2))
    print('SPEED_DATA_CPU',json.dumps(report),flush=True)


if __name__=='__main__':main()
