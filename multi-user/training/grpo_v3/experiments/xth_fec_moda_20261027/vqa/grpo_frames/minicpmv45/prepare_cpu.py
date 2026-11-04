"""无权重的完整输入核对，并验证官方处理器的图片与压缩接口。"""
import json,os,time
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor
from grpo_inputs import load_inputs,build_input,SETTINGS,write_json,cache_path
from packing import pack_users
from native_inputs import text_inputs,make_prompt

def records(root):
    items,store,header=load_inputs(root)
    result=[];groups=[];group_keys={}
    for item in items:
        fr=store[item['id']]
        for setting in SETTINGS:
            users,paths,messages=build_input(item,fr,setting,header)
            packed=pack_users(users,fr['frames'])
            for group in packed:
                key=tuple(group['paths'])
                if key not in group_keys:
                    group_keys[key]=len(groups);groups.append(group)
                group['cache_id']=group_keys[key]
            result.append({'id':item['id'],'setting':setting,'correct':item['correct'],
                'users':users,'paths':paths,'text':messages[0]['content'][-1]['text'],
                'groups':packed})
    return result,groups

def main():
    root=Path(__file__).resolve().parent;start=time.monotonic()
    model=Path(os.environ.get('MINICPM_MODEL',root/'upstream'))
    processor=AutoProcessor.from_pretrained(model,trust_remote_code=True)
    requests,groups=records(root)
    lengths=[]
    for request in requests:
        inputs=text_inputs(processor,[request])
        lengths.append({'id':request['id'],'setting':request['setting'],
            'input_tokens':int(inputs['attention_mask'].sum()),'raw_frames':len(request['paths']),
            'visual_groups':len(request['groups']),'visual_tokens':64*len(request['groups'])})
    sample_groups=groups[:2]
    images=[Image.new('RGB',(156,156),(31,63,95)) for g in sample_groups for _ in g['paths']]
    paths=[p for g in sample_groups for p in g['paths']]
    test_record={'paths':paths,'groups':sample_groups,'text':'Answer: A\nReason: test'}
    prompt=make_prompt(processor.tokenizer,paths,test_record['text'])
    actual=processor([prompt],[images],temporal_ids=[[g['temporal_ids'] for g in sample_groups]],
        max_slice_nums=1,use_image_id=False,return_tensors='pt',max_length=None)
    fast=text_inputs(processor,[test_record])
    if not torch.equal(actual['input_ids'],fast['input_ids']):
        raise ValueError('缓存路径的文本与官方真实图片处理路径不等价')
    if len(actual['pixel_values'][0])!=len(images):raise ValueError('处理器丢失原帧')
    geometry=actual['tgt_sizes'][0].tolist()
    if any(size!=[32,32] for size in geometry):raise ValueError('实际视觉尺寸不是448方图')
    cache=os.environ.get('FRAME_CACHE')
    missing=[]
    if cache:
        unique={p for g in groups for p in g['paths']}
        missing=[p for p in unique if not cache_path(cache,p).is_file()]
        if missing:raise FileNotFoundError(f'缺少{len(missing)}帧：{missing[:3]}')
        for path in paths[:3]:
            with Image.open(cache_path(cache,path)) as image:
                if image.size!=(156,156):raise ValueError('已有帧尺寸错误')
    report={'status':'CPU_READY_NO_GPU_NO_WEIGHTS','n':len(requests),'unique_groups':len(groups),
        'unique_frames':len({p for g in groups for p in g['paths']}),
        'max_input_tokens':max(x['input_tokens'] for x in lengths),
        'max_with_output':max(x['input_tokens'] for x in lengths)+512,
        'native_context':40960,'actual_size':[448,448],'missing_frames':missing,
        'official_processor_equivalence':True,'elapsed_seconds':time.monotonic()-start,
        'inputs':lengths}
    write_json(root/'cpu_ready.json',report)
    write_json(root/'groups_manifest.json',groups)
    print(json.dumps({k:v for k,v in report.items() if k!='inputs'}),flush=True)

if __name__=='__main__':main()
