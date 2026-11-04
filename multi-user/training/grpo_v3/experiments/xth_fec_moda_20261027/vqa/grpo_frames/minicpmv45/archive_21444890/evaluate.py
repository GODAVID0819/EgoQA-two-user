"""全131题三视角正式评测；官方视觉计算缓存复用，不减少原始帧。"""
import json,os,time,traceback
from pathlib import Path
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import torch
from PIL import Image
from transformers import AutoProcessor
from upstream.modeling_minicpmv import MiniCPMV
from prepare_cpu import records
from grpo_inputs import cache_path,parse_letter,parse_reason,paired_summary,write_json,SETTINGS,load_inputs
from native_inputs import text_inputs,make_prompt
from packing import generation_kwargs

class VisionCache:
    def __init__(self,model,processor,cache_root,out,groups):
        self.model=model;self.processor=processor;self.frames=Path(cache_root)
        self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True)
        self.groups=groups;self.memory=OrderedDict();self.encoded=0
        self.pool=ThreadPoolExecutor(max_workers=1)
    def file(self,index):return self.out/f'group_{index:06d}.pt'
    def load(self,index):
        if index not in self.memory:
            value=torch.load(self.file(index),map_location='cpu',weights_only=True)
            if value.shape!=(64,self.model.config.hidden_size):raise ValueError('视觉缓存尺寸错误')
            self.memory[index]=value
        self.memory.move_to_end(index)
        value=self.memory[index]
        while len(self.memory)>256:self.memory.popitem(last=False)
        return value
    def prepare(self,indexes):
        groups=[self.groups[i] for i in indexes]
        paths=[p for g in groups for p in g['paths']]
        images=[]
        for path in paths:
            with Image.open(cache_path(self.frames,path)) as im:
                if im.size!=(156,156):raise ValueError('帧尺寸与原设置不一致')
                images.append(im.convert('RGB'))
        prompt=make_prompt(self.processor.tokenizer,paths,'')
        result=self.processor([prompt],[images],temporal_ids=[[g['temporal_ids'] for g in groups]],
            max_slice_nums=1,use_image_id=False,return_tensors='pt',max_length=None)
        if len(result['pixel_values'][0])!=len(paths):raise ValueError('原帧丢失')
        if any(x!=[32,32] for x in result['tgt_sizes'][0].tolist()):raise ValueError('实际尺寸发生变化')
        result.pop('image_sizes')
        return result
    def ensure(self,indexes):
        missing=[i for i in dict.fromkeys(indexes) if not self.file(i).is_file()]
        chunks=[missing[i:i+12] for i in range(0,len(missing),12)]
        if not chunks:return
        future=self.pool.submit(self.prepare,chunks[0])
        for offset,chunk in enumerate(chunks):
            inputs=future.result()
            if offset+1<len(chunks):future=self.pool.submit(self.prepare,chunks[offset+1])
            inputs=inputs.to(self.model.device)
            with torch.inference_mode():
                _,features=self.model.get_vllm_embedding(inputs)
            values=features[0].detach().cpu()
            if values.shape!=(len(chunk),64,self.model.config.hidden_size):
                raise ValueError(f'原生压缩组数错误：{values.shape} != {len(chunk)}')
            for index,value in zip(chunk,values):
                path=self.file(index);temporary=path.with_suffix('.partial')
                torch.save(value.contiguous(),temporary);temporary.replace(path)
                self.encoded+=1
            del inputs,features,values
    def close(self):self.pool.shutdown(wait=True)

def main():
    torch.set_num_threads(4);torch.manual_seed(0)
    root=Path(os.environ['VQA_ROOT']);code=Path(__file__).resolve().parent
    config=json.loads((code/'experiment_config.json').read_text())
    ready=json.loads((code/'cpu_ready.json').read_text())
    if ready['n']!=393 or ready['max_with_output']>40960 or ready['missing_frames']:
        raise ValueError('CPU检查未通过')
    job=os.environ['SLURM_JOB_ID'];out=root/'outputs'/f'formal_{job}'
    out.mkdir(parents=True,exist_ok=False)
    config.update(job_id=job,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                  gpu_name=torch.cuda.get_device_name(0),torch_version=torch.__version__)
    write_json(out/'run_config.json',config)
    requests,groups=records(code)
    metadata={item['id']:item for item in load_inputs(code)[0]}
    processor=AutoProcessor.from_pretrained(root/'model',trust_remote_code=True)
    start=time.monotonic()
    model,loading=MiniCPMV.from_pretrained(root/'model',torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',low_cpu_mem_usage=True,
        output_loading_info=True)
    write_json(out/'model_loading.json',loading)
    if any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):
        raise ValueError('模型权重未完整加载')
    model=model.eval().cuda()
    model.llm.generation_config.do_sample=False
    model.llm.generation_config.num_beams=1
    model.llm.generation_config.repetition_penalty=1.0
    model.llm.generation_config.temperature=None
    model.llm.generation_config.top_p=None
    model.llm.generation_config.top_k=None
    print('MODEL_LOADED',time.monotonic()-start,torch.cuda.get_device_name(0),flush=True)
    cache=VisionCache(model,processor,os.environ['FRAME_CACHE'],out/'vision_cache',groups)
    predictions={s:[] for s in SETTINGS}
    streams={s:(out/f'{s}_predictions.jsonl').open('w',encoding='utf-8') for s in SETTINGS}
    try:
        for offset in range(0,len(requests),4):
            # 正常保存部分结果；不依靠Slurm强杀，预留最后10分钟。
            if time.monotonic()-start>110*60:
                write_json(out/'run_status.json',{'status':'PARTIAL_TIME_BUDGET','n':offset})
                return
            batch=requests[offset:offset+4];batch_start=time.monotonic()
            cache.ensure([g['cache_id'] for r in batch for g in r['groups']])
            inputs=text_inputs(processor,batch)
            lengths=inputs['attention_mask'].sum(dim=1).tolist()
            inputs.pop('image_sizes')
            vision=[torch.stack([cache.load(g['cache_id']) for g in r['groups']]).to(model.device)
                    for r in batch]
            inputs=inputs.to(model.device)
            with torch.inference_mode():
                tokens=model.generate(**inputs,vision_hidden_states=vision,
                    tokenizer=processor.tokenizer,decode_text=False,logits_to_keep=1,
                    **generation_kwargs())
            answers=model._decode_text(tokens,processor.tokenizer)
            for index,(request,text,length) in enumerate(zip(batch,answers,lengths)):
                prediction=parse_letter(text)
                record={'id':request['id'],'setting':request['setting'],'correct':request['correct'],
                    'source':metadata[request['id']]['source'],
                    'evidence_users_selection':metadata[request['id']]['evidence_users_selection'],
                    'users':request['users'],'frame_paths':request['paths'],
                    'raw_frame_count':len(request['paths']),'visual_groups':len(request['groups']),
                    'visual_tokens':64*len(request['groups']),'input_tokens':int(length),
                    'preprocessed_size':[156,156],'actual_processor_size':[448,448],
                    'pred':prediction,'prediction':prediction,'is_correct':prediction==request['correct'],
                    'raw':text,'reason':parse_reason(text),'parse_ok':prediction is not None,
                    'model':config['model'],'model_revision':config['model_revision'],'job_id':job,
                    'generated_tokens':int((tokens[index]!=0).sum()),
                    'batch_elapsed_seconds':time.monotonic()-batch_start}
                predictions[request['setting']].append(record)
                streams[request['setting']].write(json.dumps(record,ensure_ascii=False)+'\n')
                streams[request['setting']].flush()
            write_json(out/'run_status.json',{'status':'RUNNING','n':offset+len(batch),
                'elapsed_seconds':time.monotonic()-start,'encoded_groups':cache.encoded,
                'conditions':{s:{'n':len(p),'correct':sum(r['is_correct'] for r in p)}
                              for s,p in predictions.items()}})
            print('PROGRESS',offset+len(batch),393,'elapsed',round(time.monotonic()-start,1),
                'batch',round(time.monotonic()-batch_start,1),'cache_groups',cache.encoded,flush=True)
            del inputs,vision,tokens
        summary=paired_summary(predictions)
        write_json(out/'paired_summary.json',summary)
        write_json(out/'run_status.json',{'status':'COMPLETE_393','n':393,
            'elapsed_seconds':time.monotonic()-start,'encoded_groups':cache.encoded})
        print('COMPLETE',json.dumps(summary,ensure_ascii=False),flush=True)
    except BaseException as error:
        write_json(out/'run_status.json',{'status':'FAILED','error':repr(error),
            'traceback':traceback.format_exc(),'n':sum(map(len,predictions.values()))})
        raise
    finally:
        cache.close()
        for stream in streams.values():stream.close()

if __name__=='__main__':main()
