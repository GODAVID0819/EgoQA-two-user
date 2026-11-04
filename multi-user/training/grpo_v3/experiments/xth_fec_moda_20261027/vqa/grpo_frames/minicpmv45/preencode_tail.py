"""在原作业同一GPU上提前编码远期真实帧组；不更改主进程或进行额外答题。"""
import json,os,pathlib,sys,time
import torch
from transformers import AutoProcessor
from upstream.modeling_minicpmv import MiniCPMV
from evaluate import VisionCache
from prepare_cpu import records

def main():
    root=pathlib.Path(sys.argv[1]);job=sys.argv[2];out=root/'outputs'/f'formal_{job}'
    code=root/'code';config=json.loads((out/'run_config.json').read_text())
    torch.manual_seed(0);torch.set_num_threads(2)
    requests,groups=records(code)
    processor=AutoProcessor.from_pretrained(root/'model',trust_remote_code=True)
    model,loading=MiniCPMV.from_pretrained(root/'model',torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',low_cpu_mem_usage=True,output_loading_info=True)
    assert not any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    model=model.eval().cuda()
    cache=VisionCache(model,processor,os.environ['FRAME_CACHE'],out/'vision_cache',groups)
    start=time.monotonic();completed=[];stop_reason='tail_complete'
    def safely_ahead(first):
        status=json.loads((out/'run_status.json').read_text())
        if status['status']!='RUNNING':return False
        current=requests[:min(len(requests),status['n']+8)]
        cursor=max((g['cache_id'] for r in current for g in r['groups']),default=-1)
        return first>cursor+1000
    try:
        for offset in range(max(0,len(groups)-3000),len(groups),12):
            chunk=[i for i in range(offset,min(offset+12,len(groups))) if not cache.file(i).is_file()]
            if not chunk:continue
            if not safely_ahead(min(chunk)):
                stop_reason='main_process_nearby';break
            inputs=cache.prepare(chunk).to(model.device)
            with torch.inference_mode():_,features=model.get_vllm_embedding(inputs)
            values=features[0].detach().cpu()
            assert values.shape==(len(chunk),64,model.config.hidden_size)
            # 编码期间主程序可能前进；写入前再次确认仍相隔至少1000组。
            if not safely_ahead(min(chunk)):
                stop_reason='main_process_advanced';break
            for index,value in zip(chunk,values):
                target=cache.out/f'group_{index:06d}.pt'
                temporary=target.with_suffix(f'.worker_{os.getpid()}.partial')
                torch.save(value.clone(),temporary)
                try:os.link(temporary,target)
                except FileExistsError:pass
                else:completed.append(index)
                finally:temporary.unlink()
            if len(completed)%120==0:print('TAIL_PREENCODE',len(completed),round(time.monotonic()-start,1),flush=True)
            del inputs,features,values
    finally:cache.close()
    report={'status':'COMPLETE','job_id':job,'model_revision':config['model_revision'],
        'groups':len(completed),'group_ids':completed,'elapsed_seconds':time.monotonic()-start,
        'stop_reason':stop_reason,'new_gpu_allocation':0,'shares_existing_gpu':True,
        'answers_generated':0,'preserves_main_batch_size':4}
    (out/'tail_preencode_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('TAIL_PREENCODE_COMPLETE',json.dumps({k:v for k,v in report.items() if k!='group_ids'}),flush=True)

if __name__=='__main__':main()
