"""正式GRPO帧VQA：全131题×三条件，沿用仓库提示词、max512与batch4。"""
import argparse
import importlib.metadata
import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpo_inputs as core
from resume_results import load_resume


def validate_report(report,items,frames,header,config,cache):
    expected={(i['id'],s,i['correct'],tuple(core.build_input(i,frames[i['id']],s,header)[1]))
              for i in items for s in core.SETTINGS}
    actual={(r['id'],r['setting'],r['gold'],tuple(r['frame_paths'])) for r in report.get('records',[])}
    if (report.get('status')!='PASS' or len(report.get('records',[]))!=393 or expected!=actual
        or report.get('dataset_revision')!=config['dataset_revision']
        or report.get('model_revision')!=config['model_revision'] or not report.get('media_checked')
        or report.get('unique_frames')!=89255 or Path(report['cache_root']).resolve()!=cache.resolve()):
        raise ValueError('CPU检查未完整通过或不属于本次393条原GRPO帧输入')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--cpu-report',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--resume-from',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    config=json.loads((root/'formal_config.json').read_text(encoding='utf-8'))
    items,frames,header=core.load_inputs(root)
    cpu_report=json.loads(args.cpu_report.read_text(encoding='utf-8'))
    validate_report(cpu_report,items,frames,header,config,args.cache)
    previous,notes=load_resume(args.resume_from,items,frames,header,config)
    args.out.mkdir(parents=True,exist_ok=False)
    started=time.time();job=os.environ.get('SLURM_JOB_ID')
    groups={s:list(previous[s].values()) for s in core.SETTINGS}
    state={'status':'STARTING','slurm_job_id':job,'expected_predictions':393,
           'completed_predictions':sum(map(len,groups.values())),'resume_from':str(args.resume_from) if args.resume_from else None}
    core.write_json(args.out/'run_status.json',state)
    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForCausalLM,AutoTokenizer
        from model_adapter import ImageAdapter
        from image_batching import collate_images,enable_image_batching
        torch.set_num_threads(2);torch.manual_seed(0)
        if not torch.cuda.is_available():raise RuntimeError('正式任务必须运行在CUDA GPU上')
        adapters=[ImageAdapter(args.model) for _ in range(4)]
        adapter=adapters[0]
        # 预取线程编码、主线程解码使用不同分词器实例，避免Rust分词器并发借用冲突。
        decoder=AutoTokenizer.from_pretrained(str(args.model),local_files_only=True)
        model,loading=AutoModelForCausalLM.from_pretrained(str(args.model),trust_remote_code=True,local_files_only=True,
                   torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='flash_attention_2',
                   output_loading_info=True)
        core.write_json(args.out/'model_loading.json',loading)
        if loading.get('missing_keys') or loading.get('mismatched_keys') or loading.get('error_msgs'):
            raise ValueError('模型权重未完整加载，不能将随机初始化参数用于正式评测')
        model.eval()
        batching=enable_image_batching(model)
        # 每个模型输入均为独立图片；官方图片压缩掩码恒真，绕过恒等压缩保持原batch4。
        batch_size=4
        runtime={**config,'slurm_job_id':job,'effective_batch_size':batch_size,'image_batching':batching,
                 'actual_packages':{p:importlib.metadata.version(p) for p in ['torch','transformers','flash-attn']},
                 'gpu':torch.cuda.get_device_name(0),'resume_notes':notes,
                 'cpu_prepare_workers':4,'num_logits_to_keep':1,
                 'cpu_report':str(args.cpu_report.resolve()),'frame_cache':str(args.cache.resolve())}
        core.write_json(args.out/'run_config.json',runtime)
        traced={};original_prepare=model.prepare_inputs_labels_for_multimodal
        def traced_prepare(*positional,**kwargs):
            result=original_prepare(*positional,**kwargs)
            if result[4] is not None:
                length=int(result[4].shape[-2])
                if length+512>adapter.native_context:raise ValueError('GPU实际上下文超限')
                traced['padded_context']=length
                traced['per_row_context']=result[1].sum(dim=-1).tolist()
            return result
        model.prepare_inputs_labels_for_multimodal=traced_prepare

        def prepare_example(item,setting,slot):
            users,paths,conversation=core.build_input(item,frames[item['id']],setting,header)
            images=[]
            for relative in paths:
                with Image.open(core.cache_path(args.cache,relative)) as image:images.append(image.convert('RGB'))
            inputs,audit=adapters[slot].prepare(conversation,images)
            return item,users,paths,inputs,audit

        pad=decoder.pad_token_id
        if pad is None:pad=decoder.eos_token_id

        state['status']='RUNNING'
        for setting in core.SETTINGS:
            remaining=[i for i in items if i['id'] not in previous[setting]]
            batches=[remaining[n:n+batch_size] for n in range(0,len(remaining),batch_size)]
            with (args.out/f'{setting}_predictions.jsonl').open('w',encoding='utf-8') as stream:
                for row in groups[setting]:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                stream.flush()
                with ThreadPoolExecutor(max_workers=4) as pool:
                    pending=[pool.submit(prepare_example,item,setting,slot) for slot,item in enumerate(batches[0])] if batches else []
                    for index,batch in enumerate(batches):
                        if time.time()-started>6600:
                            state.update(status='PARTIAL_TIME_BUDGET',elapsed_seconds=time.time()-started,
                                         completed_predictions=sum(map(len,groups.values())))
                            core.write_json(args.out/'run_status.json',state)
                            print('PARTIAL_TIME_BUDGET 已保存完整预测，后续只续跑未完成题目',flush=True)
                            return
                        prepared=[future.result() for future in pending]
                        inputs=collate_images([sample[3] for sample in prepared],pad)
                        if index+1<len(batches):
                            pending=[pool.submit(prepare_example,item,setting,slot) for slot,item in enumerate(batches[index+1])]
                        tick=time.time()
                        gpu={k:v.to('cuda:0',dtype=torch.bfloat16) if k=='pixel_values'
                             else v.to('cuda:0') if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
                        traced.clear()
                        with torch.inference_mode():
                            outputs=model.generate(**gpu,do_sample=False,max_new_tokens=512,return_dict_in_generate=True,
                                use_cache=True,temperature=None,top_p=None,top_k=None,repetition_penalty=1.0,
                                num_logits_to_keep=1)
                        if traced.get('per_row_context')!=[p[4]['prompt_tokens'] for p in prepared]:
                            raise ValueError('GPU实际输入长度与逐题CPU记录不符')
                        # 官方inputs_embeds生成仅返回新token，禁止按prompt长度截取。
                        for sample,sequence in zip(prepared,outputs.sequences):
                            item,users,paths,_,audit=sample
                            raw=decoder.decode(sequence,skip_special_tokens=True).strip()
                            pred=core.parse_letter(raw)
                            row={'id':item['id'],'setting':setting,'users':users,'frame_paths':paths,
                                 'source':item['source'],'evidence_users_selection':item['evidence_users_selection'],
                                 'n_images':len(paths),'speaker':item['speaker'],'raw':raw,'pred':pred,
                                 'reason':core.parse_reason(raw),'correct':item['correct'],'is_correct':pred==item['correct'],
                                 'parse_status':'valid' if pred else 'unparsed','option_order':item.get('option_order'),
                                 'model':config['model'],'model_revision':config['model_revision'],'origin_job_id':job,
                                 'batch_elapsed_seconds':time.time()-tick,'actual_padded_context':traced['padded_context'],
                                 **audit}
                            groups[setting].append(row);stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                        stream.flush()
                        state.update(condition=setting,condition_completed=len(groups[setting]),
                            completed_predictions=sum(map(len,groups.values())),elapsed_seconds=time.time()-started)
                        core.write_json(args.out/'run_status.json',state)
                        print(f'{setting} {len(groups[setting])}/131 correct={sum(r["is_correct"] for r in groups[setting])}'
                              f' batch_seconds={time.time()-tick:.2f}',flush=True)
                        del gpu,outputs
            core.write_json(args.out/f'{setting}_summary.json',core.reference.summarize(groups[setting]))
        summary=core.paired_summary(groups)
        core.write_json(args.out/'paired_summary.json',summary)
        state.update(status='COMPLETED',elapsed_seconds=time.time()-started)
        core.write_json(args.out/'run_status.json',state)
        print('ALL_393_COMPLETED '+json.dumps(summary['comparisons']),flush=True)
    except BaseException:
        state.update(status='FAILED_OR_INTERRUPTED',error=traceback.format_exc(),elapsed_seconds=time.time()-started)
        core.write_json(args.out/'run_status.json',state)
        raise


if __name__=='__main__':main()
