"""只在 CPU 上比较原始与逐帧缓存准备路径；不加载模型权重。"""
import argparse
from collections import OrderedDict
import importlib.util
import json
import os
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--before-predictor',type=Path,required=True)
    parser.add_argument('--reward-trace',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU测量必须隐藏GPU')
    report={'status':'running','started_epoch':time.time(),'checks':{}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    save()
    try:
        import torch
        torch.set_num_threads(2)
        if torch.cuda.is_available():raise RuntimeError('CPU测量意外看到GPU')
        from transformers import AutoProcessor
        from qwen_vl_utils import process_vision_info
        from training.judge_sft.collator import qwen_vision_geometry
        from .predictor import VllmJudge,FrameImageCache
        from .judge import build_examples
        from .data import read_rows
        c=json.loads(args.config.read_text())
        judge_config=json.loads(Path(c['judge_config']).read_text())
        model=judge_config['model_id']
        processor=AutoProcessor.from_pretrained(model,local_files_only=True,trust_remote_code=True)
        spec=importlib.util.spec_from_file_location('before_cache_predictor',args.before_predictor)
        before=importlib.util.module_from_spec(spec);spec.loader.exec_module(before)
        def instance(cls):
            obj=object.__new__(cls);obj.processor=processor;obj.geometry=qwen_vision_geometry(processor)
            obj.process_vision_info=process_vision_info;obj.max_tokens=judge_config['max_input_tokens']
            obj.min_pixels=judge_config['min_pixels'];obj.max_pixels=judge_config['max_pixels']
            obj.cache=OrderedDict();obj.cache_namespace='cpu_measurement'
            obj.image_cache=FrameImageCache(8*1024**3)
            return obj
        traces=[json.loads(s) for s in args.reward_trace.read_text().splitlines() if s.strip()]
        valid=next(x for x in traces if x['judge_result']['status']=='scored')
        rows=read_rows(Path(c['prepared_data_root'])/'train.jsonl')
        first=next(x for x in rows if x['evidence_id']==valid['evidence_id'])
        second=next(x for x in rows if x['source_packet_id']==first['source_packet_id'] and x['asker_index']!=first['asker_index'])
        examples=[build_examples({**row,'request_id':'cpu_'+str(i),'completion':valid['completion']})['groundedness']
                  for i,row in enumerate((first,second))]
        results={}
        for label,obj in [('before',instance(before.VllmJudge)),('after',instance(VllmJudge))]:
            for index,example in enumerate(examples):
                start=time.perf_counter();payload,audit=obj.prepare(example)
                report['checks'][f'{label}_{index}']={'seconds':time.perf_counter()-start,
                    'frames':example.frame_count,**{k:v for k,v in audit.items() if k.startswith('decoded_image_cache')}}
                if index==1:results[label]=payload['multi_modal_data']['image']
                save()
        old,new=results['before'],results['after']
        equal=len(old)==len(new) and all(a.size==b.size and a.mode==b.mode and a.tobytes()==b.tobytes() for a,b in zip(old,new))
        if not equal:raise RuntimeError('缓存改变了真实图像像素或顺序')
        report['checks']['all_image_pixels_and_order_equal']=equal
        report.update(status='passed',finished_epoch=time.time())
    except BaseException as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}',finished_epoch=time.time())
        raise
    finally:save()


if __name__=='__main__':main()
