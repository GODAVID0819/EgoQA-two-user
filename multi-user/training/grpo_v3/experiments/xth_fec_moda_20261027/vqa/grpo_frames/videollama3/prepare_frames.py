"""CPU下载原生成器帧并按原实现缓存156PNG；不从视频重采样。"""
import argparse
import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from urllib.parse import quote

import grpo_inputs as core

local=threading.local()


def validate_png(path):
    from PIL import Image
    with Image.open(path) as image:
        image.load()
        if image.mode!='RGB' or image.size!=(156,156):raise ValueError('缓存必须为156×156 RGB')


def fetch_one(relative,cache_root,config):
    from PIL import Image
    import requests
    target=core.cache_path(cache_root,relative)
    if target.is_file():
        try:
            validate_png(target)
            return 'reused'
        except (OSError,ValueError):pass
    if not hasattr(local,'session'):local.session=requests.Session()
    url=f'https://huggingface.co/datasets/{config["dataset_repo"]}/resolve/{config["dataset_revision"]}/{quote(relative,safe="/")}'
    last=None
    for attempt in range(5):
        try:
            response=local.session.get(url,timeout=(20,90))
            if response.status_code in (429,500,502,503,504):
                delay=min(90,float(response.headers.get('Retry-After',2**(attempt+2))))
                response.close();time.sleep(delay);continue
            response.raise_for_status()
            with Image.open(io.BytesIO(response.content)) as image:
                image=image.convert('RGB')
                if image.size!=(1408,1408):raise ValueError(f'原始帧尺寸不是1408×1408：{relative} {image.size}')
                small=core.reference.rescale_image(image,24576)
                if small.size!=(156,156):raise ValueError('原预缩放未产生156×156')
                target.parent.mkdir(parents=True,exist_ok=True)
                temp=target.with_suffix('.partial.png')
                small.save(temp,format='PNG')
                temp.replace(target)
            return 'downloaded'
        except Exception as exc:
            last=exc
            if attempt<4:time.sleep(min(30,2**attempt))
    raise RuntimeError(f'帧下载或解码失败：{relative} {last}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--context-only',action='store_true')
    args=parser.parse_args()
    import torch
    from model_adapter import ImageAdapter
    torch.set_num_threads(2)
    root=Path(__file__).resolve().parent
    config=json.loads((root/'experiment_config.json').read_text(encoding='utf-8'))
    items,frames,header=core.load_inputs(root)
    adapter=ImageAdapter(args.model)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    records=[]
    for item in items:
        for setting in core.SETTINGS:
            users,paths,conversation=core.build_input(item,frames[item['id']],setting,header)
            _,audit=adapter.audit_prompt(conversation,len(paths))
            records.append({'id':item['id'],'setting':setting,'gold':item['correct'],
                            'users':users,'frame_paths':paths,**audit})
    report={'status':'CONTEXT_PASS_MEDIA_PENDING','dataset_revision':config['dataset_revision'],
            'model_revision':config['model_revision'],'reference_commit':config['reference_commit'],
            'n_questions':131,'n_inputs':len(records),'native_context':adapter.native_context,
            'max_total_with_generation':max(r['total_with_generation'] for r in records),
            'cache_root':str(args.cache.resolve()),'records':records}
    core.write_json(args.report,report)
    print(f'CPU_CONTEXT_PASS inputs={len(records)} max_total={report["max_total_with_generation"]}',flush=True)
    if args.context_only:return
    paths=sorted({p for item in items for group in item['frames'].values() for p in group})
    if len(paths)!=89255:raise ValueError('本次唯一帧数与已核验清单不一致')
    args.cache.mkdir(parents=True,exist_ok=True)
    source_meta=args.cache/'source.json'
    provenance={'dataset_repo':config['dataset_repo'],'dataset_revision':config['dataset_revision'],
                'preprocess_max_pixels':24576,'rescale_source_commit':config['reference_commit']}
    if source_meta.exists() and json.loads(source_meta.read_text(encoding='utf-8'))!=provenance:
        raise ValueError('缓存来源版本或预缩放实现不一致')
    core.write_json(source_meta,provenance)
    completed=downloaded=reused=0;failures=[];started=time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(fetch_one,path,args.cache,config):path for path in paths}
        for future in as_completed(futures):
            try:
                state=future.result();downloaded+=state=='downloaded';reused+=state=='reused'
            except Exception as exc:failures.append({'path':futures[future],'error':str(exc)})
            completed+=1
            if completed%1000==0 or completed==len(paths):
                progress={'completed':completed,'expected':len(paths),'downloaded':downloaded,'reused':reused,
                          'failures':len(failures),'elapsed_seconds':time.time()-started}
                core.write_json(args.cache/'download_progress.json',progress)
                print('FRAME_PROGRESS '+json.dumps(progress),flush=True)
    report.update(status='FAIL_MEDIA' if failures else 'PASS',unique_frames=len(paths),
                  downloaded=downloaded,reused=reused,media_errors=failures,media_checked=True)
    core.write_json(args.report,report)
    if failures:raise SystemExit(2)
    print('CPU_READY '+str(args.report),flush=True)


if __name__=='__main__':main()
