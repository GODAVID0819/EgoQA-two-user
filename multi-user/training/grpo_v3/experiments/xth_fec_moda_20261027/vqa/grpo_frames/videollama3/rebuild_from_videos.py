"""从原始EgoLife片段重建同一套156帧缓存；保留题集给出的剪枝索引。"""
import argparse
import io
import json
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path,PurePosixPath

import numpy as np
from PIL import Image
import grpo_inputs as core
from reference_decode import _decode_ffmpeg_ppm_stream

local=threading.local()
source_locks={}
source_locks_guard=threading.Lock()


def build_plan(items):
    groups={}
    for item in items:
        for video in item['videos']:
            segments={int(s['segment_index']):s for s in video['segments']}
            key=f'{item["packet_id"]}/{video["agent_dir"]}'
            ordered=[segments[i]['video_url'].split('/resolve/main/',1)[1] for i in range(20)]
            for raw in ordered:
                path=PurePosixPath(raw)
                if path.is_absolute() or '..' in path.parts or ':' in raw or '\\' in raw:
                    raise ValueError('原视频路径越界')
            task=groups.setdefault(key,{'segments':ordered,'targets':{}})
            if task['segments']!=ordered:raise ValueError('同一窗口的原片段顺序不一致')
            for relative in item['frames'][video['agent_name']]:
                match=re.fullmatch(r'frame_(\d+)_(\d+\.\d+)s\.jpg',PurePosixPath(relative).name)
                if not match:raise ValueError('无法解析原帧号和时间')
                index,seconds=int(match[1]),float(match[2])
                if abs(seconds-index*2)>1e-7:raise ValueError('帧号与原0.5FPS采样时间不一致')
                segment_index=int(seconds//30);segment=segments[segment_index]
                offset=seconds-float(segment['window_start_seconds'])
                if offset<0 or offset>=30 or abs(offset/2-round(offset/2))>1e-7:
                    raise ValueError('原片段与帧时间无法精确对齐')
                raw=segment['video_url'].split('/resolve/main/',1)[1]
                path=PurePosixPath(raw)
                if path.is_absolute() or '..' in path.parts:raise ValueError('源视频路径越界')
                targets=task['targets']
                if relative in targets and targets[relative]!=index:raise ValueError('同帧映射冲突')
                targets[relative]=index
    return groups


def window_command(videos,ffmpeg):
    # 保留原20段拼接后连续抽帧，不能独立抽每段；短片段会影响concat时间轴。
    inputs=[];filters=[];labels=[]
    for i,video in enumerate(videos):
        inputs+=['-threads','1','-i',str(video)]
        filters.append(f'[{i}:v]setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration=30.000000,trim=duration=30.000000[v{i}]')
        labels.append(f'[v{i}]')
    filters.append(''.join(labels)+f'concat=n={len(videos)}:v=1:a=0,fps=0.5:start_time=0,format=rgb24[outv]')
    return [ffmpeg,'-hide_banner','-loglevel','error','-filter_threads','1','-filter_complex_threads','1',
            *inputs,'-filter_complex',';'.join(filters),'-map','[outv]','-t',f'{30*len(videos):.6f}',
            '-frames:v',str(15*len(videos)),'-threads','1','-f','image2pipe','-vcodec','ppm','pipe:1']


def cached_ok(path):
    try:
        with Image.open(path) as image:
            image.load()
            return image.mode=='RGB' and image.size==(156,156)
    except (OSError,ValueError):return False


def get_source(relative,sizes,original_root,cache_root,revision):
    with source_locks_guard:
        lock=source_locks.setdefault(relative,threading.Lock())
    with lock:
        return get_source_locked(relative,sizes,original_root,cache_root,revision)


def get_source_locked(relative,sizes,original_root,cache_root,revision):
    expected=sizes[relative]
    for root in [original_root,cache_root]:
        path=root/relative
        if path.is_file() and path.stat().st_size==expected:return path,False
    import requests
    if not hasattr(local,'session'):local.session=requests.Session()
    target=cache_root/relative;target.parent.mkdir(parents=True,exist_ok=True)
    url=f'https://huggingface.co/datasets/lmms-lab/EgoLife/resolve/{revision}/{relative}'
    failure=None
    for attempt in range(5):
        try:
            with local.session.get(url,stream=True,timeout=(20,120)) as response:
                if response.status_code in [429,500,502,503,504]:
                    reset=re.search(r't=(\d+)',response.headers.get('RateLimit',''))
                    delay=float(response.headers.get('Retry-After',reset[1] if reset else 2**(attempt+2)))
                    time.sleep(max(1,delay));continue
                response.raise_for_status()
                temporary=target.with_suffix('.partial.mp4')
                with temporary.open('wb') as stream:
                    for chunk in response.iter_content(chunk_size=1024*1024):stream.write(chunk)
            if temporary.stat().st_size!=expected:raise ValueError('原视频下载不完整或版本不符')
            temporary.replace(target)
            return target,True
        except Exception as exc:
            failure=exc
            if attempt<4:time.sleep(2**attempt)
    raise RuntimeError(f'源视频下载失败：{relative} {failure}')


def process_window(key,task,args,size_info):
    targets=task['targets']
    missing=[p for p in targets if not cached_ok(core.cache_path(args.cache,p))]
    if not missing:return {'new_frames':0,'existing_frames':len(targets),'downloaded_video':False,'skipped':True}
    sources=[];downloaded=0
    for relative in task['segments']:
        source,was_downloaded=get_source(relative,size_info['files'],args.original_videos,args.video_cache,size_info['revision'])
        sources.append(source);downloaded+=was_downloaded
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:raise RuntimeError('缺少FFmpeg')
    images=_decode_ffmpeg_ppm_stream(window_command(sources,ffmpeg),expected_frame_count=300,source_description=key)
    if len(images)!=300 or any(image.size!=(1408,1408) for image in images):
        raise ValueError('原视频解码帧数或1408尺寸不一致')
    small={}
    for i in set(targets.values()):
        memory=io.BytesIO();images[i].save(memory,format='JPEG',quality=95,subsampling=0);memory.seek(0)
        with Image.open(memory) as encoded:
            small[i]=core.reference.rescale_image(encoded.convert('RGB'),24576)
    compared=0
    for path,index in targets.items():
        cached=core.cache_path(args.cache,path)
        if cached.is_file() and cached_ok(cached):
            with Image.open(cached) as image:
                if not np.array_equal(np.asarray(image),np.asarray(small[index])):
                    raise ValueError(f'重建像素与已下载参考帧不一致：{path}')
            compared+=1
    for path in missing:
        target=core.cache_path(args.cache,path);target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_suffix('.rebuild_partial.png')
        small[targets[path]].save(temporary,format='PNG');temporary.replace(target)
    return {'new_frames':len(missing),'existing_frames':len(targets)-len(missing),
            'downloaded_video':downloaded,'reference_frames_compared':compared,'skipped':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--video-cache',type=Path,required=True)
    import getpass
    parser.add_argument('--original-videos',type=Path,default=Path('/scratch')/getpass.getuser()/'datasets/EgoLife')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    # 在启动线程前完成torchvision导入，避免每个线程冷启动。
    core.reference.rescale_image(Image.new('RGB',(1408,1408)),24576)
    root=Path(__file__).resolve().parent
    full=json.loads((root/'dataset.json').read_text(encoding='utf-8'))['items']
    groups=build_plan(full)
    info=json.loads((root/'original_video_sizes.json').read_text(encoding='utf-8'))
    if {p for task in groups.values() for p in task['segments']}!=set(info['files']):raise ValueError('原视频清单与题集不符')
    if sum(len(task['targets']) for task in groups.values())!=89255:raise ValueError('重建目标不是原89255张唯一帧')
    completed=created=reused=downloaded=compared=0;errors=[];started=time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(process_window,key,task,args,info):key for key,task in groups.items()}
        for future in as_completed(futures):
            try:
                value=future.result();created+=value['new_frames'];reused+=value['existing_frames']
                downloaded+=value['downloaded_video'];compared+=value.get('reference_frames_compared',0)
            except Exception as exc:errors.append({'video':futures[future],'error':str(exc)})
            completed+=1
            if completed%10==0 or completed==len(groups):
                progress={'completed_windows':completed,'expected_windows':len(groups),'created_frames':created,
                          'reused_frames':reused,'downloaded_videos':downloaded,'compared_reference_frames':compared,
                          'errors':len(errors),'elapsed_seconds':time.time()-started}
                core.write_json(args.report.with_name('rebuild_progress.json'),progress)
                print('REBUILD_PROGRESS '+json.dumps(progress),flush=True)
    result={'status':'PASS' if not errors else 'FAIL','gpu_used':False,'source_revision':info['revision'],
            'created_frames':created,'reused_frames':reused,'unique_frames':created+reused,
            'reference_frames_compared':compared,'errors':errors,'elapsed_seconds':time.time()-started}
    core.write_json(args.report,result)
    if errors:raise SystemExit(2)


if __name__=='__main__':main()
