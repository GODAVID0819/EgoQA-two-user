"""只补齐五个新窗口，复用旧帧并构建48/6输入；训练由独立作业提交。"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback


def clock_centiseconds(token):
    if not isinstance(token,str) or len(token)!=8 or not token.isdigit():
        raise ValueError('时间标识必须是八位实际时间字段')
    h,m,s,c=int(token[:2]),int(token[2:4]),int(token[4:6]),int(token[6:])
    if not (h<24 and m<60 and s<60):raise ValueError('时间字段越界')
    return h*360000+m*6000+s*100+c


def verify_disjoint_ranges(training,validation):
    ranges=[]
    for split,windows in [('train',training),('validation',validation)]:
        for w in windows:
            start=clock_centiseconds(w['time_token']);duration=float(w['duration_seconds'])*100
            if duration<=0:raise ValueError('窗口时长必须为正')
            for day,a,b,old_split in ranges:
                if day==w['day'] and max(start,a)<min(start+duration,b):
                    raise ValueError(f'真实媒体窗口重叠：{w["day"]}/{w["time_token"]} {split}/{old_split}')
            ranges.append((w['day'],start,start+duration,split))


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)


def probe(path,ffprobe,minimum,maximum):
    p=Path(path)
    if not p.is_file() or not p.stat().st_size:return None
    try:
        result=subprocess.run([ffprobe,'-v','error','-show_entries','format=duration:stream=codec_type,width,height','-of','json',str(p)],text=True,capture_output=True,timeout=30)
        if result.returncode:return None
        d=json.loads(result.stdout);duration=float(d['format']['duration'])
        if not minimum<=duration<=maximum:return None
        if not any(x.get('codec_type')=='video' and x.get('width',0)>0 and x.get('height',0)>0 for x in d.get('streams',[])):return None
        return duration
    except (OSError,ValueError,KeyError,subprocess.TimeoutExpired):return None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--original-data',type=Path,required=True)
    parser.add_argument('--clip-model',required=True)
    parser.add_argument('--ffmpeg',required=True)
    parser.add_argument('--ffprobe',required=True)
    parser.add_argument('--work-seconds',type=int,default=6900)
    args=parser.parse_args();root=args.root.resolve();job=os.environ['SLURM_JOB_ID'];deadline=time.monotonic()+args.work_seconds
    output=root/'outputs'/('prepare_'+job);output.mkdir(parents=True,exist_ok=True)
    record={'status':'running','job_id':job,'started_epoch':time.time(),'stage':'source_validation','completed_windows':[]}
    def save():atomic_json(output/'status.json',record)
    def partial(reason):
        record.update(status='partial',reason=reason,finished_epoch=time.time());save();print('TRAIN8_PREPARATION_PARTIAL',reason,flush=True)
    save()
    try:
        source=json.loads((root/'source_manifest.json').read_text());windows=source['windows'];assert len(windows)==5
        from training.judge_sft.prepare_real_data import _repo_module
        from training.grpo_v3.six_user_binary.data import make_row,read_rows,validate_row,validate_splits
        from training.grpo_v3.six_user_binary.workflow import unique_windows
        import torch
        torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','16')))
        prep=_repo_module('rlhf_evidence_preprocessing')
        old_train=read_rows(args.original_data/'train.jsonl');old_val=read_rows(args.original_data/'validation.jsonl')
        assert len(old_train)==18 and len(old_val)==6
        old_windows=unique_windows(read_rows('/scratch/xl6775/datasets/egoqa_fps_newprompt_20260918/base_candidates.jsonl'))
        train_packets={x['source_packet_id'] for x in old_train};val_packets={x['source_packet_id'] for x in old_val}
        old_train_windows=[w for w in old_windows if prep.packet_id_for_group(w) in train_packets]
        val_windows=[w for w in old_windows if prep.packet_id_for_group(w) in val_packets]
        assert len(old_train_windows)==3 and len(val_windows)==1
        verify_disjoint_ranges(old_train_windows+windows,val_windows)
        tasks=[]
        for w in windows:
            assert len(w['clips'])==6 and len({c['agent_dir'] for c in w['clips']})==6
            for c in w['clips']:
                assert len(c['paths'])==len(c['urls'])==len(c['sizes'])==20
                assert all(not Path(p).is_absolute() and '..' not in Path(p).parts
                           and p.startswith(c['agent_dir']+'/'+w['day']+'/') for p in c['paths'])
                tasks.extend(zip(c['paths'],c['urls'],c['sizes']))
        assert len(tasks)==600 and len({p for p,_,_ in tasks})==600
        record.update(stage='downloading_segments',expected_segments=600,expected_bytes=source['expected_bytes']);save()
        def download(item):
            import requests
            rel,url,expected=item;target=root/'segments'/rel
            if time.monotonic()>deadline-180:return {'path':rel,'status':'time_budget'}
            for candidate in (Path('/scratch/xl6775/datasets/EgoLife')/rel,target):
                if candidate.is_file() and candidate.stat().st_size==expected and probe(candidate,args.ffprobe,29.5,30.5):
                    return {'path':rel,'status':'reused','local_path':str(candidate),'bytes':expected}
            target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_suffix('.download-part');started=time.monotonic()
            last=None
            for attempt in range(3):
                if time.monotonic()>deadline-180:return {'path':rel,'status':'time_budget'}
                try:
                    with requests.get(url,stream=True,timeout=(15,60)) as response:
                        response.raise_for_status()
                        with temp.open('wb') as f:
                            for block in response.iter_content(1024*1024):
                                if time.monotonic()>deadline-120:raise TimeoutError('数据准备时限即将到达')
                                if block:f.write(block)
                    if temp.stat().st_size!=expected:raise ValueError(f'文件大小不符：{temp.stat().st_size}/{expected}')
                    if not probe(temp,args.ffprobe,29.5,30.5):raise ValueError('分段媒体不可读或时长不符')
                    temp.replace(target)
                    return {'path':rel,'status':'downloaded','local_path':str(target),'bytes':expected,'elapsed_seconds':time.monotonic()-started}
                except Exception as exc:last=f'{type(exc).__name__}: {exc}';time.sleep(attempt+1)
            return {'path':rel,'status':'failed','error':last}
        downloads=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for item in pool.map(download,tasks):
                downloads.append(item)
                with (output/'download_progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(item,ensure_ascii=False)+'\n')
                if len(downloads)%20==0:print('SEGMENTS',len(downloads),'/600',flush=True)
        atomic_json(output/'segment_manifest.json',downloads)
        if any(x['status']=='time_budget' for x in downloads):partial('下载阶段预算用尽，已保存有效分段');return
        bad=[x for x in downloads if x['status']=='failed']
        if bad:raise RuntimeError(f'{len(bad)}个分段失败，详见当前JobID下载清单')
        local={x['path']:x['local_path'] for x in downloads}
        record.update(stage='stitching_videos',completed_segments=600);save();materialized=[]
        for w in windows:
            token=w['time_token'];clock=f'{token[:2]}:{token[2:4]}:{token[4:6]}.{token[6:]}'
            group={k:w[k] for k in ('day','time_token','generation_group_id','duration_seconds')};group.update(clip_clock=clock,clips=[])
            for c in w['clips']:
                if time.monotonic()>deadline-180:partial('拼接阶段预算用尽');return
                video=root/'stitched'/(w['day']+'_'+token)/(c['agent_name']+'.mp4');video.parent.mkdir(parents=True,exist_ok=True)
                if not probe(video,args.ffprobe,599,601):
                    listing=output/('concat_'+w['day']+'_'+token+'_'+c['agent_dir']+'.txt')
                    listing.write_text(''.join("file '"+local[p].replace("'","'\\''")+"'\n" for p in c['paths']),encoding='utf-8')
                    temp=video.with_name(video.stem+'.joining.mp4')
                    result=subprocess.run([args.ffmpeg,'-nostdin','-v','error','-y','-f','concat','-safe','0','-i',str(listing),'-c','copy',str(temp)],text=True,capture_output=True,timeout=600)
                    if result.returncode or not probe(temp,args.ffprobe,599,601):raise RuntimeError('六用户成片拼接失败：'+str(video)+' '+result.stderr[-1000:])
                    temp.replace(video)
                segments=[{'clip_id':Path(p).stem,'time_token':Path(p).stem.rsplit('_',1)[-1],'video_url':url,'local_video':local[p],'segment_index':i,'window_start_seconds':30.*i,'window_end_seconds':30.*(i+1)} for i,(p,url) in enumerate(zip(c['paths'],c['urls']))]
                group['clips'].append({'agent_dir':c['agent_dir'],'agent_name':c['agent_name'],'day':w['day'],'time_token':token,'clip_clock':clock,'duration_seconds':600.,'local_video':str(video),'full_local_video':str(video),'video_url':c['urls'][0],'segments':segments,'source_video_urls':c['urls']})
            group['selection']={'source_manifest':str(root/'source_manifest.json'),'stitched_paths':[c['full_local_video'] for c in group['clips']]};materialized.append(group)
        atomic_json(root/'materialized_candidates.json',materialized)
        dataset=root/'data/frames';cache=Path(os.environ['TMPDIR'])/'frame_cache';config=prep.build_preprocessing_config(clip_model_id=args.clip_model);fp=prep._ensure_dataset_metadata(dataset,config)
        encoder=prep.LazyBatchedImageEncoder(args.clip_model,device='cpu',batch_size=32);prepared=[];durations=[]
        for w in materialized:
            estimate=max([900]+durations)
            if time.monotonic()>deadline-estimate-180:partial('抽帧与CLIP阶段剩余时间不足，保留已完成packet');return
            record.update(stage='preprocessing_packet',current_window=w['generation_group_id']);save();started=time.monotonic()
            item=prep.prepare_packet(w,dataset_root=dataset,cache_dir=cache,config=config,config_fingerprint=fp,encoder=encoder,device='cpu',clip_batch_size=32,ffmpeg_binary=args.ffmpeg,materialized_packet=w,video_sample_workers=4,media_prepare_workers=4)
            prepared.append(item);durations.append(time.monotonic()-started);record['completed_windows'].append(w['generation_group_id']);save()
        prep.rebuild_dataset_index(dataset)
        profile=old_train[0].get('generation_profile');new_rows=[make_row(dataset,p['packet_id'],asker,generation_profile=profile) for p in prepared for asker in range(6)]
        train=old_train+new_rows;splits={'train':train,'validation':old_val};validate_splits(splits)
        assert len(train)==48 and len({(x['source_packet_id'],x['asker_index']) for x in train})==48 and len({x['source_packet_id'] for x in train})==8
        assert len(old_val)==6 and len({x['source_packet_id'] for x in old_val})==1
        for rows in splits.values():
            for row in rows:validate_row(row)
        target=root/'data/trainval';target.mkdir(parents=True,exist_ok=True)
        for name,rows in splits.items():
            path=target/(name+'.jsonl')
            if path.exists():raise FileExistsError('不覆盖已有数据：'+str(path))
            path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
        final={'status':'passed','cpu_job_id':job,'root':str(root),'prepared_data_root':str(target),'new_frame_root':str(dataset),'old_frame_roots':sorted({x['dataset_root'] for x in old_train+old_val}),'split_counts':{'train':48,'validation':6},'train_windows':8,'validation_windows':1,'new_windows':[w['generation_group_id'] for w in materialized],'learning_rate':1e-5,'source_manifest':str(root/'source_manifest.json'),'boundary':'媒体/数据已准备；新LoRA GPU运行仍需独立必要验证'}
        atomic_json(root/'dataset_ready.json',final);record.update(status='passed',stage='complete',finished_epoch=time.time(),dataset=final);save();print('TRAIN8_DATA_READY',json.dumps(final,ensure_ascii=False),flush=True)
    except BaseException as exc:
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),finished_epoch=time.time());save();raise


if __name__=='__main__':main()
