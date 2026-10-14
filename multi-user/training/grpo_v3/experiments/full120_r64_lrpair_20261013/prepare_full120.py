"""在CPU Slurm作业中补齐120窗口媒体与帧，并构建600/120数据。"""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from pair_data import condition_key,validate_conditions


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)


def probe(path,ffprobe,minimum,maximum):
    p=Path(path)
    if not p.is_file() or not p.stat().st_size:return False
    try:
        r=subprocess.run([ffprobe,'-v','error','-show_entries','format=duration:stream=codec_type,width,height','-of','json',str(p)],capture_output=True,text=True,timeout=30)
        if r.returncode:return False
        d=json.loads(r.stdout)
        return minimum<=float(d['format']['duration'])<=maximum and any(s.get('codec_type')=='video' and s.get('width',0)>0 for s in d.get('streams',[]))
    except (OSError,ValueError,KeyError,subprocess.TimeoutExpired):return False


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args();root=args.root.resolve();job=os.environ['SLURM_JOB_ID']
    output=root/'outputs'/('prepare_'+job);output.mkdir(parents=True,exist_ok=True)
    status={'status':'running','job_id':job,'stage':'source_validation','completed_windows':[],'started_epoch':time.time()}
    def save():atomic_json(output/'status.json',status)
    save()
    try:
        from training.judge_sft.prepare_real_data import _repo_module
        from training.grpo_v3.six_user_binary.data import make_row,validate_row
        import torch
        torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','32')))
        prep=_repo_module('rlhf_evidence_preprocessing')
        c=json.loads((root/'source_workflow.json').read_text())
        source=json.loads((root/'source_manifest.json').read_text())
        definition=json.loads((root/'condition_definition.json').read_text())
        validate_conditions(definition['train'],definition['validation'])
        config=prep.build_preprocessing_config(clip_model_id=c['clip_model'])
        dataset=root/'data/frames';fp=prep._ensure_dataset_metadata(dataset,config)
        base=Path(c['parent_project_root'])
        previous=[base/'data/frames_18719659',base/'variants/train8-data-20261007T162701Z/data/frames',base/'variants/speaker-holdout-design-20261009/data/frames',dataset]
        packets={};missing=[]
        for w in source['windows']:
            packet=prep.packet_id_for_group(w)
            for old in previous:
                packet_dir=old/'packets'/packet
                if not (packet_dir/'COMPLETE').is_file():continue
                metadata=json.loads((old/'dataset.json').read_text())
                if metadata['config_fingerprint']!=fp:raise ValueError('复用帧的配置与当前不符：'+str(old))
                prep.validate_preprocessed_packet(packet_dir)
                packets[w['source_packet_id']]={'root':str(old),'packet':packet}
                break
            else:missing.append(w)
        status.update(stage='downloading_segments',reused_windows=len(packets),missing_windows=len(missing));save()
        existing=json.loads((root/'existing_segments.json').read_text()) if (root/'existing_segments.json').is_file() else {}
        tasks={}
        for w in missing:
            for clip in w['clips']:
                for rel,url,size in zip(clip['paths'],clip['urls'],clip['sizes']):tasks[rel]=(rel,url,size)
        def download(item):
            import requests
            rel,url,size=item;target=root/'segments'/rel
            choices=[Path(existing[rel])] if rel in existing else []
            choices.extend([Path('/scratch/xl6775/datasets/EgoLife')/rel,target])
            for p in choices:
                if p.is_file() and p.stat().st_size==size and probe(p,c['ffprobe'],29.5,30.5):
                    return {'path':rel,'local_path':str(p),'bytes':size,'status':'reused'}
            target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_suffix('.download-part')
            for attempt in range(4):
                try:
                    with requests.get(url,stream=True,timeout=(20,120)) as response:
                        response.raise_for_status()
                        with temp.open('wb') as f:
                            for block in response.iter_content(1024*1024):
                                if block:f.write(block)
                    if temp.stat().st_size!=size or not probe(temp,c['ffprobe'],29.5,30.5):
                        raise ValueError('分段大小、可读性或时长不符')
                    temp.replace(target)
                    return {'path':rel,'local_path':str(target),'bytes':size,'status':'downloaded'}
                except Exception:
                    if attempt==3:raise
                    time.sleep(2*(attempt+1))
        local={};counts={'reused':0,'downloaded':0}
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for item in pool.map(download,tasks.values()):
                local[item['path']]=item['local_path'];counts[item['status']]+=1
                with (output/'segments.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(item)+'\n')
                if len(local)%60==0:
                    status.update(completed_segments=len(local),segment_counts=dict(counts));save()
                    print('SEGMENTS',len(local),'/',len(tasks),counts,flush=True)
        atomic_json(output/'segment_paths.json',local)
        encoder=prep.LazyBatchedImageEncoder(c['clip_model'],device='cpu',batch_size=32)
        durations=[]
        for w in missing:
            status.update(stage='stitching',current_window=w['generation_group_id']);save()
            started=time.monotonic();token=w['time_token'];clock=f'{token[:2]}:{token[2:4]}:{token[4:6]}.{token[6:]}'
            group={k:w[k] for k in ('day','time_token','generation_group_id','duration_seconds')};group.update(clip_clock=clock,clips=[])
            for clip in w['clips']:
                video=root/'stitched'/(w['day']+'_'+token)/(clip['agent_name']+'.mp4');video.parent.mkdir(parents=True,exist_ok=True)
                if not probe(video,c['ffprobe'],599,601):
                    listing=output/('concat_'+w['day']+'_'+token+'_'+clip['agent_dir']+'.txt')
                    listing.write_text(''.join("file '"+local[p].replace("'","'\\''")+"'\n" for p in clip['paths']))
                    temp=video.with_name(video.stem+'.joining.mp4')
                    result=subprocess.run([c['ffmpeg'],'-nostdin','-v','error','-y','-f','concat','-safe','0','-i',str(listing),'-c','copy',str(temp)],capture_output=True,text=True,timeout=600)
                    if result.returncode or not probe(temp,c['ffprobe'],599,601):raise RuntimeError('完整成片拼接失败：'+str(video)+' '+result.stderr[-1500:])
                    temp.replace(video)
                segments=[{'clip_id':Path(p).stem,'time_token':Path(p).stem.rsplit('_',1)[-1],'video_url':url,'local_video':local[p],'segment_index':i,'window_start_seconds':30.*i,'window_end_seconds':30.*(i+1)} for i,(p,url) in enumerate(zip(clip['paths'],clip['urls']))]
                group['clips'].append({'agent_dir':clip['agent_dir'],'agent_name':clip['agent_name'],'day':w['day'],'time_token':token,'clip_clock':clock,'duration_seconds':600.,'local_video':str(video),'full_local_video':str(video),'video_url':clip['urls'][0],'segments':segments,'source_video_urls':clip['urls']})
            group['selection']={'source_manifest':str(root/'source_manifest.json'),'stitched_paths':[x['full_local_video'] for x in group['clips']]}
            status.update(stage='frames_and_clip');save()
            item=prep.prepare_packet(group,dataset_root=dataset,cache_dir=Path(os.environ['TMPDIR'])/'frame_cache',config=config,config_fingerprint=fp,encoder=encoder,device='cpu',clip_batch_size=32,ffmpeg_binary=c['ffmpeg'],materialized_packet=group,video_sample_workers=4,media_prepare_workers=4)
            packets[w['source_packet_id']]={'root':str(dataset),'packet':item['packet_id']}
            durations.append(time.monotonic()-started);status['completed_windows'].append(w['generation_group_id'])
            status['mean_new_window_seconds']=sum(durations)/len(durations);save()
        prep.rebuild_dataset_index(dataset)
        rows={}
        for split in ['train','validation']:
            rows[split]=[]
            for d in definition[split]:
                p=packets[d['source_packet_id']]
                row=make_row(p['root'],p['packet'],int(d['speaker_index']),generation_profile=c['generation_profile'])
                if row['required_users'][0]!=d['speaker_user']:raise ValueError('实际speaker与条件清单不符')
                validate_row(row);rows[split].append(row)
        target=root/'data/trainval';target.mkdir(parents=True,exist_ok=True)
        for name,values in rows.items():
            with (target/(name+'.jsonl')).open('x',encoding='utf-8') as f:
                for row in values:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        largest=[];seen=set()
        for row in sorted(rows['train'],key=lambda r:-len(r['images'])):
            if row['source_packet_id'] in seen:continue
            seen.add(row['source_packet_id']);largest.append(row)
            if len(largest)==4:break
        runtime_val=[max(rows['validation'],key=lambda r:len(r['images']))]
        for name,values in [('runtime_train',largest),('runtime_validation',runtime_val)]:
            with (target/(name+'.jsonl')).open('x',encoding='utf-8') as f:
                for row in values:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        atomic_json(target/'packet_mapping.json',packets)
        from validate_full120 import validate_files
        report=validate_files(target/'train.jsonl',target/'validation.jsonl',definition,check_media=False)
        final={'status':'passed','cpu_job_id':job,'prepared_data_root':str(target),'split_counts':{k:len(v) for k,v in rows.items()},'windows':len(packets),'report':report,'finished_epoch':time.time()}
        atomic_json(root/'dataset_ready.json',final)
        status.update(status='passed',stage='complete',dataset=final,finished_epoch=time.time());save()
        print('FULL120_DATA_READY',json.dumps(final,ensure_ascii=False),flush=True)
    except BaseException as e:
        status.update(status='failed',error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc(),finished_epoch=time.time());save();raise


if __name__=='__main__':main()
