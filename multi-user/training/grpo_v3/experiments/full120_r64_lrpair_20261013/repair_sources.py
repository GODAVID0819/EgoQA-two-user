"""完整源替换允许最小新增重叠，连通组共用留出speaker，保留120/600/120。"""
import collections
import concurrent.futures
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
import traceback
from urllib.parse import quote

from media_quality import assess_probe,window_problem,overlaps
from pair_data import USERS,make_order,validate_conditions


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)


def seconds(token):return int(token[:2])*3600+int(token[2:4])*60+int(token[4:6])+int(token[6:])/100


def token(sec):return f'{int(sec)//3600:02d}{int(sec)%3600//60:02d}{int(sec)%60:02d}00'


def overlap_seconds(start_a,start_b):
    return max(0,min(start_a+600,start_b+600)-max(start_a,start_b))


def replacement_score(old,start,windows):
    others=[w for w in windows if w['source_packet_id']!=old['source_packet_id'] and w['day']==old['day']]
    if any(seconds(w['time_token'])==start for w in others):return None
    old_start=seconds(old['time_token'])
    added=sum(max(0,overlap_seconds(start,seconds(w['time_token']))-overlap_seconds(old_start,seconds(w['time_token']))) for w in others)
    return added,abs(start-old_start),start


def overlap_components(windows):
    groups=[];previous_day=None;end=-1
    for w in sorted(windows,key=lambda w:(w['day'],seconds(w['time_token']),w['source_packet_id'])):
        start=seconds(w['time_token'])
        if w['day']!=previous_day or start>=end:groups.append([])
        groups[-1].append(w['source_packet_id']);previous_day=w['day'];end=start+600
    return sorted(groups,key=lambda g:(-len(g),tuple(g)))


def assign_component_holdouts(windows,preferred,seed=42):
    groups=overlap_components(windows)
    if len(windows)!=120 or max(map(len,groups))>20:
        raise ValueError('须有120窗，且每个重叠连通组至多20窗以满足每名speaker留出20条件')
    rng=random.Random(seed);tie_order=list(range(6));rng.shuffle(tie_order)
    capacity=[20]*6;assigned={};failed=set()
    def place(index):
        if index==len(groups):return not any(capacity)
        key=(index,tuple(sorted(capacity)))
        if key in failed:return False
        group=groups[index];n=len(group);preferences=collections.Counter(preferred[w] for w in group)
        options=sorted(tie_order,key=lambda u:(-preferences[u],-capacity[u]))
        tried=set()
        for user in options:
            if capacity[user]<n or capacity[user] in tried:continue
            tried.add(capacity[user]);capacity[user]-=n
            if place(index+1):
                assigned.update({w:user for w in group});return True
            capacity[user]+=n
        failed.add(key);return False
    if not place(0):raise ValueError('连通组不能分成六组各20窗；不自动放宽均衡要求')
    return assigned,groups


def repartition_conditions(windows,definition):
    previous={r['source_packet_id']:int(r['speaker_index']) for r in definition['validation']}
    held,groups=assign_component_holdouts(windows,previous,seed=42)
    result=deepcopy(definition);result['train']=[];result['validation']=[]
    for row in definition['train']+definition['validation']:
        row=deepcopy(row);split='validation' if int(row['speaker_index'])==held[row['source_packet_id']] else 'train'
        row['split']=split;result[split].append(row)
    result['train']=make_order(result['train'],seed=42)
    result['validation'].sort(key=lambda r:(r['day'],r['time_token']))
    validate_conditions(result['train'],result['validation'])
    return result,{'components':groups,'component_sizes':dict(collections.Counter(map(len,groups))),
        'changed_holdout_windows':sum(held[w]!=previous[w] for w in held),
        'held_out_by_source_packet':{w:USERS[u] for w,u in held.items()},'seed':42}


def reuse_source_audit(path):
    records={}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        row=json.loads(line);media=Path(row['local_path'])
        if media.is_file() and media.stat().st_size==row['bytes']:records[row['path']]=row
    return records


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--reuse-audit-output',type=Path)
    args=parser.parse_args();root=args.root.resolve();job=os.environ['SLURM_JOB_ID'];output=root/'outputs'/('source_repair_'+job)
    output.mkdir(parents=True,exist_ok=True)
    state={'status':'running','job_id':job,'stage':'auditing_source_media','started_epoch':time.time(),
           'failed_preparation_job_ids':['19717659','19722680'],
           'user_approved_policy':'2026-10-14允许最小额外重叠，保持120；重叠连通组共用留出speaker，600/120'}
    def save():atomic_json(output/'status.json',state)
    save()
    try:
        source_path=root/'source_manifest.json';definition_path=root/'condition_definition.json'
        source=json.loads(source_path.read_text());definition=json.loads(definition_path.read_text())
        validate_conditions(definition['train'],definition['validation'])
        official=json.loads((root/'official_files_sizes.json').read_text())
        existing=json.loads((root/'existing_segments.json').read_text())
        base=Path('/scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929')
        previous=[base/'data/frames_18719659',base/'variants/train8-data-20261007T162701Z/data/frames',base/'variants/speaker-holdout-design-20261009/data/frames']
        def already_prepared(window):
            packet='RLHF6U_'+window['day']+'_'+window['time_token']+'_'+'_'.join(c['agent_dir'] for c in sorted(window['clips'],key=lambda c:c['agent_dir']))
            return any((r/'packets'/packet/'COMPLETE').is_file() for r in previous)
        audited=[w for w in source['windows'] if not already_prepared(w)]
        media={};all_tasks={}
        if args.reuse_audit_output:
            journal=args.reuse_audit_output.resolve()/'source_media_audit.jsonl'
            media=reuse_source_audit(journal)
            state.update(reused_audit_source=str(journal),reused_audit_records=len(media))
            with (output/'source_media_audit.jsonl').open('w',encoding='utf-8') as f:
                for row in media.values():f.write(json.dumps(row)+'\n')
            print('SOURCE_AUDIT_REUSED',len(media),flush=True)
        for window in audited:
            for clip in window['clips']:
                for rel,url,size in zip(clip['paths'],clip['urls'],clip['sizes']):all_tasks[rel]=(rel,url,size)
        ffprobe='/scratch/xl6775/envs/egoqa-ffmpeg-runtime/bin/ffprobe'
        def inspect(path):
            result=subprocess.run([ffprobe,'-v','error','-show_entries','format=duration:stream=codec_type,width,height,start_time,duration','-of','json',str(path)],capture_output=True,text=True,timeout=30)
            if result.returncode:raise ValueError('源文件不可读：'+str(path)+' '+result.stderr[-500:])
            info=assess_probe(json.loads(result.stdout))
            if info['problem'] in ('no_video','unknown_video_duration'):raise ValueError('源文件缺少视频信息：'+str(path))
            return info
        def obtain(item):
            import requests
            rel,url,size=item;target=root/'segments'/rel
            choices=[target]
            if rel in existing:choices.append(Path(existing[rel]))
            choices.extend([Path('/scratch/xl6775/datasets/EgoLife')/rel,target.with_suffix('.download-part')])
            for p in choices:
                if p.is_file() and p.stat().st_size==size:
                    try:info=inspect(p)
                    except (ValueError,OSError,subprocess.TimeoutExpired):continue
                    if p==target.with_suffix('.download-part'):
                        target.parent.mkdir(parents=True,exist_ok=True);p.replace(target);p=target
                    return {'path':rel,'local_path':str(p),'bytes':size,'download_status':'reused',**info}
            target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_suffix('.download-part')
            for attempt in range(4):
                try:
                    with requests.get(url,stream=True,timeout=(20,120)) as response:
                        response.raise_for_status()
                        with temp.open('wb') as f:
                            for block in response.iter_content(1024*1024):
                                if block:f.write(block)
                    if temp.stat().st_size!=size:raise ValueError(f'大小不符：{rel} {temp.stat().st_size}/{size}')
                    info=inspect(temp);temp.replace(target)
                    return {'path':rel,'local_path':str(target),'bytes':size,'download_status':'downloaded',**info}
                except Exception:
                    if attempt==3:raise
                    time.sleep(2*(attempt+1))
        def get_many(tasks):
            pending=[item for item in tasks if item[0] not in media]
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                for item in pool.map(obtain,pending):
                    media[item['path']]=item
                    with (output/'source_media_audit.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(item)+'\n')
                    if len(media)%60==0:
                        state.update(checked_segments=len(media),source_anomaly_segments=sum(bool(v['problem']) for v in media.values()));save()
                        print('SOURCE_AUDIT',len(media),'/',len(all_tasks),'anomalies',state['source_anomaly_segments'],flush=True)
        get_many(all_tasks.values())
        bad=[]
        for window in audited:
            problem=window_problem(window,media)
            if problem:bad.append((window,problem))
        atomic_json(output/'original_window_problems.json',[{'window_id':w['day']+'_'+w['time_token'],'problem':p} for w,p in bad])
        state.update(stage='selecting_complete_replacements',bad_windows=len(bad));save()
        windows=deepcopy(source['windows']);replacements=[]
        for old,problem in bad:
            day=old['day'];users=sorted(old['clips'],key=lambda c:c['agent_dir']);available=[]
            for user in users:
                prefix=user['agent_dir']+'/'+day+'/'+day+'_'+user['agent_dir']+'_'
                available.append({seconds(p[len(prefix):-4]) for p,size in official.items()
                    if p.startswith(prefix) and p.endswith('.mp4') and size>0 and len(p[len(prefix):-4])==8 and p[len(prefix):-4].isdigit() and p[-6:-4]=='00'})
            common=set.intersection(*available)
            candidates=[s for s in common if s%30==0 and all(s+30*i in common for i in range(20))
                and replacement_score(old,s,windows) is not None]
            candidates.sort(key=lambda s:replacement_score(old,s,windows))
            chosen=None;rejected=[]
            for start in candidates:
                new=deepcopy(old);new['time_token']=token(start);new['generation_group_id']=day+'::'+new['time_token']
                new['source_packet_id']='RLHF6U_'+day+'_'+new['time_token']+'_'+'_'.join(c['agent_dir'].split('_',1)[0] for c in users)
                new['clips']=[]
                for user in users:
                    paths=[user['agent_dir']+'/'+day+'/'+day+'_'+user['agent_dir']+'_'+token(start+30*i)+'.mp4' for i in range(20)]
                    new['clips'].append({'agent_dir':user['agent_dir'],'agent_name':user['agent_name'],'paths':paths,
                        'urls':['https://huggingface.co/datasets/lmms-lab/EgoLife/resolve/main/'+quote(p,safe='/') for p in paths],
                        'sizes':[official[p] for p in paths]})
                if any(p in media and media[p]['problem'] for c in new['clips'] for p in c['paths']):continue
                tasks=[(p,u,n) for c in new['clips'] for p,u,n in zip(c['paths'],c['urls'],c['sizes'])]
                get_many(tasks)
                bad_candidate=window_problem(new,media)
                if bad_candidate:rejected.append({'time_token':new['time_token'],'problem':bad_candidate});continue
                chosen=new;break
            if chosen is None:raise RuntimeError('同日没有六视角完整且不重复的替代窗口：'+old['generation_group_id'])
            old_id=old['source_packet_id'];new_id=chosen['source_packet_id']
            score=replacement_score(old,seconds(chosen['time_token']),windows)
            windows=[chosen if w['source_packet_id']==old_id else w for w in windows]
            for split in ['train','validation']:
                for row in definition[split]:
                    if row['source_packet_id']!=old_id:continue
                    row.update(source_packet_id=new_id,window_id=day+'_'+chosen['time_token'],time_token=chosen['time_token'],
                        replacement_for_window_id=day+'_'+old['time_token'],original_source_packet_id=old_id,
                        source_metadata_file=str(root/'source_manifest.json'))
            replacements.append({'old_window':day+'_'+old['time_token'],'new_window':day+'_'+chosen['time_token'],
                'old_source_packet_id':old_id,'new_source_packet_id':new_id,'reason':problem,
                'added_pairwise_overlap_seconds':score[0],'shift_seconds':score[1],
                'selection_order':'新增配对重叠秒数、移动距离、起点；按顺序选择实际完整候选',
                'rejected_candidates':rejected})
            print('COMPLETE_WINDOW_REPLACEMENT',json.dumps(replacements[-1],ensure_ascii=False),flush=True)
        old_held={r['source_packet_id']:r['speaker_user'] for r in definition['validation']}
        definition,split_summary=repartition_conditions(windows,definition)
        for item in replacements:
            new_id=item['new_source_packet_id'];speaker=split_summary['held_out_by_source_packet'][new_id]
            item.update(held_out_speaker=speaker,held_out_speaker_unchanged=speaker==old_held[new_id])
        assert len(windows)==120 and len({w['source_packet_id'] for w in windows})==120
        for name in ['source_manifest','condition_definition']:
            p=root/(name+'.json');backup=root/(name+'_before_replacements.json')
            if not backup.exists():shutil.copy2(p,backup)
        source['windows']=windows;source['replacements']=replacements
        source['original_overlaps']=source.pop('overlaps',[])
        source['overlaps']=[{'day':a['day'],'window_a':a['source_packet_id'],'window_b':b['source_packet_id'],
            'overlap_seconds':overlap_seconds(seconds(a['time_token']),seconds(b['time_token']))}
            for i,a in enumerate(windows) for b in windows[i+1:]
            if a['day']==b['day'] and overlaps(seconds(a['time_token']),seconds(b['time_token']))]
        unique={p:n for w in windows for c in w['clips'] for p,n in zip(c['paths'],c['sizes'])}
        source.update(expected_unique_segments=len(unique),expected_bytes=sum(unique.values()))
        atomic_json(source_path,source);atomic_json(definition_path,definition)
        atomic_json(root/'window_replacements.json',{'source_repair_job_id':job,'failed_preparation_job_id':'19717659',
            'policy':state['user_approved_policy']+'；不补帧，不继承旧QA或证据',
            'replacements':replacements,'counts':{'windows':120,'train':600,'validation':120},
            'split':split_summary,'overlaps':source['overlaps'],
            'source_media_audit':str(output/'source_media_audit.jsonl')})
        state.update(status='passed',stage='complete',replacement_count=len(replacements),checked_segments=len(media),finished_epoch=time.time());save()
        print('SOURCE_REPAIR_READY',json.dumps(state,ensure_ascii=False),flush=True)
    except BaseException as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),finished_epoch=time.time());save();raise


if __name__=='__main__':main()
