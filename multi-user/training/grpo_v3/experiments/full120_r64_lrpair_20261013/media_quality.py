"""检查真实视频覆盖，不用音轨时长或补帧冒充完整视觉输入。"""


def assess_probe(metadata):
    videos=[s for s in metadata.get('streams',[]) if s.get('codec_type')=='video' and s.get('width',0)>0 and s.get('height',0)>0]
    if not videos:return {'problem':'no_video'}
    try:
        video=videos[0];duration=float(video['duration']);start=float(video.get('start_time',0))
        total=float(metadata['format']['duration'])
    except (KeyError,ValueError,TypeError):return {'problem':'unknown_video_duration'}
    problem=None
    if duration<29.5:problem='short_video'
    elif duration>30.5:problem='long_video'
    elif abs(start)>.1:problem='video_start_offset'
    elif not 29.5<=total<=30.5:problem='container_duration_outside_slot'
    return {'problem':problem,'video_duration':duration,'video_start':start,'format_duration':total}


def window_problem(window,media):
    errors=[]
    for clip in window['clips']:
        values=[media[p] for p in clip['paths']]
        bad=[{'path':path,**info} for path,info in zip(clip['paths'],values) if info.get('problem')]
        if bad:errors.append({'agent_dir':clip.get('agent_dir'),'segments':bad});continue
        video=sum(v['video_duration'] for v in values);total=sum(v['format_duration'] for v in values)
        if not 599<=video<=601 or not 599<=total<=601:
            errors.append({'agent_dir':clip.get('agent_dir'),'problem':'accumulated_window_duration','video_duration':video,'format_duration':total})
    return errors or None


def overlaps(start_a,start_b):
    return max(start_a,start_b)<min(start_a+600,start_b+600)
