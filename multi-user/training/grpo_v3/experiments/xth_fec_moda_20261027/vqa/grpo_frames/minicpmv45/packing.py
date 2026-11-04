"""原始帧全部保留；按用户独立使用官方六帧压缩，不跨用户、不抹去时间空隙。"""
import re

def pack_users(users,frames):
    groups=[]
    for user in users:
        paths=frames[user]
        for offset in range(0,len(paths),6):
            chunk=paths[offset:offset+6]
            times=[]
            for path in chunk:
                match=re.search(r'_([0-9]+\.[0-9]+)s\.jpg$',path)
                if not match:raise ValueError(f'不能解析原帧时间：{path}')
                times.append(round(float(match.group(1))*10))
            if times!=sorted(times) or len(set(times))!=len(times):
                raise ValueError('原帧时间必须严格递增')
            groups.append({'user':user,'paths':chunk,'temporal_ids':times})
    return groups

def generation_kwargs():
    return {'do_sample':False,'num_beams':1,'repetition_penalty':1.0,
            'max_new_tokens':512,'use_cache':True}
