"""按真实speaker轮换划分；只允许训练与同窗口验证共享媒体。"""
from collections import Counter,defaultdict


def partition_training(rows):
    windows=defaultdict(list)
    for row in rows:windows[row['source_packet_id']].append(row)
    if len(windows)!=8:raise ValueError('必须有8个训练窗口')
    first=min(windows)
    first_rows=sorted(windows[first],key=lambda row:row['asker_index'])
    users=[row['required_users'][0] for row in first_rows]
    if len(users)!=6 or len(set(users))!=6:raise ValueError('首窗口必须有六名不同真实speaker')
    train=[];validation=[];mapping=[]
    for index,packet in enumerate(sorted(windows)):
        values=windows[packet]
        by_user={row['required_users'][0]:row for row in values}
        if len(values)!=6 or set(by_user)!=set(users):raise ValueError('窗口缺失或重复真实speaker：'+packet)
        chosen=users[index%6]
        validation.append(by_user[chosen])
        train.extend(by_user[u] for u in users if u!=chosen)
        mapping.append({'source_packet_id':packet,'held_out_user':chosen,'held_out_asker_index':by_user[chosen]['asker_index']})
    return train,validation,{'canonical_users':users,'windows':mapping,
        'holdout_user_counts':dict(Counter(row['required_users'][0] for row in validation))}


def validate_holdout(splits):
    if set(splits)-{'train','validation','test'}:raise ValueError('未知集合')
    train=splits.get('train',[]);val=splits.get('validation',[])
    if (len(train),len(val))!=(40,8):raise ValueError('训练/验证必须为40/8')
    keys={}
    for split,rows in splits.items():
        for row in rows:
            key=(row['source_packet_id'],row['asker_index'])
            if key in keys:raise ValueError('重复或跨集合相同窗口/speaker：'+str(key))
            keys[key]=split
    train_windows={row['source_packet_id'] for row in train};val_windows={row['source_packet_id'] for row in val}
    if len(train_windows)!=8 or train_windows!=val_windows:raise ValueError('训练/验证须为相同8窗口')
    for packet in train_windows:
        t=[row for row in train if row['source_packet_id']==packet]
        v=[row for row in val if row['source_packet_id']==packet]
        if (len(t),len(v))!=(5,1):raise ValueError('每窗口须为5/1')
        users=[row['required_users'][0] for row in t+v]
        if len(set(users))!=6:raise ValueError('每窗口必须有六名不同真实speaker')
    if 'test' in splits:
        tests=splits['test'];test_windows={row['source_packet_id'] for row in tests}
        if test_windows & train_windows:raise ValueError('测试窗口不得与训练/验证重叠')
        if len(tests)!=6 or len(test_windows)!=3:raise ValueError('测试必须为3窗口/6输入')
        if any(sum(row['source_packet_id']==packet for row in tests)!=2 for packet in test_windows):raise ValueError('每测试窗口必须2输入')
