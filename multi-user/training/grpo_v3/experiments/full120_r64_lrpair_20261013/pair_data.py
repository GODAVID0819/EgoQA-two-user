"""600/120条件核查以及每四条来自不同窗口的数据顺序。"""
import collections
import random

USERS = ['Jake','Alice','Tasha','Lucia','Katrina','Shure']


def condition_key(row):
    return (row['source_packet_id'], int(row['speaker_index']))


def validate_conditions(train, validation):
    if (len(train), len(validation)) != (600,120):
        raise ValueError('必须为600训练条件与120验证条件')
    keys = set()
    windows = collections.defaultdict(lambda: collections.Counter())
    users = collections.defaultdict(collections.Counter)
    for split, rows in [('train',train),('validation',validation)]:
        for row in rows:
            key = condition_key(row)
            if key in keys:
                raise ValueError('重复或跨集合的窗口/speaker条件')
            keys.add(key)
            if not 0 <= key[1] < 6 or USERS[key[1]] != row['speaker_user']:
                raise ValueError('speaker索引与真实用户不符')
            if int(row['input_view_count']) != 6 or float(row['duration_seconds']) != 600:
                raise ValueError('须保留六视角十分钟窗口')
            windows[key[0]][split] += 1
            users[split][row['speaker_user']] += 1
    if len(windows)!=120 or any(c != {'train':5,'validation':1} for c in windows.values()):
        raise ValueError('每窗口必须恰好5训练和1验证条件')
    if set(users['train'].values()) != {100} or set(users['validation'].values()) != {20}:
        raise ValueError('六名speaker必须均衡为100训练与20验证')


def make_order(train, seed=42):
    windows=collections.defaultdict(list)
    for row in train:
        windows[row['source_packet_id']].append(row)
    if len(train)!=600 or len({condition_key(r) for r in train})!=600 or len(windows)!=120:
        raise ValueError('训练条件缺失或重复')
    if any(len(rows)!=5 for rows in windows.values()):
        raise ValueError('每窗口必须有5个训练条件')
    rng=random.Random(seed)
    for window in sorted(windows):
        windows[window].sort(key=lambda r:int(r['speaker_index']))
        rng.shuffle(windows[window])
    ordered=[]
    for round_index in range(5):
        names=sorted(windows)
        rng.shuffle(names)
        for start in range(0,120,4):
            ordered.extend(windows[name][round_index] for name in names[start:start+4])
    return ordered
