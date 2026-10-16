"""核查同窗口固定留出speaker；不放宽旧独立窗口模式。"""
import collections
import json
from pathlib import Path
import re

from pair_data import validate_conditions


def window_key(packet):
    match=re.match(r'^RLHF6U_(DAY\d+)_(\d{8})_',packet)
    if not match:raise ValueError('无法定位真实窗口：'+packet)
    return match.group(1)+'_'+match.group(2)


def validate_files(train_path,val_path,definition,check_media=True):
    validate_conditions(definition['train'],definition['validation'])
    keys=set();counts=collections.defaultdict(collections.Counter)
    actual={}
    for split,path in [('train',train_path),('validation',val_path)]:
        rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        if len(rows)!=len(definition[split]):raise ValueError('数据数量与条件清单不符')
        for row,expected in zip(rows,definition[split]):
            window=window_key(row['source_packet_id']);key=(window,row['asker_index'])
            if key in keys:raise ValueError('训练/验证存在重复窗口speaker条件')
            keys.add(key);counts[window][split]+=1
            if (window,row['asker_index'],row['required_users'][0])!=(expected['window_id'],int(expected['speaker_index']),expected['speaker_user']):
                raise ValueError('实际条件或采样顺序与清单不符')
            if len(row['required_users'])!=6 or len(set(row['required_users']))!=6:
                raise ValueError('六视角用户集合不完整')
            if row['generation_profile']!='compact_full_qa_v1':raise ValueError('原prompt配置发生变化')
            if row['generator_frame_counts'][0]!=300:raise ValueError('speaker须保留300帧')
            if check_media:
                from training.grpo_v3.six_user_binary.data import validate_row
                validate_row(row)
        actual[split]=rows
    if len(counts)!=120 or any(count!={'train':5,'validation':1} for count in counts.values()):
        raise ValueError('每窗口须5训练/1验证')
    for start in range(0,600,4):
        if len({row['source_packet_id'] for row in actual['train'][start:start+4]})!=4:
            raise ValueError('更新批次包含重复窗口')
    return {'train':600,'validation':120,'windows':120,'distinct_windows_per_update':4,'six_views':True,'media_checked':check_media}


def validate_minimum_files(train_path,val_path,definition):
    from training.grpo_v3.six_user_binary.data import validate_row
    rows={split:[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()] for split,path in [('train',train_path),('validation',val_path)]}
    if (len(rows['train']),len(rows['validation']))!=(4,1):raise ValueError('唯一最小检查须4训练条件/1验证条件')
    if len({r['source_packet_id'] for r in rows['train']})!=4:raise ValueError('最小检查必须包含四不同窗口')
    for split in rows:
        allowed={(r['window_id'],int(r['speaker_index']),r['speaker_user']) for r in definition[split]}
        for row in rows[split]:
            if (window_key(row['source_packet_id']),row['asker_index'],row['required_users'][0]) not in allowed:
                raise ValueError('最小检查条件与正式划分不符')
            validate_row(row)
    return {'runtime_train':4,'runtime_validation':1,'distinct_train_windows':4,'media_checked':True}


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--train',type=Path,required=True);p.add_argument('--val',type=Path,required=True);p.add_argument('--definition',type=Path,required=True);p.add_argument('--minimum-runtime-check',action='store_true')
    args=p.parse_args()
    validate=validate_minimum_files if args.minimum_runtime_check else validate_files
    print(json.dumps(validate(args.train,args.val,json.loads(args.definition.read_text())),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
