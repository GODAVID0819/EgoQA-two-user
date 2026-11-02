"""将旧保留测试窗口并入训练，原验证集合与历史输入保持可追溯。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .data import read_rows, validate_splits


def prepare(source, destination):
    source, destination = Path(source), Path(destination)
    old = {k: read_rows(source/(k+'.jsonl')) for k in ('train','validation','test')}
    validate_splits(old)
    if {k:len(v) for k,v in old.items()} != {'train':12,'validation':6,'test':6}:
        raise ValueError('迁移输入必须为已确认的12/6/6行')
    splits = {'train':old['train']+old['test'], 'validation':old['validation']}
    validate_splits(splits)
    destination.mkdir(parents=True,exist_ok=False)
    (destination/'train.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in splits['train']),encoding='utf-8')
    (destination/'validation.jsonl').write_bytes((source/'validation.jsonl').read_bytes())
    for name in ('train','validation'):
        (destination/('smoke_'+name+'.jsonl')).write_text(json.dumps(splits[name][0],ensure_ascii=False)+'\n',encoding='utf-8')
    report={'source':str(source),'rows':{k:len(v) for k,v in splits.items()},
            'source_windows':{k:sorted({r['source_packet_id'] for r in v}) for k,v in splits.items()},
            'validation_unchanged':True,'test_split_in_active_experiment':False}
    (destination/'split_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    parser.add_argument('--destination',required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.source,args.destination),ensure_ascii=False))
