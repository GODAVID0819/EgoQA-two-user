"""按事实组去重并分层选题。来源、题文和标签均作为私有输入。"""
import argparse
import collections
import json
import random
from pathlib import Path


def select(rows, count, seed=42):
    """每条记录须有唯一 id、人工核过的 family 和原始 scores。"""
    rows = list(rows)
    if len({r['id'] for r in rows}) != len(rows) or any(not r.get('family') for r in rows):
        raise ValueError('需要唯一候选 ID 和非空事实组')
    if not 0 < count <= len({r['family'] for r in rows}):
        raise ValueError('目标数量超出可用事实组')
    random.Random(seed).shuffle(rows)
    groups = collections.defaultdict(list)
    for r in rows:
        s = r['scores']
        f,e,c = (s[k] for k in ('formality','groundedness','combined_answerability'))
        if (c is None) != (e <= .5) or not all(0 <= v <= 1 for v in (f,e,c) if v is not None):
            raise ValueError('评分缺失或门控状态不一致')
        groups[(f>=.5,e>=.5,-1 if c is None else int(c>=.5))].append(r)
    selected=[]; seen=set(); sizes=collections.Counter()
    def add(key,row):
        if row['family'] in seen:
            return False
        seen.add(row['family']); selected.append(row); sizes[key]+=1
        return True
    for key in [(True,True,1),(False,True,1)]:
        for r in groups[key]:
            if not add(key,r) and key==(True,True,1):
                raise ValueError('全过记录在同一事实组重复；请决定是否按证据条件分别标注')
    if len(selected)>count:raise ValueError('目标数量不足以覆盖保留的C正例')
    while len(selected)<count:
        avail={k:[r for r in rr if r['family'] not in seen] for k,rr in groups.items()}
        k=min((k for k,v in avail.items() if v),key=lambda k:(sizes[k],k))
        add(k,avail[k][0])
    return selected


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('out',type=Path)
    ap.add_argument('--count',type=int,default=200);ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    result=select(json.loads(args.source.read_text(encoding='utf8')),args.count,args.seed)
    with args.out.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False)
