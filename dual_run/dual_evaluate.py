"""每卡独立生成固定基座/adapter配对，按原全局验证序号合并。"""
import argparse
import json
import os
from pathlib import Path
import subprocess


def merge_validation(output):
    from training.grpo_v3.six_user_binary.evaluation import compare_rows, summarize, same_frozen_judge
    output=Path(output);c=json.loads((output/'run_config.json').read_text())
    bases=[];policies=[]
    for rank in range(c['world_size']):
        p=output/('rank'+str(rank));b=json.loads((p/'validation_baseline.json').read_text());a=json.loads((p/'validation_policy.json').read_text())
        if b['status']!='completed' or a['status']!='completed':raise ValueError('配对分片未完整完成')
        if b['settings']!=a['settings'] or b['input_bindings']!=a['input_bindings'] or not same_frozen_judge(b['judge'],a['judge']):
            raise ValueError('分片基座/adapter输入、设置或Judge实例不一致')
        bases.append(b);policies.append(a)
    combined=[]
    for label,parts in [('baseline',bases),('policy',policies)]:
        bindings=[r for p in parts for r in p['input_bindings']]
        rows=sorted([r for p in parts for r in p['rows']],key=lambda r:(r['seed'],r['slot']))
        expected=2 if c.get('dual_runtime_check') else 120
        if len(bindings)!=expected or len(rows)!=expected*4:raise ValueError('固定验证全局数量不完整')
        if len({(r['evidence_id'],r['slot']) for r in rows})!=len(rows):raise ValueError('配对候选重复')
        record={'status':'completed','settings':parts[0]['settings'],'input_bindings':bindings,'rows':rows,
                'adapter':parts[0]['adapter'],'judge_shards':[p['judge'] for p in parts],
                'summary':summarize(rows),'paired_judge_identity_per_shard_verified':True}
        (output/('validation_'+label+'.json')).write_text(json.dumps(record,indent=2))
        combined.append(record)
    comparison=compare_rows(combined[0]['rows'],combined[1]['rows'])
    (output/'validation_comparison.json').write_text(json.dumps(comparison,indent=2))


def main():
    p=argparse.ArgumentParser()
    for key in ['config','output','adapter','url','instance']:p.add_argument('--'+key,required=True)
    a=p.parse_args();output=Path(a.output)
    python=os.environ.get('EGOQA_EVAL_PYTHON',__import__('sys').executable)
    for label in ['baseline','policy']:
        cmd=[python,'-m','training.grpo_v3.six_user_binary.evaluation','--config',a.config,
             '--output',str(output/('validation_'+label+'.json')),'--judge-url',a.url,'--judge-instance',a.instance]
        if label=='policy':cmd+=['--adapter',a.adapter,'--baseline',str(output/'validation_baseline.json')]
        subprocess.run(cmd,check=True)

if __name__=='__main__':main()
