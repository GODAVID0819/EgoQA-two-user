"""独立重算三组完整预测、原帧映射、解析及配对准确率，不加载模型。"""
import argparse
import json
from collections import Counter
from pathlib import Path
import grpo_inputs as core


def audit(output):
    root=Path(__file__).resolve().parent
    config=json.loads((output/'run_config.json').read_text(encoding='utf-8'))
    state=json.loads((output/'run_status.json').read_text(encoding='utf-8'))
    if state['status']!='COMPLETED' or state['completed_predictions']!=393:
        raise ValueError('程序尚未完成全部393条预测')
    items,frames,header=core.load_inputs(root);by_id={x['id']:x for x in items}
    cpu=json.loads(Path(config['cpu_report']).read_text(encoding='utf-8'))
    cpu_rows={(r['id'],r['setting']):r for r in cpu['records']}
    groups={};stats={};sizes={}
    for setting in core.SETTINGS:
        rows=[json.loads(line) for line in (output/f'{setting}_predictions.jsonl').read_text(encoding='utf-8').splitlines()]
        if len(rows)!=131 or {r['id'] for r in rows}!=set(by_id) or len({r['id'] for r in rows})!=131:
            raise ValueError(f'{setting}题号缺失或重复')
        for row in rows:
            item=by_id[row['id']]
            users,paths,_=core.build_input(item,frames[item['id']],setting,header)
            predicted=core.parse_letter(row['raw']);expected=cpu_rows[(item['id'],setting)]
            if (row['users']!=users or row['frame_paths']!=paths or row['n_images']!=len(paths)
                or row['correct']!=item['correct'] or row['pred']!=predicted
                or row['reason']!=core.parse_reason(row['raw']) or row['is_correct']!=(predicted==item['correct'])
                or row['model_revision']!=config['model_revision']
                or row['prompt_tokens']!=expected['prompt_tokens']
                or row['actual_size_histogram']!=expected['actual_size_histogram']
                or row['actual_padded_context']+512>32768):
                raise ValueError(f'预测/原始帧/处理器记录不一致：{setting} {item["id"]}')
        correct=sum(r['pred']==by_id[r['id']]['correct'] for r in rows)
        stats[setting]={'n':131,'correct':correct,'accuracy':correct/131,
                        'unparsed':sum(r['pred'] is None for r in rows)}
        sizes[setting]=dict(sum((Counter(r['actual_size_histogram']) for r in rows),Counter()))
        groups[setting]={r['id']:r for r in rows}
    comparisons={}
    producer=json.loads((output/'paired_summary.json').read_text(encoding='utf-8'))
    for setting,stat in stats.items():
        if producer['conditions'][setting]['overall']['correct']!=stat['correct']:
            raise ValueError('生产统计与独立重算不一致')
    for first,second in [('evidence2','all6'),('evidence2','speaker1'),('all6','speaker1')]:
        pairs=Counter((groups[first][i]['pred']==by_id[i]['correct'],groups[second][i]['pred']==by_id[i]['correct']) for i in by_id)
        gap=100*(stats[first]['correct']-stats[second]['correct'])/131
        name=f'{first}_minus_{second}'
        if abs(producer['comparisons'][name]['gap_pp']-gap)>1e-9:raise ValueError('配对差值不一致')
        comparisons[name]={'n':131,'gap_pp':gap,'both_correct':pairs[(True,True)],
                          'first_only_correct':pairs[(True,False)],'second_only_correct':pairs[(False,True)],
                          'both_wrong':pairs[(False,False)]}
    strict=[i['id'] for i in items if len(i['evidence_videos'])==2]
    review=[i['id'] for i in items if i['selection'].get('human_review_speaker_only')!='fail']
    subsets={label:{s:{'n':len(ids),'correct':sum(groups[s][i]['is_correct'] for i in ids)} for s in core.SETTINGS}
             for label,ids in [('strict_two_evidence_130',strict),('exclude_review_fail_114',review)]}
    result={'status':'VERIFIED_393','slurm_job_id':config['slurm_job_id'],'conditions':stats,
            'comparisons':comparisons,'actual_image_sizes':sizes,'subsets':subsets,
            'retained_review_failures':17,'evidence_single_view_exceptions':1}
    core.write_json(output/'final_audit.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(audit(args.out),ensure_ascii=False,indent=2))
