"""从原始输出独立重算全部131题配对结果和已知复审子集，验证输入未被替换。"""
import json,sys,math
from pathlib import Path
import numpy as np
from grpo_inputs import load_inputs,build_input,parse_letter,parse_reason,write_json,SETTINGS
from packing import pack_users

def audit(out):
    code=Path(__file__).resolve().parent
    items,frames,header=load_inputs(code);byid={i['id']:i for i in items}
    config=json.loads((out/'run_config.json').read_text())
    cpu=json.loads((code/'cpu_ready.json').read_text())
    lengths={(r['id'],r['setting']):r for r in cpu['inputs']}
    status=json.loads((out/'run_status.json').read_text())
    assert status['status']=='COMPLETE_393' and status['n']==393
    loading=json.loads((out/'model_loading.json').read_text())
    assert not any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    maps={};unparsed={};missing_reason={};max_outputs={}
    for setting in SETTINGS:
        rows=[json.loads(line) for line in (out/f'{setting}_predictions.jsonl').read_text().splitlines()]
        assert len(rows)==131 and len({r['id'] for r in rows})==131
        assert {r['id'] for r in rows}==set(byid)
        maps[setting]={r['id']:r for r in rows}
        for row in rows:
            item=byid[row['id']];fr=frames[row['id']]
            users,paths,_=build_input(item,fr,setting,header)
            assert row['setting']==setting and row['users']==users and row['frame_paths']==paths
            assert row['correct']==item['correct'] and row['pred']==parse_letter(row['raw'])
            assert row['is_correct']==(row['pred']==item['correct'])
            assert row['visual_groups']==len(pack_users(users,fr['frames']))
            assert row['input_tokens']==lengths[(row['id'],setting)]['input_tokens']
            assert row['actual_processor_size']==[448,448] and row['preprocessed_size']==[156,156]
            assert row['model_revision']==config['model_revision'] and row['job_id']==config['job_id']
            row['audit_correct']=parse_letter(row['raw'])==item['correct']
        unparsed[setting]=sum(r['pred'] is None for r in rows)
        missing_reason[setting]=sum(parse_reason(r['raw']) is None for r in rows)
        max_outputs[setting]=sum(r['generated_tokens']>=512 for r in rows)
    def results(ids):
        result={'n':len(ids),'conditions':{},'comparisons':{}}
        for s in SETTINGS:
            count=sum(maps[s][i]['audit_correct'] for i in ids)
            result['conditions'][s]={'correct':count,'n':len(ids),'accuracy_percent':100*count/len(ids)}
        for a,b in [('evidence2','all6'),('evidence2','speaker1'),('all6','speaker1')]:
            first=sum(maps[a][i]['audit_correct'] and not maps[b][i]['audit_correct'] for i in ids)
            second=sum(not maps[a][i]['audit_correct'] and maps[b][i]['audit_correct'] for i in ids)
            both=sum(maps[a][i]['audit_correct'] and maps[b][i]['audit_correct'] for i in ids)
            result['comparisons'][a+'_minus_'+b]={'first_only_correct':first,'second_only_correct':second,
                'both_correct':both,'both_wrong':len(ids)-first-second-both,
                'gap_pp':100*(first-second)/len(ids)}
        return result
    ids=list(byid)
    strict=[i for i in ids if len(byid[i]['evidence_users'])==2]
    clean=[i for i in ids if byid[i].get('selection',{}).get('human_review_speaker_only')!='fail']
    windows={}
    for i in ids:windows.setdefault(byid[i]['packet_id'],[]).append(i)
    numer=np.array([sum(int(maps['evidence2'][i]['audit_correct'])-int(maps['all6'][i]['audit_correct'])
                            for i in cluster) for cluster in windows.values()])
    denom=np.array([len(cluster) for cluster in windows.values()])
    indexes=np.random.default_rng(0).integers(0,len(windows),(10000,len(windows)))
    samples=100*numer[indexes].sum(1)/denom[indexes].sum(1)
    report={'status':'VERIFIED_393','job_id':config['job_id'],'all131':results(ids),
        'strict_two_evidence':results(strict),'exclude_review_fail':results(clean),
        'unparsed':unparsed,'missing_reason':missing_reason,'hit_max_output_tokens':max_outputs,
        'cluster_bootstrap':{'clusters':len(windows),'replicates':10000,'seed':0,
            'evidence2_minus_all6_ci95_pp':np.quantile(samples,[.025,.975]).tolist()},
        'model':config['model'],'model_revision':config['model_revision'],
        'native_context':40960,'actual_processor_size':[448,448]}
    write_json(out/'final_audit.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report

if __name__=='__main__':audit(Path(sys.argv[1]))
