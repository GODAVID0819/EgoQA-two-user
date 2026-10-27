"""仅汇总实际已完成更新；历史池、待完成评分与旧实验不进入训练分母。"""
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import math
import statistics
import subprocess
import time

ROOT=Path(__file__).resolve().parent


def json_lines(path):
    if not path.exists():return []
    result=[]
    for line in path.read_text(errors='replace').splitlines():
        try:result.append(json.loads(line))
        except json.JSONDecodeError:continue
    return result


def main():
    record=json.loads((ROOT/'submissions/current_training_submission.json').read_text())
    job=record['job_id'];output=Path(record['output_dir'])
    accounting=subprocess.run(['sacct','-j',job,'--format=JobIDRaw,State,ExitCode,ElapsedRaw,NodeList','-nP'],capture_output=True,text=True)
    if accounting.returncode:raise RuntimeError(accounting.stderr)
    slurm=[line.split('|') for line in accounting.stdout.splitlines() if line.strip()]
    stage=json.loads((output/'progress_stage.json').read_text()) if (output/'progress_stage.json').exists() else {'phase':'queued_or_starting'}
    logs=[row for path in (output/'swift').glob('*/logging.jsonl') for row in json_lines(path)]
    completed=0
    for row in logs:
        if 'loss' in row and 'global_step/max_steps' in row:
            completed=max(completed,int(row['global_step/max_steps'].split('/')[0]))
    by_step=defaultdict(list)
    for rank in range(2):
        for row in json_lines(output/('rank'+str(rank))/'reward_trace.jsonl'):
            step=row.get('global_step')
            if isinstance(step,int) and 1<=step<=completed:by_step[step].append(row)
    metrics=[]
    for step,rows in sorted(by_step.items()):
        if len(rows)!=8:continue
        values=[row['reward'] for row in rows]
        if not all(math.isfinite(v) for v in values):raise ValueError('训练总奖励含非有限值')
        passed=sum(bool(row['bonus_gate']) for row in rows)
        metrics.append({'step':step,'training_mean_reward':statistics.mean(values),
            'quality_reward_mean':statistics.mean(row['quality_reward'] for row in rows),
            'diversity_distance_mean':statistics.mean(row['diversity_distance'] for row in rows),
            'diversity_bonus_mean':statistics.mean(row['diversity_bonus'] for row in rows),
            'pass_count':passed,'candidate_count':8,'auto_pass_rate':passed/8,
            'invalid_count':sum(row['judge_result']['status']=='invalid_completion' for row in rows)})
    summary={'captured_at_utc':datetime.now(timezone.utc).isoformat(),'job_id':job,'slurm_rows':slurm,
             'stage':stage,'completed_optimizer_updates':completed,'planned_updates':360,
             'updates_with_complete_reward_records':len(metrics),'historical_rescoring_performed':False,
             'auto_pass_count':sum(r['pass_count'] for r in metrics),'generated_slots':sum(r['candidate_count'] for r in metrics),
             'boundary':'训练分布固定Judge代理指标；无独立验证，不代表人工正确率或泛化'}
    if metrics:
        summary.update(training_mean_reward=statistics.mean(r['training_mean_reward'] for r in metrics),
            quality_reward_mean=statistics.mean(r['quality_reward_mean'] for r in metrics),
            diversity_bonus_mean=statistics.mean(r['diversity_bonus_mean'] for r in metrics),
            auto_pass_rate=summary['auto_pass_count']/summary['generated_slots'])
        for key,items in [('steps_1_10',[r for r in metrics if 1<=r['step']<=10]),('last_10',metrics[-10:])]:
            summary[key]=statistics.mean(r['training_mean_reward'] for r in items) if len(items)==10 else None
        with (output/'training_metrics_by_update.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    for name in ['run_manifest.json','training_result.json']:
        if (output/name).exists():summary[name[:-5]]=json.loads((output/name).read_text())
    summary['GPU_samples']={}
    for rank in range(2):
        path=output/('rank'+str(rank))/'gpu_metrics.csv'
        if not path.exists():continue
        samples=[];errors=0
        with path.open() as stream:
            for raw in csv.DictReader(stream):
                row={key.strip():value.strip() for key,value in raw.items() if key and value is not None}
                try:
                    stamp=datetime.strptime(row['timestamp'],'%Y/%m/%d %H:%M:%S.%f')
                    util=float(next(value for key,value in row.items() if key.startswith('utilization.gpu')).split()[0])
                    memory=float(next(value for key,value in row.items() if key.startswith('memory.used')).split()[0])
                    samples.append((stamp,util,memory))
                except (KeyError,ValueError,StopIteration):errors+=1
        if samples:
            end=samples[-1][0]
            stats={'samples':len(samples),'parse_error_rows':errors,'file_age_seconds':max(0,time.time()-path.stat().st_mtime),
                   'observed_seconds':(end-samples[0][0]).total_seconds(),'mean_utilization':statistics.mean(r[1] for r in samples),
                   'peak_memory_MiB':max(r[2] for r in samples),'windows':{},
                   'aggregation':'已记录样本算术平均，样本数不当作精确秒数'}
            for minutes in [10,30,60,120]:
                subset=[r for r in samples if (end-r[0]).total_seconds()<=minutes*60]
                stats['windows'][str(minutes)+'min']={'mean_utilization':statistics.mean(r[1] for r in subset),
                    'samples':len(subset),'observed_span_seconds':(end-subset[0][0]).total_seconds(),
                    'full_window_available':(end-samples[0][0]).total_seconds()>=minutes*60}
            summary['GPU_samples']['rank'+str(rank)]=stats
    timings=[row for rank in range(2) for row in json_lines(output/('rank'+str(rank))/'phase_timing.jsonl')
             if row.get('phase')=='optimizer_step_wall' and row.get('global_step',0)<=completed]
    recent=[r['seconds'] for r in timings if r['global_step']>completed-10]
    if recent:
        summary['recent_step_wall_seconds_median']=statistics.median(recent)
        summary['remaining_training_seconds_estimate']=(360-completed)*statistics.median(recent)
    if output.exists():(output/'live_training_status.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
