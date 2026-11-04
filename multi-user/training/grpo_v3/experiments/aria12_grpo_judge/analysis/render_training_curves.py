"""按双rank已确认更新的逐候选派生统计绘制两张四面板图。"""
import argparse,csv,json,math,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator,PercentFormatter

KEYS=('F','E','C','Total')
LABELS=('Formality (F)','Groundedness (E)','Gated answerability (C)','Total training reward')

def render(data,kind,destination):
 points=data['points'];last=data['completed_updates'];steps=[p['step'] for p in points]
 assert steps==list(range(1,last+1))
 start=20 if last>=100 else 1
 fig=plt.figure(figsize=(15.2,10.6),dpi=180)
 label='Rewards' if kind=='mean_reward' else 'Pass rates'
 fig.text(.5,.975,f"Job {data['job_id']} | {label} | {last}/24 updates",
          ha='center',fontsize=18,weight='bold',color='#101820')
 fig.text(.5,.948,f"Captured {data['captured_at_beijing']} Beijing | {data['candidate_slots']} original generated slots | init: checkpoint-348 | completed updates only",
          ha='center',fontsize=10.4,color='#718096')
 stats={}
 for i,(key,title) in enumerate(zip(KEYS,LABELS)):
  x0=.06 if i%2==0 else .565;y0=.565 if i<2 else .14
  ax=fig.add_axes([x0,y0,.415,.255])
  if kind=='pass_rate' and key=='Total':title='All three judges'
  values=[float(p[kind][key]) for p in points]
  assert all(math.isfinite(v) and 0<=v<=1 for v in values)
  average=statistics.fmean
  first=average(values[:10]) if last>=10 else None
  recent=average(values[-10:]) if last>=10 else None
  all_mean=average(values)
  fmt=(lambda v:f'{v:.1%}') if kind=='pass_rate' else (lambda v:f'{v:.4f}')
  stats[key]={'steps_1_10':first,'last_10_updates':recent,'all_logged_updates':all_mean}
  fig.text(x0,y0+.325,title,fontsize=13.4,weight='bold',color='#22354b')
  text='    '.join(f'{name}  {fmt(v) if v is not None else "Not available"}'
      for name,v in [('Steps 1–10',first),('Last 10 updates',recent),('All logged updates',all_mean)])
  fig.text(x0,y0+.271,text,fontsize=9.1,color='#46556b')
  ax.plot(steps,values,color='#87bade',linewidth=.85,marker='o',markersize=2.4,
          markeredgewidth=0,alpha=.95,label='Per update',zorder=2)
  if last>=10:
   rolling=[average(values[j-9:j+1]) for j in range(9,last)]
   ax.plot(steps[9:],rolling,color='#ee8b2b',linewidth=2,
           marker='o' if last==10 else None,markersize=3,label='10-update trailing mean',zorder=4)
   ax.axhline(first,color='#8c939d',linestyle='--',linewidth=1.05,
              label='Steps 1–10 training reference',zorder=1)
  shown=[v for s,v in zip(steps,values) if s>=start]
  if kind=='pass_rate':
   limits=(-.035,1.035)
   ax.yaxis.set_major_formatter(PercentFormatter(1.0,decimals=0))
   ax.set_yticks([0,.2,.4,.6,.8,1.]);ax.set_ylabel('Pass rate',fontsize=9.4)
  else:
   span=max(shown)-min(shown);pad=max(.02,.08*span)
   limits=(min(shown)-pad,max(shown)+pad)
   ax.yaxis.set_major_locator(MaxNLocator(6));ax.set_ylabel('Mean reward / score',fontsize=9.4)
  ax.set_xlim(start-.25,max(last+.25,start+1));ax.set_ylim(*limits)
  ax.xaxis.set_major_locator(MaxNLocator(nbins=7,integer=True))
  ax.set_xlabel('Completed optimizer update',fontsize=9.5)
  ax.tick_params(labelsize=8.6)
  ax.grid(axis='y',color='#e4e9ef',linewidth=.7);ax.set_axisbelow(True)
  ax.spines[['top','right']].set_visible(False)
  for name in ['left','bottom']:ax.spines[name].set_color('#b8c3d0')
  ax.legend(loc='best',frameon=False,fontsize=7.7,handlelength=2.1)
  extra=' (non-zero origin)' if kind=='mean_reward' and limits[0]!=0 else ''
  fig.text(x0,y0-.067,f'Display: updates {start}–{last}; y = {limits[0]:.3f} to {limits[1]:.3f}{extra}',
           fontsize=8.1,color='#768191')
  stats[key]['display_y_limits']=list(limits)
 foot='Training proxies from fixed Judges; not human accuracy. Invalid completions count in the denominator. '
 foot+=('C = effective term before weighting (zero contribution when skipped). Total = 0.2F + 0.4E + 0.4C.' if kind=='mean_reward'
        else 'C = scored C passes / all slots; C scored only if E > 0.50. Total = all three pass / all slots.')
 fig.text(.5,.025,foot,ha='center',fontsize=8.6,color='#718096')
 stem=f"{kind}_job{data['job_id']}_step{last:03d}_four_panels"
 files={}
 for ext in ['png','pdf','svg']:
  target=destination/(stem+'.'+ext);fig.savefig(target,facecolor='white');files[ext]=str(target)
 plt.close(fig)
 return {'files':files,'statistics':stats,'display_x_range':[start,last]}

def main():
 p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
 a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 marked=[x.split(' ',1)[1] for x in a.result.read_text(encoding='utf-8-sig').splitlines() if x.startswith('ARIA3_EIGHT_CURVE_DATA ')]
 if len(marked)!=1:raise ValueError('需要唯一新鲜的8曲线派生统计')
 d=json.loads(marked[0]);n=d['completed_updates']
 assert d['candidate_slots']==8*n and len(d['points'])==n
 for point in d['points']:
  assert point['candidate_count']==8
  for k in KEYS:assert point['pass_rate'][k]==point['pass_count'][k]/8
  assert math.isclose(point['mean_reward']['Total'],.2*point['mean_reward']['F']+.4*point['mean_reward']['E']+.4*point['mean_reward']['C'],abs_tol=1e-12)
 plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
 figures={kind:render(d,kind,a.output_dir) for kind in ['mean_reward','pass_rate']}
 stem=f"eight_curves_job{d['job_id']}_step{n:03d}"
 (a.output_dir/(stem+'_data.json')).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
 summary={k:v for k,v in d.items() if k!='points'};summary['figures']=figures
 (a.output_dir/(stem+'_summary.json')).write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 fields=['step','candidate_count','invalid_count']+[f'{kind}_{key}' for kind in ['mean_reward','pass_rate','pass_count'] for key in KEYS]
 with (a.output_dir/(stem+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
  for point in d['points']:writer.writerow({k:point[k] for k in fields[:3]}|{f'{kind}_{key}':point[kind][key] for kind in ['mean_reward','pass_rate','pass_count'] for key in KEYS})
 print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
