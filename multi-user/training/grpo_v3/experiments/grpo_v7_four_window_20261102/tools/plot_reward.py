"""绘制本次只读快照的奖励总览及原始分量，不修改训练或监控。"""
import json,csv,textwrap
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator,FormatStrFormatter
ROOT=Path(__file__).resolve().parent
d=json.loads((ROOT/'reward_snapshot.json').read_text(encoding='utf-8'))
rows=d['updates'];N=d['completed_global_step']
assert [r['global_step'] for r in rows]==list(range(1,N+1))
x=np.array([r['global_step'] for r in rows])
weighted={k:np.array([r['weighted_mean'][k] for r in rows],float) for k in ['F','E','C','speaker','J','R']}
raw={k:np.array([np.nan if r['raw_valid_mean'][k] is None else r['raw_valid_mean'][k] for r in rows]) for k in ['F','E','C','speaker']}
assert np.allclose(weighted['F']+weighted['E']+weighted['C']+weighted['speaker'],weighted['R'],rtol=0,atol=1e-10)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11.5,
    'axes.titleweight':'semibold','axes.labelsize':10,'axes.edgecolor':'#CFD6E0',
    'xtick.color':'#526078','ytick.color':'#526078','axes.labelcolor':'#3A4860',
    'text.color':'#1D2D46','pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'#FAFCFF'})
BLUE='#8CBCE5';ORANGE='#E88C32';GREY='#8993A2'
xmin=20 if N>=100 else 1
limits={}
def full_ma(y):
    answer=np.full(len(y),np.nan)
    for i in range(9,len(y)):
        window=y[i-9:i+1]
        if np.isfinite(window).all():answer[i]=window.mean()
    return answer
def basic_axis(ax,y,key,title,ylabel,reference=True):
    mask=(x>=xmin)&np.isfinite(y)
    vals=y[mask];span=float(np.ptp(vals)) if len(vals) else 0
    padding=max(span*.14,.008 if key!='speaker' else .012)
    lo=float(vals.min())-padding;hi=float(vals.max())+padding
    if hi<=lo:hi=lo+.02
    ax.set_ylim(lo,hi);ax.set_xlim(xmin-.25,N+.25)
    limits[key]=(lo,hi)
    ax.plot(x,y,color=BLUE,lw=1.3,marker='o',ms=3.4,label='Per-update mean',zorder=3)
    ma=full_ma(y)
    ax.plot(x,ma,color=ORANGE,lw=2.5,label='10-update trailing mean',zorder=4)
    if reference and N>=10 and np.isfinite(y[:10]).all():
        ax.axhline(y[:10].mean(),color=GREY,ls=(0,(4,3)),lw=1.15,label='Steps 1–10 training reference',zorder=2)
    ax.grid(axis='y',color='#E3E9F1',lw=.8,zorder=0)
    ax.spines[['top','right']].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=14))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f' if key!='speaker_raw' else '%.2f'))
    ax.set_title(title,loc='left',pad=10)
    ax.set_ylabel(ylabel)
    if xmin<=3.5<=N:
        ax.axvline(3.5,color='#B6C6D7',ls=':',lw=.9,zorder=1)
    return ma
def stats_header(fig):
    a=weighted['R']
    stats=[('Steps 1–10',a[:10].mean() if N>=10 else None),
           ('Last 10 updates',a[-10:].mean() if N>=10 else None),
           ('All logged updates',a.mean())]
    for i,(label,value) in enumerate(stats):
        left=.068+i*.307
        fig.add_artist(FancyBboxPatch((left,.828),.286,.065,boxstyle='round,pad=0.008,rounding_size=0.01',
            transform=fig.transFigure,facecolor='#EFF4FA',edgecolor='#DFE7F1',lw=.8))
        fig.text(left+.012,.872,label,size=10,color='#526078')
        fig.text(left+.012,.840,'N/A' if value is None else f'{value:.4f}',size=19,weight='bold',color='#203D61')
fig=plt.figure(figsize=(13.6,11.2),facecolor='#FAFCFF')
fig.text(.065,.965,'GRPO v7  Training reward and four contributions',size=18,weight='bold')
fig.text(.065,.936,f'Completed global updates 1–{N} / 180  |  16 original candidate slots per update',size=11,color='#526078')
stamp=d['captured_at_beijing'][:19].replace('T',' ')
fig.text(.065,.914,'Jobs 21644657 (steps 1–3) → 21650205 (steps 4 onward)',size=10,color='#526078')
fig.text(.975,.914,f'Snapshot: {stamp} Beijing',size=10,color='#526078',ha='right')
stats_header(fig)
gs=fig.add_gridspec(3,2,left=.075,right=.975,bottom=.185,top=.785,hspace=.48,wspace=.22,
                   height_ratios=[1.18,1,1])
ax0=fig.add_subplot(gs[0,:])
basic_axis(ax0,weighted['R'],'R','Total reward  R = F + E + C + speaker contribution','Mean training reward')
ax0.legend(loc='upper right',ncol=3,fontsize=8.8,frameon=False,handlelength=2.4,columnspacing=1.5)
if N>=4:
    ax0.text(3.62,.95,'Resume from checkpoint-3',transform=ax0.get_xaxis_transform(),size=8,color='#697C94',va='top')
panels=[('F','F  Formality contribution  (0.2 × pF)'),
        ('E','E  Evidence contribution  (0.4 × pE)'),
        ('C','C  Cross-view answerability contribution  (0.4 × pC)'),
        ('speaker','Speaker-only contribution  (−λ × hit)')]
for i,(key,title) in enumerate(panels):
    ax=fig.add_subplot(gs[1+i//2,i%2])
    basic_axis(ax,weighted[key],key,title,'Mean reward contribution')
    if i>=2:ax.set_xlabel('Global optimizer update')
    if key=='speaker':
        ax.text(.98,.90,'Closer to zero = less penalty',transform=ax.transAxes,ha='right',size=9,color='#66758B')
ax0.set_xlabel('Global optimizer update')
ranges='; '.join(f'{key}: [{lo:.3f}, {hi:.3f}]' for key,(lo,hi) in limits.items())
footer=[
f'Displayed x: {xmin}–{N}. Y ranges: '+ranges+'.',
'Y axes are zoomed and may start above or below zero. 10-update means use complete trailing windows only.',
'All 16 slots enter contribution means; invalid generations contribute actual zeros. These are training proxy rewards, not validation accuracy.',
'Steps 1–3 are retained training updates from Job 21644657 (original Slurm status: FAILED/1:0). Current Job 21650205: RUNNING.'
]
fy=.112
for line in footer:
    wrapped=textwrap.wrap(line,176)
    for part in wrapped:
        fig.text(.065,fy,part,size=8.2,color='#65758D');fy-=.016
fig.savefig(ROOT/'reward_components_overview.png',dpi=180)
fig.savefig(ROOT/'reward_components_overview.pdf')
plt.close(fig)
# 原始评审概率与speaker命中率：只对实际取得该分量的有效记录取均值，不补造缺失评分。
fig,axs=plt.subplots(2,2,figsize=(13.6,8.5),facecolor='#FAFCFF')
fig.subplots_adjust(left=.075,right=.975,bottom=.21,top=.805,hspace=.38,wspace=.22)
fig.text(.065,.957,'GRPO v7  Raw judge scores and speaker-only hit rate',size=18,weight='bold')
fig.text(.065,.925,f'Completed global updates 1–{N} / 180  |  Snapshot: {stamp} Beijing',size=10.5,color='#526078')
fig.text(.065,.898,'Jobs 21644657 → 21650205  |  Scores below are unweighted; the total reward uses the weighted contributions in the overview.',size=10,color='#526078')
raw_limits={}
for ax,key,title in zip(axs.flat,['F','E','C','speaker'],[
        'F  Formality pass probability','E  Evidence-groundedness pass probability',
        'C  Cross-view answerability pass probability','Speaker-only correct-answer rate']):
    plotkey='speaker_raw' if key=='speaker' else 'raw_'+key
    basic_axis(ax,raw[key],plotkey,title,'Valid-answer hit rate' if key=='speaker' else 'Mean judge pass probability')
    raw_limits[key]=limits[plotkey]
    ax.set_xlabel('Global optimizer update')
    if key=='speaker':
        ax.axhline(.2,color='#B56583',ls=(0,(2,3)),lw=1.0,label='Chance = 0.20 (5 choices)')
        lo,hi=ax.get_ylim()
        ax.set_ylim(min(lo,.18),hi)
        raw_limits[key]=ax.get_ylim()
handles,labels=axs[1,1].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.065,.878),ncol=4,fontsize=9,
           frameon=False,handlelength=2.4,columnspacing=1.8)
rawfooter=[
'F/E/C: mean over scored QA only. Speaker-only: correct / valid blind answers; parsing failures and invalid QA are excluded.',
'No missing judge scores are filled with zero. Weighted contributions in the overview retain all original candidate slots.',
f'Displayed x: {xmin}–{N}. Y ranges: '+'; '.join(f'{key}: [{lo:.3f}, {hi:.3f}]' for key,(lo,hi) in raw_limits.items())+'.',
'Raw per-update means (blue), complete 10-update trailing means (orange), Steps 1–10 training references (gray); axes may start above zero.'
]
fy=.12
for line in rawfooter:
    for part in textwrap.wrap(line,176):
        fig.text(.065,fy,part,size=8.5,color='#65758D');fy-=.018
fig.savefig(ROOT/'raw_reward_components.png',dpi=180)
fig.savefig(ROOT/'raw_reward_components.pdf')
plt.close(fig)
fields=['step','source_job','slots','legal_qa','speaker_valid','speaker_hits','speaker_lambda',
        'F_contribution','E_contribution','C_contribution','speaker_contribution','quality_J','total_R',
        'raw_F_valid_mean','raw_E_valid_mean','raw_C_valid_mean','speaker_valid_hit_rate']
with (ROOT/'reward_by_update.csv').open('w',newline='',encoding='utf-8-sig') as stream:
    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
    for r in rows:
        values=[r['global_step'],r['source_job'],r['slots'],r['legal_qa'],r['speaker_valid'],r['speaker_hits'],r['speaker_lambda']]
        values+=[r['weighted_mean'][k] for k in ['F','E','C','speaker','J','R']]
        values+=[r['raw_valid_mean'][k] for k in ['F','E','C','speaker']]
        writer.writerow(dict(zip(fields,values)))
summary={'completed_steps':N,'slots':sum(r['slots'] for r in rows),'legal_qa':sum(r['legal_qa'] for r in rows),
    'total_reward_stats':{'Steps 1–10':float(weighted['R'][:10].mean()) if N>=10 else None,
        'Last 10 updates':float(weighted['R'][-10:].mean()) if N>=10 else None,'All logged updates':float(weighted['R'].mean())},
    'weighted_plot_y_ranges':{k:list(limits[k]) for k in ['R','F','E','C','speaker']},
    'raw_plot_y_ranges':{k:list(v) for k,v in raw_limits.items()},
    'checks':d['checks'],'captured_at_beijing':d['captured_at_beijing']}
(ROOT/'plot_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))

