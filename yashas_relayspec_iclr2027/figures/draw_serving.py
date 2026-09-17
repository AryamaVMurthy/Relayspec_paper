"""Shared multi-LoRA serving: the two throughput conventions diverge with load."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2]
out=Path(__file__).parent
txt=(root/'phase1_results.md').read_text()
blk=txt[txt.index('### serving'):txt.index('### eight standalone')]
rows=[[c.strip() for c in l.strip('|').split('|')] for l in blk.splitlines() if l.startswith('|')]
h=rows[0]
ix={n:h.index(n) for n in ['trial','policy','concurrency','method','mean_request_tps','aggregate_serving_tps','acceptance']}
S={}
for r in rows[2:]:
 if len(r)<len(h): continue
 if r[ix['trial']]!='peak_clients_trial0' or r[ix['policy']]!='oracle': continue
 S[(r[ix['method']],int(r[ix['concurrency']]))]=(float(r[ix['mean_request_tps']]),float(r[ix['aggregate_serving_tps']]))
C=[1,4,8,16,32]
M=[('ar','Autoregressive','#9aa7b0','^'),('native','Unchanged drafter','#7fa8c4','s'),('mapped','RelaySpec maps','#266a94','o')]
for m,_,_,_ in M: assert all((m,c) in S for c in C),m
json.dump({'setting':'Qwen3-4B, three LoRA targets, oracle routing, 384 requests, vLLM on one L40S',
 'source':'phase1_results.md: serving export, trial peak_clients_trial0, oracle policy',
 'aggregate_definition':'returned tokens divided by workload wall time including drain',
 'rows':[{'method':m,'clients':c,'mean_request_tps':S[(m,c)][0],'aggregate_tps':S[(m,c)][1]}
         for m,_,_,_ in M for c in C]},open(out/'serving_data.json','w'),indent=2)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
fig,(a,b)=plt.subplots(1,2,figsize=(7.6,2.6))
for m,lab,col,mk in M:
 a.plot(range(len(C)),[S[(m,c)][1] for c in C],'-',marker=mk,color=col,ms=4,lw=1.6,label=lab)
 b.plot(range(len(C)),[S[(m,c)][0] for c in C],'-',marker=mk,color=col,ms=4,lw=1.6,label=lab)
a.set(xticks=range(len(C)),xticklabels=C,xlabel='Concurrent clients',ylim=(0,4700))
a.set_ylabel('Aggregate tokens/s')
a.set_title('(a) Server throughput',loc='left',fontsize=10,weight='bold')
b.set(xticks=range(len(C)),xticklabels=C,xlabel='Concurrent clients',ylim=(0,420))
b.set_ylabel('Mean client-side per-request tokens/s')
b.set_title('(b) Client-side per-request throughput',loc='left',fontsize=10,weight='bold')
for ax in (a,b):
 ax.spines[['top','right']].set_visible(False)
 ax.set_axisbelow(True); ax.yaxis.grid(True,alpha=.14)
 ax.legend(fontsize=7,frameon=False)
fig.tight_layout(w_pad=1.4)
fig.savefig(out/'relayspec_serving.pdf',bbox_inches='tight',pad_inches=.025)
