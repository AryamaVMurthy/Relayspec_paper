"""Supervision budget for both adaptation settings, from 16 to 16,384 examples."""
from pathlib import Path
import json, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2]
out=Path(__file__).parent

# --- fine-tuned targets: percent gain over the unadapted drafter (phase 1 scaling export, 4k-16k)
txt=(root/'phase1_results.md').read_text()
blk=txt[txt.index('### scaling'):txt.index('### serving')]
rows=[[c.strip() for c in l.strip('|').split('|')] for l in blk.splitlines() if l.startswith('|')]
h=rows[0]; ix={n:h.index(n) for n in ['domain','architecture','examples','anchors','speedup_pct_vs_native','loss']}
F={}
for r in rows[2:]:
 if len(r)<len(h) or r[ix['loss']]!='AUF': continue
 try: g=float(r[ix['speedup_pct_vs_native']])
 except ValueError: continue
 arch='maps' if r[ix['architecture']].startswith('maps') else 'body'
 F[(r[ix['domain']],arch,int(r[ix['examples']]),int(r[ix['anchors']]))]=g

# --- cross-model transfer: throughput relative to the native drafter (phase 2 MSE50 coverage, 4k-16k)
txt2=(root/'phase2_results.md').read_text()
blk2=txt2[txt2.index('## MSE 50% coverage results'):txt2.index('### Completed training')]
rows2=[[c.strip() for c in l.strip('|').split('|')] for l in blk2.splitlines() if l.startswith('|')]
h2=rows2[0]; jx={n:h2.index(n) for n in ['Fit','Workload','× native (mean TPS)']}
C={}
for r in rows2[2:]:
 if len(r)<len(h2): continue
 f=r[jx['Fit']]
 if not f.startswith('T1_mse50_n') or not f.endswith('_e3_lr0.001'): continue
 C[(int(f.split('_n')[1].split('_')[0]), r[jx['Workload']])]=float(r[jx['× native (mean TPS)']])

# --- small-data extension: 16 to 2,048 examples, same native/AR controls reused
small=(root/'small_scaling.md').read_text()
NATIVE_FT={'math':128.90,'kicad':168.48,'nanocoder':150.79}
NATIVE_T1={'math':219.79,'gsm':177.94,'code':151.63,'chat':89.50}
def parse_block(name):
    i=small.index('|', small.index(name)+len(name))
    j=small.index('\n\n',i)
    return [[c.strip() for c in l.strip('|').split('|')] for l in small[i:j].splitlines() if l.startswith('|')][2:]
SF={}
for r in parse_block('**LoRA target adapters, AUF**'):
    dom,ex,tps=r[0],int(r[1].replace(',','')),float(r[4])
    SF[(dom,'maps',ex)]=100*(tps/NATIVE_FT[dom]-1)
SC={}
for r in parse_block('**T1 cross-size transfer, MSE**'):
    wl,ex,tps=r[0],int(r[1].replace(',','')),float(r[4])
    SC[(ex,wl)]=tps/NATIVE_T1[wl]

EX_FT=[4096,8192,12288,16384]; EX_T1=[4096,8192,16384]
SM=[16,128,512,1024,2048]
ALL_FT=SM+EX_FT; ALL_T1=SM+EX_T1
AN=[8,24,32,64,128]
DOM=[('kicad','KiCad','#266a94'),('math','GSM8K','#c2703a'),('nanocoder','NanoCoder','#5b8c5a')]
WL=[('math','MATH','#266a94'),('gsm','GSM8K','#c2703a'),('code','Code','#5b8c5a'),('chat','Chat','#8c6a9c')]
for d,_,_ in DOM:
 assert all((d,'maps',n) in SF for n in SM),d
 for a in ('maps','body'): assert all((d,a,4096,k) in F for k in AN),(d,a)
for n in EX_T1: assert all((n,w) in C for w,_,_ in WL),n
for n in SM: assert all((n,w) in SC for w,_,_ in WL),n

json.dump({'fine_tuned_metric':'percent gain in mean per-request tokens/s over the unadapted drafter',
 'transfer_metric':'mean per-request tokens/s divided by the native drafter',
 'sources':['phase1_results.md: scaling export (4k-16k)','phase2_results.md: MSE 50% coverage results (4k-16k)','small_scaling.md: 16-2,048 example extension, same native/AR controls reused'],
 'fine_tuned_examples':[{'domain':d,'examples':n,'gain_percent':(SF[(d,'maps',n)] if n in SM else F[(d,'maps',n,32)])} for d,_,_ in DOM for n in ALL_FT],
 'fine_tuned_anchors':[{'domain':d,'arch':a,'anchors':k,'gain_percent':F[(d,a,4096,k)]} for d,_,_ in DOM for a in ('maps','body') for k in AN],
 'transfer_examples':[{'examples':n,'workload':w,'x_native':(SC[(n,w)] if n in SM else C[(n,w)])} for n in ALL_T1 for w,_,_ in WL]},
 open(out/'scaling_data.json','w'),indent=2)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'pdf.fonttype':42})
fig,(a,b,c)=plt.subplots(1,3,figsize=(7.6,2.35))
for d,lab,col in DOM:
 y=[SF[(d,'maps',n)] if n in SM else F[(d,'maps',n,32)] for n in ALL_FT]
 a.plot(range(len(ALL_FT)),y,'-o',color=col,ms=3.2,lw=1.5,label=lab)
a.set(xticks=range(len(ALL_FT)),xticklabels=[f'{x//1024}k' if x>=1024 else str(x) for x in ALL_FT],
      xlabel='Unique examples (log-spaced)',ylim=(0,178))
a.set_ylabel('Gain over unadapted drafter (%)')
a.set_title('(a) Fine-tuned: 16 to 16k examples',loc='left',fontsize=9,weight='bold')
a.legend(fontsize=6.4,frameon=False,loc='upper left',handlelength=1.5)
a.tick_params(axis='x',labelrotation=35)

for d,lab,col in DOM:
 y=[F[(d,'maps',4096,k)] for k in AN]
 yb=[F[(d,'body',4096,k)] for k in AN]
 b.plot(range(len(AN)),y,'-o',color=col,ms=3.2,lw=1.5,label=f'{lab}, maps')
 b.plot(range(len(AN)),yb,'--s',color=col,ms=2.8,lw=1.0,alpha=.7,label=f'{lab}, body LoRA')
b.set(xticks=range(len(AN)),xticklabels=[str(x) for x in AN],xlabel='Anchors per example (4,096 ex.)',ylim=(0,178))
b.set_title('(b) Fine-tuned: more anchors',loc='left',fontsize=9,weight='bold')
b.legend(fontsize=5.6,frameon=False,ncol=2,loc='upper left',columnspacing=.6,handlelength=1.3)

for w,lab,col in WL:
 y=[100*(SC[(n,w)] if n in SM else C[(n,w)]) for n in ALL_T1]
 c.plot(range(len(ALL_T1)),y,'-o',color=col,ms=3.2,lw=1.5,label=lab)
c.axhline(100,color='#707070',lw=.7,ls='--',zorder=0)
c.set(xticks=range(len(ALL_T1)),xticklabels=[f'{x//1024}k' if x>=1024 else str(x) for x in ALL_T1],
      xlabel='Unique examples (log-spaced)',ylim=(0,108))
c.set_ylabel('Native drafter throughput (%)')
c.set_title('(c) Transfer: 16 to 16k examples',loc='left',fontsize=9,weight='bold')
c.legend(fontsize=6.0,frameon=False,ncol=2,loc='lower right',columnspacing=.6,handlelength=1.3)
c.tick_params(axis='x',labelrotation=35)
a.tick_params(axis='x',labelrotation=35)

for ax in (a,b,c):
 ax.spines[['top','right']].set_visible(False)
 ax.set_axisbelow(True); ax.yaxis.grid(True,alpha=.14)
fig.tight_layout(w_pad=1.1)
fig.savefig(out/'relayspec_scaling.pdf',bbox_inches='tight',pad_inches=.02)
