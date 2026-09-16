"""Render reported single-request throughput ratios, with separate baselines."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parents[2]
out=Path(__file__).parent
# Both panels read the 16 September 2026 snapshots, so the figure, the mapper
# ablation and the EAGLE results all come from one evaluation cohort. Those
# documents name datasets directly in panel (a). Panel (b) writes `Math` for the
# MATH dataset and `GSM8K` for GSM8K, so resolve its labels explicitly and keep
# both the raw domain and the resolved dataset in the saved data.
PHASE2_DATASET={'math':'MATH','gsm8k':'GSM8K','code':'Code','chat':'Chat'}
def ratio(cell): return float(cell.replace('×','').strip())
rows=[]
for line in (root/'new_phase1.md').read_text().splitlines():
 c=[x.strip() for x in line.split('|')]
 if len(c)==10 and c[1] in ('GSM8K','KiCad') and c[2]=='5W AUF' and c[5].endswith('\u00d7'):
  rows.append({'domain':c[1],'dataset':c[1],'relayspec':ratio(c[5]),'mean_tps':float(c[3])})
assert len(rows)==2,rows
rows.sort(key=lambda r: ['GSM8K','KiCad'].index(r['domain']))
native,mapped={},{}
for line in (root/'new_phase2.md').read_text().splitlines():
 c=[x.strip() for x in line.split('|')]
 if len(c)==7 and c[2] in ('DFlash native','5W MSE25') and c[5].endswith('\u00d7'):
  (native if c[2]=='DFlash native' else mapped)[c[1].lower()]=ratio(c[5])
assert set(native)==set(mapped)==set(PHASE2_DATASET),(native,mapped)
cross=[{'domain':k,'dataset':PHASE2_DATASET[k],'native':native[k],'relayspec':mapped[k]} for k in PHASE2_DATASET]
(out/'speedups_data.json').write_text(json.dumps({'lora':rows,'cross_size':cross,'metric':'ratios of arithmetic mean per-request tokens/second','domain_naming':'`domain` is the raw report string. In new_phase2.md `Math` is the MATH dataset and `GSM8K` is GSM8K. Use `dataset` for labels and tables.','sources':['new_phase1.md: 5W AUF versus DFlash native, 16 September 2026 snapshot','new_phase2.md: 5W MSE25 and DFlash native versus autoregressive, 16 September 2026 snapshot']},indent=2)+'\n')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
fig,(a,b)=plt.subplots(1,2,figsize=(7.6,2.55),gridspec_kw={'width_ratios':[1,1.65]})
blue='#266a94'; gray='#bed0dc'
x=np.arange(2); vals=[r['relayspec'] for r in rows]
a.bar(x-.17,[1,1],.32,color=gray,label='Unchanged DFlash')
a.bar(x+.17,vals,.32,color=blue,label='RelaySpec')
for xx,v in zip(x,vals):a.text(xx+.17,v+.05,f'{v:.2f}×',ha='center',fontsize=9,weight='bold')
a.set(xticks=x,xticklabels=[r['dataset'] for r in rows],ylim=(0,2.65),ylabel='Throughput / unchanged DFlash')
a.set_title('(a) LoRA-adapted Qwen3-4B',loc='left',fontsize=10,weight='bold')
a.legend(loc='upper left',fontsize=7,frameon=False)
x=np.arange(4); nv=[r['native'] for r in cross];rv=[r['relayspec'] for r in cross]
b.bar(x-.18,nv,.34,color=gray,label='Native 8B drafter')
b.bar(x+.18,rv,.34,color=blue,label='RelaySpec (4B drafter)')
for xx,v in zip(x,rv):b.text(xx+.18,v+.13,f'{v:.2f}×',ha='center',fontsize=9,weight='bold')
b.set(xticks=x,xticklabels=[r['dataset'] for r in cross],ylim=(0,7.8),ylabel='Throughput / autoregressive target')
b.set_title('(b) Qwen3-4B drafter → Qwen3-8B target',loc='left',fontsize=10,weight='bold')
b.legend(loc='upper right',fontsize=7,frameon=False)
for ax in (a,b):
 ax.spines[['top','right']].set_visible(False)
 ax.axhline(1,color='#707070',lw=.7,ls='--',zorder=0)
 ax.set_axisbelow(True);ax.yaxis.grid(True,alpha=.14)
fig.tight_layout(w_pad=1.5)
fig.savefig(out/'relayspec_speedups.pdf',bbox_inches='tight',pad_inches=.025)
fig.savefig(out/'relayspec_speedups.png',dpi=180,bbox_inches='tight',pad_inches=.025)
