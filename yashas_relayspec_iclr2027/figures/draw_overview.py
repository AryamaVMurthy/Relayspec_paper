from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(8,4.5))
ax.set(xlim=(0,17),ylim=(0,10.0));ax.axis('off')
blue='#e7f0f8'; orange='#ffead0'; ink='#24384b'; gray='#f5f6f7'
def box(x,y,w,h,label,color=blue,fs=10):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.025,rounding_size=0.07',linewidth=.8,edgecolor=ink,facecolor=color))
 ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=fs,color=ink)
def arrow(a,b):
 assert a[0]==b[0] or a[1]==b[1],(a,b)
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=9,linewidth=.9,color=ink,shrinkA=0,shrinkB=0))
ax.text(0,9.7,'A  Training: only the maps change',weight='bold',color=ink)
box(.05,5.0,2.0,3.25,'Changed\ntarget\n(frozen)')
ax.text(3.05,8.65,'Hidden\nstates',ha='center',fontsize=9)
ax.text(4.85,8.65,'Linear\nmaps',ha='center',fontsize=9)
ax.text(6.85,8.65,'Mapped\nstates',ha='center',fontsize=9)
for i in range(5):
 y=7.6-i*.59
 box(2.6,y,0.9,.43,f'$H_{i+1}$',gray)
 box(4.35,y,1.0,.43,f'$W_{i+1}$',orange)
 box(6.15,y,1.4,.43,f'$W_{i+1}H_{i+1}$',gray)
 for x0,x1 in [(2.075,2.55),(3.525,4.30),(5.375,6.10),(7.575,8.2)]:
  arrow((x0,y+.215),(x1,y+.215))
box(8.25,5.0,1.7,3.25,'Original\nfusion +\nRMSNorm')
arrow((9.975,6.6),(10.65,6.6))
box(10.7,5.5,2.45,2.2,'Frozen\nDFlash\ntransformer\n+ output head')
arrow((13.175,6.6),(13.85,6.6))
box(13.9,5.5,2.8,2.2,'Candidate\ntoken block',gray)
ax.text(11.925,8.65,'Token / mask embeddings',ha='center',fontsize=9)
arrow((11.925,8.4),(11.925,7.75))
box(.05,3.1,8.0,1.25,'LoRA-fine-tuned target\nIdentity maps + token supervision\nCorrect prefix and first incorrect proposal',gray,10)
box(8.55,3.1,8.15,1.25,'Different target model\nPaired text through new and original targets\nReconstruct original-target features and context',gray,10)
ax.text(0,2.5,'B  Inference: maps are folded into fusion',weight='bold',color=ink)
box(.05,.95,2.6,1.0,'New target\nhidden states')
arrow((2.675,1.45),(3.2,1.45))
box(3.25,.95,3.7,1.0,'Effective fusion\n'+r'$[F_1W_1\;\cdots\;F_5W_5]$','#eef5e8')
arrow((6.975,1.45),(7.5,1.45))
box(7.55,.95,3.7,1.0,'RMSNorm + frozen\nDFlash drafter')
arrow((11.275,1.45),(11.8,1.45))
box(11.85,.95,4.85,1.0,'Target verifies token proposals\nAccept prefix / correct rejection')
ax.text(8.4,.35,r'$H_i:\ d_t\!\times\! n\qquad W_i:\ d_r\!\times\! d_t\qquad W_iH_i:\ d_r\!\times\! n\qquad F_i:\ d_d\!\times\! d_r$',ha='center',fontsize=11,color=ink)
fig.subplots_adjust(left=.01,right=.99,bottom=.01,top=.99)
out=Path(__file__).parent
fig.savefig(out/'relayspec_overview.pdf',bbox_inches='tight',pad_inches=.03)
fig.savefig(out/'relayspec_overview.png',dpi=180,bbox_inches='tight',pad_inches=.03)
