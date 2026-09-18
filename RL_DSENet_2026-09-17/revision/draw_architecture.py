"""Draw the implemented D05/D06 architecture; no proposed modules in the figure."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
fig, ax = plt.subplots(figsize=(18, 11))
ax.set(xlim=(0, 18), ylim=(0, 11)); ax.axis('off')
navy, blue, teal, amber = '#172f46', '#e8f1fc', '#e6f5ef', '#fff1d8'
def box(x,y,w,h,title,body='',color=blue,fs=11):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06,rounding_size=0.10',
                               facecolor=color,edgecolor='#91a6b7',linewidth=1.1))
    ax.text(x+w/2,y+h-.23,title,ha='center',va='top',fontsize=fs+1,weight='bold',color=navy)
    if body: ax.text(x+w/2,y+h/2-.14,body,ha='center',va='center',fontsize=fs,color=navy,linespacing=1.4)
def arrow(p,q,dashed=False,label=None):
    ax.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=14,lw=1.4,
                               color=navy,linestyle='--' if dashed else '-'))
    if label: ax.text((p[0]+q[0])/2,(p[1]+q[1])/2+.13,label,ha='center',fontsize=9,color=navy)
ax.text(.2,10.65,'DSENet + model-based basal-insulin policy',fontsize=23,weight='bold',color=navy)
ax.text(.2,10.23,'Implemented single-seed model  |  5-min observations  |  4-h forecast / planning horizon',fontsize=12,color='#4c6578')
box(.2,7.65,2.55,1.9,'Observed history', '72 × 22 features\n6 h CGM + insulin\nfood / exercise + masks\n+ ages / timing',fs=11)
box(3.35,8.6,3.2,.95,'CGM normalization','Observed-only RevIN',fs=10)
box(3.35,6.8,3.2,1.3,'RL-DITR history encoder','Frozen Transformer\ncontext z: 256 dimensions',color=teal,fs=10)
arrow((2.8,8.9),(3.3,9.05)); arrow((2.8,8.15),(3.3,7.5))
box(7.05,8.55,3.3,1,'Global stream','72 points · patch / stride = 12 / 3\nBidirectional gated Mamba',fs=10)
box(7.05,7.1,3.3,1,'Local stream','Last 36 points · patch / stride = 2 / 1\nLoRE local attention',fs=10)
arrow((6.6,9.05),(7,9.05)); arrow((6.55,8.9),(7,7.8))
box(10.85,7.7,2.65,1.45,'Router + Fusion','Adaptive stream weights\nConcatenate + linear head\nInverse RevIN',fs=10)
arrow((10.4,9.05),(10.8,8.7)); arrow((10.4,7.6),(10.8,8.1))
box(14.1,7.7,3.5,1.45,'DSENet forecast','48 future CGM values\nUnconditioned on candidate actions',fs=10)
arrow((13.55,8.4),(14.05,8.4))
box(7.05,5.05,4.0,1.35,'Reference calibration','DSENet + MLP(z, forecast, current rate)\n48-step reference trajectory',color=teal,fs=10)
arrow((6.6,7.1),(7.5,6.45)); arrow((15.8,7.65),(11.05,5.95))
ax.text(14.0,6.5,'Current actual basal is the\ncounterfactual reference rate.',fontsize=10,color='#4c6578')
box(.2,4.55,5.1,1.45,'Seven timed candidate plans','Hold anchor; or ±0.25 U/h in one\nof three 80-min blocks (7 × 48 actions).\nAnchor = basal observed after warm-up.',color=amber,fs=10)
box(.2,2.1,5.1,1.55,'Causal action-response model','Context-conditioned Bernstein convolution\nΔG(z, candidate − current basal)\nZero intervention → zero response',color=teal,fs=10)
arrow((2.75,4.5),(2.75,3.7),True)
box(6.15,2.1,4.4,1.55,'Trajectory return → policy training','G(candidate) = reference + ΔG\nQ = discounted RL-DITR status reward\nMaximize Eπ[Q] − 0.05 KL(π || prior)',color=amber,fs=10)
arrow((5.35,2.9),(6.1,2.9),True); arrow((8.5,5),(8.5,3.7),True)
box(11.65,4.55,5.95,1.45,'Learned actor: 306 → 128 → 7','Input: z, reference trajectory, current basal, anchor\nChoose argmax π; execute first 5 min only\nRe-observe and repeat',color=blue,fs=11)
arrow((11.1,5.65),(11.6,5.3)); arrow((10.6,3.05),(12.3,4.5),True)
box(11.65,2.1,5.95,1.55,'Independent simglucose environment','Requested basal U/h → pump-delivered insulin\nShared meal bolus protocol; new CGM observation\nTrue BG used only for evaluation',color='#f1f2f4',fs=10)
arrow((14.6,4.5),(14.6,3.7))
ax.text(.25,1.35,'TRAINING',fontsize=11,weight='bold',color=navy)
ax.text(2.0,1.35,'1  DSENet: Loop prediction   →   2  Reference + response: paired simulation   →   3  Actor: frozen-world return',fontsize=11,color=navy)
ax.text(.25,.83,'Solid arrows: inference data flow. Dashed arrows: candidate scoring / policy training. The deployed actor does not score candidates with Q.',fontsize=10,color='#4c6578')
ax.text(.25,.4,'The inherited glucose / reward / value heads are unused. No learned terminal value bootstrap. This is a task adaptation, not the original clinical RL-DITR.',fontsize=10,color='#4c6578')
fig.subplots_adjust(left=.02,right=.99,top=.98,bottom=.01)
out=ROOT/'figures';out.mkdir(exist_ok=True)
for ext in ('svg','pdf','png'):fig.savefig(out/f'architecture_actual.{ext}',dpi=180,facecolor='white')
print(out/'architecture_actual.png')
