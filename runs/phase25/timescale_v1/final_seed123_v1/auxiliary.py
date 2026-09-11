"""Fixed lag/covariance and full trajectories; no new generation or training loss."""
import os,json,csv
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/hirp25_final_matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path('runs/phase25/timescale_v1');f=r/'final_seed123_v1';out=f/'auxiliary';out.mkdir(exist_ok=False)
man=json.load(open('runs/phase24/evaluation_v1/multitarget_development_manifest.json'));targets=json.load(open('runs/phase24/evaluation_v1/processed_targets/completed.json'))['result']['files']
results={}
for arm in ('C0','C1','C2'):
 p=(f if arm=='C0' else r)/'task_multitarget'/f'seed123_{arm}_step6000.json';results[arm]=json.load(open(p))['result']
def covariance(x):
 z=x.astype(np.float64)-x.mean(1,keepdims=True);return np.einsum('kti,ktj->kij',z,z)/x.shape[1]
def corr(x,lag):
 a=x[:,:-lag].astype(np.float64);b=x[:,lag:].astype(np.float64);a-=a.mean(1,keepdims=True);b-=b.mean(1,keepdims=True)
 den=np.sqrt((a*a).sum(1)*(b*b).sum(1));num=(a*b).sum(1)
 return np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
rows=[]
for i,s in enumerate(man['sources']):
 y=np.load(targets[i]['path']);real_cov=covariance(y).mean(0)
 for arm in ('C0','C1','C2'):
  x=np.load(results[arm]['exports'][i]['path']);row=dict(arm=arm,index=i,session=s['clip_id'].split('/')[1],length=s['length'],covariance_RMSE=float(np.sqrt(np.mean((covariance(x).mean(0)-real_cov)**2))))
  for lag in (1,10,30):
   aa=corr(x,lag);bb=corr(y,lag)
   # Compare only channels whose lag statistic is defined for every slot.
   valid=np.isfinite(aa).all(0)&np.isfinite(bb).all(0)
   row[f'autocorr_lag{lag}_MAE']=float(np.abs(aa[:,valid].mean(0)-bb[:,valid].mean(0)).mean()) if valid.any() else None
   row[f'autocorr_lag{lag}_valid_channels']=int(valid.sum())
  rows.append(row)
 if i in (0,20,40,60):
  paired=y[0];fig,axs=plt.subplots(3,3,figsize=(15,8),sharex=True)
  for a,arm in enumerate(('C0','C1','C2')):
   x=np.load(results[arm]['exports'][i]['path'])
   for b,(ch,name) in enumerate(((0,'AU channel0'),(15,'VA channel0'),(17,'expression channel0'))):
    ax=axs[a,b]
    for k in range(10):ax.plot(x[k,:,ch],color='tab:blue',alpha=.18,linewidth=.5)
    ax.plot(x[0,:,ch],color='tab:blue',linewidth=.7,label='sample0 (fixed order)');ax.plot(paired[:,ch],color='black',linewidth=.8,label='paired target')
    for boundary in range(750,len(paired),750):ax.axvline(boundary,color='gray',alpha=.3,linewidth=.5)
    ax.set_title(arm+' / '+name)
    if b==0:ax.legend(fontsize=7)
  fig.suptitle(s['clip_id']+' / full length / seed123 step6000 K10');fig.tight_layout();fig.savefig(out/f'full_trajectory_case{i}.png',dpi=140);plt.close(fig)
with (out/'per_input.csv').open('x') as ff:
 w=csv.DictWriter(ff,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with (out/'protocol.json').open('x') as ff:json.dump(dict(source='existing full K10 exports, no new generation',targets='frozen Processor 10 target slots with original multiplicity',metrics='per-input mean channel covariance RMSE and lag1/10/30 autocorrelation MAE between generated and target slot averages; undefined constant channels excluded and counted',units='raw channel units; no new fitted scale or tuned threshold',cases=[0,20,40,60],visual_channels=[0,15,17],limitations='auxiliary temporal diagnostics, not official metrics or semantic mode/probability recovery; processed target temporal statistics inherit Processor behavior'),ff,indent=2)
print('AUX COMPLETE')
