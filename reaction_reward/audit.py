"""Frozen once-only RM_audit evaluation and independent fixed-pool value test."""
import random,collections
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .common import *
from .data import Data,load
from .model import Judge
from .train import evaluate,FAMILIES,ARMS

def summary(rows,T=1.):
 valid=[r for r in rows if r['weight']>0];unknown=len(rows)-len(valid)
 if not valid:return dict(valid=0,unknown=unknown,verdict='inconclusive')
 ds=np.array([r['difference']/T for r in valid]);groups=collections.defaultdict(list)
 for r,d in zip(valid,ds):groups[r['group']].append(float(d>0)+.5*float(d==0))
 means=np.array([np.mean(v) for v in groups.values()]);rng=np.random.default_rng(123);boot=np.mean(rng.choice(means,(1000,len(means)),replace=True),1)
 return dict(valid=len(valid),unknown=unknown,coverage=len(valid)/len(rows),accuracy=float(np.mean((ds>0)+.5*(ds==0))),pair_NLL=float(np.logaddexp(0,-ds).mean()),interaction_groups=len(groups),group_macro_accuracy=float(means.mean()),group_bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),scope='conservative recording-date component bootstrap',small_group_warning=len(groups)<10)

def temperatures(cal):
 out={}
 for fam in FAMILIES:
  d=np.array([r['difference'] for r in cal if r['family']==fam and r['weight']>0]);grid=np.exp(np.linspace(np.log(.05),np.log(20),121));out[fam]=float(min(grid,key=lambda t:np.logaddexp(0,-d/t).mean())) if len(d) else 1.
 return out

@torch.no_grad()
def candidates(model,data,w,name):
 rows=[dict(source=w['source'],start=w['start'],n=w['n'],A=dict(id=w['source'],start=w['start'],generated=str(OUT/'bank'/w['id']/(name+'.npy')),candidate=i)) for i in range(16)];return torch.cat([model(*data.batch(rows[i:i+4],'A')) for i in range(0,16,4)]).cpu().numpy()

def main():
 torch.set_num_threads(3);data=Data();records=read(OUT/'REAL_EVIDENCE.json')+read(OUT/'GENERATED_EVIDENCE.json');audit=[r for r in records if r['split']=='RM_audit'];windows=[w for w in read(OUT/'WINDOWS.json') if w['split']=='RM_audit'];allresults={}
 for arm in ARMS:
  dest=OUT/'training'/arm;selection=read(dest/'SELECTION.json');out=OUT/'audit'/arm;out.mkdir(parents=True,exist_ok=True)
  if (out/'complete.json').exists():allresults[arm]=read(out/'complete.json');continue
  model=Judge().cuda();state=torch.load(selection['selected_checkpoint'],map_location='cpu',weights_only=False);model.load_state_dict(state['model']);model.eval();cal=read(dest/f"cal_{selection['selected_step']:06d}.json");T=temperatures(cal);write(out/'CALIBRATION.json',dict(temperatures=T,source='RM_cal only',checkpoint_sha256=sha(selection['selected_checkpoint']),raw_score_is_probability=False))
  raw=evaluate(model,data,audit);write(out/'pair_scores.json',raw);tasks={f:summary([r for r in raw if r['family']==f],T[f]) for f in FAMILIES}
  for shift in [-4.,-2.,2.,4.]:tasks['shift_'+str(shift)]=summary([r for r in raw if r['family']=='temporal' and r['shift_seconds']==shift],T['temporal'])
  tasks['unseen_generator']=summary([r for r in raw if r['family']=='generated' and r['unseen']],T['generated'])
  pool_results=[]
  for w in windows:
   ref=candidates(model,data,w,'P2').sum(-1)/T['generated'];refmean=float(ref.mean())
   for name in ['P2','N0','N1']:
    ss=ref if name=='P2' else candidates(model,data,w,name).sum(-1)/T['generated'];metric=read(OUT/'bank'/w['id']/(name+'_scores.json'));C=np.array(metric['C']).max(1);D=np.array(metric['D']).min(1);Y=np.load(OUT/'bank'/w['id']/(name+'.npy'));rng=np.random.default_rng(seed('random10|'+w['id']+'|'+name))
    # Oracle is a GT-reading Pareto-front heuristic, explicitly not a method output.
    domination=np.array([np.sum((C>=C[i])&(D<=D[i])&((C>C[i])|(D<D[i]))) for i in range(16)])
    sets={'first10':np.arange(10),'random10':rng.choice(16,10,replace=False),'RM_top10':np.argsort(-ss,kind='stable')[:10],'metric_oracle_diagnostic':np.lexsort((-C,domination))[:10]}
    results={}
    for key,ids in sets.items():
     y=Y[ids].reshape(10,-1).astype(np.float64);dist=((y[:,None]-y[None,:])**2).mean(-1);results[key]=dict(sum_best_CCC=float(C[ids].sum()),sum_best_native_DTW=float(D[ids].sum()),S_MSE=float(dist.sum()/90),selected=ids.tolist(),mean_RM_score=float(ss[ids].mean()))
    p2m=read(OUT/'bank'/w['id']/'P2_scores.json');qualified=(C>=np.array(p2m['C']).max(1).mean())&(D<=np.array(p2m['D']).min(1).mean());accepted=ss>=refmean
    pool_results.append(dict(window=w['id'],group=w['group'],generator=name,selection=results,qualified_count=int(qualified.sum()),qualified_accepted=int((qualified&accepted).sum()),reference_relative_nonnegative_count=int(accepted.sum()),acceptance_scope='diagnostic score >= same-source fixed P2 mean; not probability of correctness',mode_count_status='not established; no independently validated mode taxonomy',scope='native full window, not official full recording FRC/FRD; best-of16 cost'))
   print('AUDIT',arm,w['id'],flush=True)
  write(out/'selection_probe.json',pool_results);result=dict(arm=arm,selected_step=selection['selected_step'],tasks=tasks,selection_records=len(pool_results));write(out/'complete.json',result);allresults[arm]=result
 write(OUT/'RESULTS_RM.json',allresults)
 # No automatic stage B even if individual thresholds pass.
 write(OUT/'FINAL_DECISION.json',dict(stage_B_started=False,decision='inconclusive until listener-only and interaction probes, multi-answer and full-recording consistency review are complete',pair_results_available=True,selection_probe_available=True,remaining=['independent X-only/Y-only probes','balanced 2x2 interaction audit','full-recording/window score consistency','validated multi-answer acceptance interpretation'],no_generator_updates=True))
if __name__=='__main__':main()
