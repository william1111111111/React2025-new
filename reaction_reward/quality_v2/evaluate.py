"""Frozen cal-selected checkpoints; reused audit is a DEVELOPMENT diagnostic."""
import numpy as np
import torch
from scipy.stats import spearmanr
from reaction_reward.common import ROOT,OUT as OLD,read,write,sha
from reaction_reward.data import Data
from reaction_reward.audit import summary,temperatures
from .train import OUT,ARMS,evaluate
from .model import Judge

def main():
 torch.set_num_threads(3);data=Data();records=[r for r in read(OUT/'RECORDS.json') if r['split']=='RM_audit'];windows=[w for w in read(OLD/'WINDOWS.json') if w['split']=='RM_audit'];sc=read(OUT/'QUALITY_SCALE.json');allresults={}
 for arm in ARMS:
  root=OUT/'evaluation'/arm;root.mkdir(parents=True,exist_ok=True);sel=read(OUT/'training'/arm/'SELECTION.json');m=Judge().cuda();m.load_state_dict(torch.load(sel['path'],map_location='cpu',weights_only=False)['model']);m.eval();cal=read(OUT/'training'/arm/f"cal_{sel['step']:06d}.json");T=temperatures(cal);write(root/'CALIBRATION.json',dict(temperatures=T,split='RM_cal',selected_checkpoint_sha256=sha(sel['path'])));pairs=evaluate(m,data,records,arm);write(root/'PAIR_SCORES.json',pairs);metrics={f:summary([r for r in pairs if r['family']==f],T[f]) for f in ['base_context','temporal','weak_context','generated']};metrics['unseen_generator']=summary([r for r in pairs if r['family']=='generated' and r['unseen']],T['generated']);pool=[];errors=[];predictions={}
  for w in windows:
   for gen in ['P2','N0','N1']:
    rows=[dict(source=w['source'],start=w['start'],n=w['n'],A=dict(id=w['source'],start=w['start'],generated=str(OLD/'bank'/w['id']/(gen+'.npy')),candidate=i)) for i in range(16)]
    with torch.no_grad():v=torch.cat([m(*data.batch(rows[j:j+4],'A')) for j in range(0,16,4)]).cpu().numpy()
    scores=v[:,2]-v[:,3] if arm==ARMS[1] else v[:,:2].sum(-1);meta=read(OLD/'bank'/w['id']/(gen+'_scores.json'));C=np.array(meta['C']).max(1);D=np.array(meta['D']).min(1);q=v[:,2:]*np.array(sc['std'])+np.array(sc['mean']);c_hat=q[:,0];d_hat=np.expm1(np.clip(q[:,1],-20,20))*np.sqrt(w['n']);predictions[w['id'],gen]=(scores,C,D)
    if arm==ARMS[1]:errors.append(dict(window=w['id'],source=w['source'],group=w['group'],generator=gen,C_MAE=float(np.abs(c_hat-C).mean()),D_MAE=float(np.abs(d_hat-D).mean()),C_rank_correlation=float(spearmanr(c_hat,C).statistic) if np.std(c_hat)>0 and np.std(C)>0 else None,D_rank_correlation=float(spearmanr(d_hat,D).statistic) if np.std(d_hat)>0 and np.std(D)>0 else None,predicted_C=c_hat.tolist(),predicted_D=d_hat.tolist(),target_C=C.tolist(),target_D=D.tolist()))
    old=next(r for r in read(OLD/'audit/RM-Gen/selection_probe.json') if r['window']==w['id'] and r['generator']==gen);Y=np.load(OLD/'bank'/w['id']/(gen+'.npy'));p2=read(OLD/'bank'/w['id']/'P2_scores.json');qualified=(C>=np.array(p2['C']).max(1).mean())&(D<=np.array(p2['D']).min(1).mean());sets={key:old['selection'][key]['selected'] for key in ['first10','random10','metric_oracle_diagnostic']};sets['RM_top10']=np.argsort(-scores,kind='stable')[:10].tolist();vals={}
    for key,ids in sets.items():
     ys=Y[ids].reshape(10,-1).astype(np.float64);smse=float(((ys[:,None]-ys[None,:])**2).mean(-1).sum()/90);vals[key]=dict(selected=ids,sum_best_CCC=float(C[ids].sum()),sum_best_native_DTW=float(D[ids].sum()),S_MSE=smse,qualified_retained=int(qualified[ids].sum()))
    pool.append(dict(window=w['id'],group=w['group'],generator=gen,selection=vals,qualified_count=int(qualified.sum()),scope='reused development best-of16 native-window diagnostic'))
   print('EVAL',arm,w['id'],flush=True)
  write(root/'SELECTION_PROBE.json',pool);write(root/'QUALITY_HEAD_ERRORS.json',errors)
  # Balanced reciprocal 2x2 interaction; no claim that marginals50% proves conditionality.
  contexts=[r for r in records if r['family']=='base_context'];seen=set();interactions=[]
  for r in contexts:
   key=tuple(sorted([r['source'],r['B']['id']]))
   if key in seen:continue
   other=next((z for z in contexts if z['source']==r['B']['id'] and z['B']['id']==r['source']),None)
   if other is None:continue
   seen.add(key)
   with torch.no_grad():
    val=sum(float((m(*data.batch([z],'A'))-m(*data.batch([z],'B')))[0,0]) for z in [r,other])
   interactions.append(dict(sources=key,I=val,weak_label=r['state']))
  write(root/'INTERACTION_2X2.json',interactions)
  gain={}
  for gen in ['P2','N0','N1']:
   rs=[r for r in pool if r['generator']==gen];gain[gen]={}
   for baseline in ['first10','random10']:
    diffs=np.array([[r['selection']['RM_top10'][k]-r['selection'][baseline][k] for k in ['sum_best_CCC','sum_best_native_DTW','S_MSE']] for r in rs]);gain[gen][baseline]=dict(mean_deltas=diffs.mean(0).tolist(),joint_quality_improved=float(((diffs[:,0]>0)&(diffs[:,1]<0)).mean()),date_groups=len({r['group'] for r in rs}))
  allresults[arm]=dict(selected_step=sel['step'],pair_metrics=metrics,selection_gain=gain,quality_head_supervised=arm==ARMS[1],scope='previously examined RM_audit; development only')
 write(OUT/'RESULTS.json',allresults);write(OUT/'FINAL_DECISION.json',dict(RL_started=False,generator_updated=False,automatic_promotion=False,scope='stage A quality prediction A/B; scientific review required',remaining_limitations=['few recording-date groups','fresh independent confirmation unavailable','full temporal listener-only probe not part of this two-arm training run; old marginal50% is not sufficient']))
if __name__=='__main__':main()
