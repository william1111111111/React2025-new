"""Read-only final endpoint analysis and one fixed AU policy diagnostic."""
import json,csv,sys
from pathlib import Path
import numpy as np
import torch
from .dynamics import GROUPS,decomposition
from hirp.phase15_audit import sha256_file,canonical_hash
OUT=Path('runs/reaction_flow/t0_next_v1')
PATHS={'MAM':'runs/phase24/evaluation_v1/task_multitarget/MAM_archive_offline.json','R2':'runs/mam_target/task_v1/task_multitarget/seed123_R2_step6000.json','A12000':'runs/reaction_flow/trajectory_v1/task_multitarget/seed123_A_step12000.json','T0':'runs/reaction_flow/task_dynamics_v1/task_multitarget/seed123_T0-task_step14000.json','T1':'runs/reaction_flow/task_dynamics_v1/task_multitarget/seed123_T1-task-dynamics_step14000.json'}
def csvout(name,rows):
 with (OUT/name).open('x') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
def load(item):
 assert sha256_file(item['path'])==item['sha256'];return torch.from_numpy(np.load(item['path']))
def main():
 torch.set_num_threads(1);sys.path.insert(0,'/home/zhengshiyi/react2025')
 from framework.metrics.FRC import compute_FRC
 from framework.metrics.S_MSE import compute_s_mse
 from framework.metrics.FRVar import compute_FRVar
 targetpath=Path('runs/phase24/evaluation_v1/processed_targets/completed.json');ix=json.loads(targetpath.read_text());assert canonical_hash(ix['result'])==ix['result_sha256'];targets=[load(x) for x in ix['result']['files']]
 rows=[];matches=[];candidates=[];policy=[];native=[];refs={}
 for name,path in PATHS.items():
  payload=json.loads(Path(path).read_text());assert canonical_hash(payload['result'])==payload['result_sha256'];x=payload['result'];exports=x['exports'];assert len(exports)==80
  refs[name]=dict(result=path,result_sha256=sha256_file(path),checkpoint_sha256=payload['eval_identity'].get('checkpoint_sha256'),exports=exports)
  preds=[]
  for i,item in enumerate(exports):
   y=load(item);assert y.shape==targets[i].shape and y.shape[0]==10;preds.append(y)
   for pop,v in [('prediction',y),('processed_targets',targets[i])]:
    if pop=='processed_targets' and name!='MAM':continue
    for g,a,b in GROUPS:
     q=v[...,a:b].double();total,parts=decomposition(q);err=float(abs(total-sum(parts.values())));assert err<1e-12
     row=dict(model=name if pop=='prediction' else 'processed_targets',source=i,group=g,dimension_weight=(b-a)/25,total=float(total),**{k:float(z) for k,z in parts.items()},identity_error=err,speed=float(q.diff(dim=1).abs().mean()),acceleration=float(q.diff(dim=1).diff(dim=1).abs().mean()))
     row.update({k+'_contribution':row[k]*(b-a)/25 for k in ('total','DC','slow','fast')});rows.append(row)
   mat=np.asarray(x['per_input'][i]['CCC_candidate_target']);assert mat.shape==(10,10)
   matches.append(dict(model=name,source=i,target_side_mean=float(mat.max(0).mean()),candidate_best_mean=float(mat.max(1).mean())))
   for k,v in enumerate(mat.max(1)):candidates.append(dict(model=name,source=i,candidate=k,best_CCC=float(v)))
  native.append(dict(model=name,**{k:x['multi_target'][k] for k in ('FRC','smse','FRVar')}))
  rounded=[]
  for y in preds:
   q=y.clone();q[...,:15]=(q[...,:15]>=.5).to(q.dtype);assert torch.equal(q[...,15:],y[...,15:]);rounded.append(q)
  policy.append(dict(model=name,policy='AU>=0.5; VA/expression unchanged',FRC=float(compute_FRC(rounded,targets,p=4)),S_MSE=float(compute_s_mse(rounded)),FRVar=float(compute_FRVar(rounded)),exact_FRD20='not evaluated for this diagnostic policy'))
  print(name,policy[-1],flush=True)
 aggregate=[]
 for name in PATHS.keys()|{'processed_targets'}:
  for group in [g for g,_,_ in GROUPS]+['overall']:
   rs=[r for r in rows if r['model']==name and (group=='overall' or r['group']==group)]
   vals={}
   for key in ('total','DC','slow','fast','speed','acceleration'):
    vals[key]=sum(r[key]*(r['dimension_weight'] if group=='overall' else 1) for r in rs)/80
   aggregate.append(dict(model=name,group=group,**vals,**{k+'_contribution':vals[k]*(1 if group=='overall' else rs[0]['dimension_weight']) for k in ('total','DC','slow','fast')}))
 csvout('decomposition_per_source.csv',rows);csvout('decomposition_summary.csv',aggregate);csvout('target_matching.csv',matches);csvout('candidate_best_CCC.csv',candidates);csvout('AU_policy_diagnostic.csv',policy);csvout('native_scores.csv',native)
 match_summary=[]
 for name in PATHS:
  vals=[r['best_CCC'] for r in candidates if r['model']==name];m=[r['target_side_mean'] for r in matches if r['model']==name]
  match_summary.append(dict(model=name,target_side_mean=float(np.mean(m)),candidate_best_mean=float(np.mean(vals)),**{f'quantile_{q}':float(np.quantile(vals,q)) for q in (0,.1,.25,.5,.75,.9,1)}))
 csvout('matching_summary.csv',match_summary)
 rawpath=Path('runs/reaction_flow/task_dynamics_v1/raw_TRAIN_activity.json');raw=json.loads(rawpath.read_text());csvout('raw_TRAIN_activity_summary.csv',[dict(group=g,speed=float(np.mean([r['speed'] for r in raw['rows'] if r['group']==g])),acceleration=float(np.mean([r['acceleration'] for r in raw['rows'] if r['group']==g])),population='TRAIN16 actual paired raw crops; no Processor') for g,_,_ in GROUPS])
 with (OUT/'references.json').open('x') as f:json.dump(dict(models=refs,processed_targets=dict(path=str(targetpath),sha256=sha256_file(targetpath)),raw_TRAIN=dict(path=str(rawpath),sha256=sha256_file(rawpath)),metric_sources={str(p):sha256_file(p) for p in [Path('/home/zhengshiyi/react2025/framework/metrics')/n for n in ('FRC.py','S_MSE.py','FRVar.py')]},max_identity_error=max(r['identity_error'] for r in rows)),f,indent=2)
if __name__=='__main__':main()
