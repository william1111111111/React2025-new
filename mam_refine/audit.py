"""Read-only existing predictions/CCC/FRD audit. No generation or DTW calls."""
import json,csv,hashlib,time
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from hirp.phase15_audit import canonical_hash,sha256_file
ROOT=Path('runs/mam_target/refine_v1');OLD=Path('runs/mam_target/task_v1');PH24=Path('runs/phase24/evaluation_v1')
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def load_verified(p):
 x=json.loads(p.read_text());assert canonical_hash(x['result'])==x['result_sha256'];assert canonical_hash(x['eval_identity'])==x['eval_identity_sha256'];return x
def csvwrite(p,rows):
 with p.open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def group_stats(p):
 p=p.astype(np.float64);k=p.shape[0];flat=p.reshape(k,-1);center=p-p.mean(1,keepdims=True);cf=center.reshape(k,-1)
 distances=((flat[:,None]-flat[None])**2).mean(-1);center_dist=((cf[:,None]-cf[None])**2).mean(-1)
 return dict(candidate_mse_offdiag=float(distances.sum()/(k*(k-1))),centered_candidate_mse_offdiag=float(center_dist.sum()/(k*(k-1))),temporal_variance=float(p.var(1,ddof=1).mean()),speed_abs=float(np.abs(np.diff(p,axis=1)).mean()),near_duplicate_pairs_mse_lt_1e_8=int((distances[np.triu_indices(k,1)]<1e-8).sum()),minimum=float(p.min()),maximum=float(p.max()))
def main():
 protocol=json.loads((OLD/'frd20/protocol.json').read_text());indices=protocol['source_indices'];manifest_path=PH24/'multitarget_development_manifest.json';manifest=json.loads(manifest_path.read_text());summaries=[];per_input=[];groups=[];links=[]
 cases=[('MAM','archive',PH24/'task_multitarget/MAM_archive_offline.json',Path('runs/phase25/timescale_v1/frd20/MAM_archive_offline'))]+[(arm,step,OLD/'task_multitarget'/f'seed123_{arm}_step{step}.json',OLD/'frd20'/f'seed123_{arm}_step{step}') for arm in ('R1','R2') for step in (2000,6000)]
 for arm,step,task,frdroot in cases:
  payload=load_verified(task);res=payload['result'];ident=payload['eval_identity'];tasksha=sha256_file(task)
  assert (ident.get('target_manifest_sha256') or ident.get('manifest_sha256'))==sha256_file(manifest_path)
  m=res['multi_target'];source_frc={r['input_index']:float(np.array(r['CCC_candidate_target']).max(1).sum()) for r in res['per_input']}
  assert abs(np.mean(list(source_frc.values()))-m['FRC'])<1e-6
  values=[]
  for i in indices:
   matrix=np.full((10,10),np.nan)
   for k in range(10):
    for j in range(10):
     x=load_verified(frdroot/f'{i:03d}_{k:02d}_{j:02d}.json');a=x['eval_identity'];assert a['task_sha256']==tasksha and (a['source'],a['candidate'],a['target'])==(i,k,j)
     matrix[k,j]=x['result']['distance']
   values.append(float(matrix.min(1).sum()))
  status=json.loads(sorted(frdroot.glob('status_*.json'))[-1].read_text());assert status['completed'] and status['completed_pairs']==2000 and abs(status['FRD']-np.mean(values))<1e-9
  for i,(raw,export) in enumerate(zip(res['per_input'],res['exports'])):
   assert raw['input_index']==i and sha256_file(export['path'])==export['sha256']
   p=np.load(export['path']);assert p.shape==(10,manifest['sources'][i]['length'],25)
   c=np.asarray(raw['CCC_candidate_target']);assigned=c.argmax(1);rr,cc=linear_sum_assignment(-c)
   refs=manifest['sources'][i]['targets'];unique=len(set(r['sha256'] for r in refs));slots=assigned.tolist()
   per_input.append(dict(arm=arm,step=step,input_index=i,FRC=float(c.max(1).sum()),target_side_mean=float(c.max(0).mean()),one_to_one_mean=float(c[rr,cc].mean()),selected_slots=json.dumps(slots),selected_distinct_content=len(set(refs[s]['sha256'] for s in slots)),unique_reference_contents=unique,duplicate_reference_slots=10-unique,selected_distinct_slots=len(set(slots))))
   for group,a,b in [('AU',0,15),('VA',15,17),('expression',17,25)]:groups.append(dict(arm=arm,step=step,input_index=i,group=group,**group_stats(p[...,a:b])))
  selected=[r for r in per_input if r['arm']==arm and r['step']==step]
  summaries.append(dict(arm=arm,step=step,FRC80=m['FRC'],FRC20=float(np.mean([source_frc[i] for i in indices])),FRD20=status['FRD'],FRD_pairs=2000,S_MSE80=m['smse'],FRVar80=m['FRVar'],target_side_mean=float(np.mean([v['target_side_mean'] for v in selected])),one_to_one_mean=float(np.mean([v['one_to_one_mean'] for v in selected])),mean_selected_distinct_reference_contents=float(np.mean([v['selected_distinct_content'] for v in selected]))))
  links.append(dict(arm=arm,step=step,task_path=str(task),task_sha256=tasksha,checkpoint_sha256=ident.get('checkpoint_sha256') or ident['model']['sha256'],eval_identity=ident,FRD_status=status,source_indices=indices,output_policy=res.get('output_policy','native archive AU rounding'),verified_export_files=80))
  print('VERIFIED',arm,step,summaries[-1],flush=True)
 csvwrite(ROOT/'baseline_per_input.csv',per_input);csvwrite(ROOT/'baseline_channel_per_input.csv',groups);csvwrite(ROOT/'baseline_summary.csv',summaries)
 grouped=[]
 for arm,step,*_ in cases:
  for g in ('AU','VA','expression'):
   ss=[r for r in groups if r['arm']==arm and r['step']==step and r['group']==g]
   grouped.append(dict(arm=arm,step=step,group=g,**{k:float(np.mean([v[k] for v in ss])) for k in ss[0] if k not in ('arm','step','group','input_index')}))
 csvwrite(ROOT/'baseline_channel_summary.csv',grouped)
 write(ROOT/'RESULTS_LATEST.json',dict(models=summaries,provenance=links,channel_summary=grouped,verified_pairs=10000,method='read existing cached exact DTW distances only; no new generation or DTW',population='FRC80 Development80; FRC20/FRD20 same frozen20 subset',time=time.time()))
 lines=['# Latest verified baseline results','', '| Model | step | FRC80 ↑ | FRC20 ↑ | exact FRD20 ↓ | S-MSE80 | FRVar80 |','|---|---:|---:|---:|---:|---:|---:|']
 for r in summaries:lines.append(f"| {r['arm']} | {r['step']} | {r['FRC80']:.12f} | {r['FRC20']:.12f} | {r['FRD20']:.12f} | {r['S_MSE80']:.12f} | {r['FRVar80']:.12f} |")
 lines+=['','All exact FRD scores verified from2000/2000 cached pairs each. Identity hashes link FRD to the SAME task exports and checkpoint. No recomputation. R2 step6000 is the fixed common refinement parent. Native AU policies differ from archived MAM; all data are development, not independent confirmation.','', 'Detailed channel/coverage diagnostics are in baseline_channel_summary.csv and baseline_per_input.csv. A matched target SLOT is not a semantic mode; duplicate target content slots are explicitly counted.']
 (ROOT/'RESULTS_LATEST.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
