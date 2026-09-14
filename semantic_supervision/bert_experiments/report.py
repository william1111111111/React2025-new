"""Completed main points, fixed-cache strata, paired deltas and separate interventions."""
import argparse,collections
import numpy as np
from .common import *
from .evaluate import label_for

def bootstrap_delta(rows,base,field):
 sessions=sorted({r['session'] for r in rows});diff=np.array([np.mean([r[field]-b[field] for r,b in zip(rows,base) if r['session']==s]) for s in sessions]);rng=np.random.default_rng(123);sample=diff[rng.integers(0,len(diff),(2000,len(diff)))].mean(1)
 return dict(session_equal_mean=float(diff.mean()),session_cluster_bootstrap_95=[float(x) for x in np.quantile(sample,[.025,.975])],sessions=len(sessions),note='fixed-checkpoint DEV uncertainty only; not training-seed robustness')
def main():
 protocol=verify();results=[];curves=[];exposures={};baseline=None
 for arm in ARMS:
  logs=OUT/'training'/arm/'training.jsonl'
  if logs.exists():
   rows=[__import__('json').loads(s) for s in logs.read_text().splitlines()];assert [r['step'] for r in rows]==list(range(1,len(rows)+1));exposures[arm]=rows
  for step in [500,1000]:
   path=OUT/'task_multitarget'/(label_for(arm,step)+'.json')
   if path.exists():curves.append(dict(arm=arm,step=step,metrics=read(path)['result']['metrics']))
  path=OUT/'task_multitarget'/(label_for(arm,1000)+'.json')
  if not path.exists():continue
  obj=read(path);r=obj['result'];statuses=sorted((OUT/'frd20'/label_for(arm,1000)).glob('status_*.json'));status=read(statuses[-1]) if statuses else {};completed=status.get('completed',False)
  if arm==ARMS[0]:baseline=r
  row=dict(arm=arm,metrics=r['metrics'],FRD20=status.get('FRD') if completed else None,FRD_completed=completed,FRD_pairs=status.get('completed_pairs',0),subgroups=r['subgroups'],checkpoint_sha256=obj['eval_identity']['checkpoint_sha256'],parameters=read(OUT/'training'/arm/'initial.json'),inference_seconds=r['inference_seconds'],peak_gpu_bytes=r['peak_gpu_bytes'])
  if baseline is not None:
   row['paired_source_deltas']=[dict(index=x['index'],session=x['session'],**{k:x[k]-b[k] for k in ['FRC','paired_CCC_mean','smse','FRVar','target_best_CCC_mean']}) for x,b in zip(r['per_source'],baseline['per_source'])]
   row['session_bootstrap']={k:bootstrap_delta(r['per_source'],baseline['per_source'],k) for k in ['FRC','smse','FRVar','target_best_CCC_mean']}
  results.append(row)
 matched=True;reference=exposures.get(ARMS[0],[])
 for arm,rows in exposures.items():
  for i in range(min(len(reference),len(rows))):assert reference[i]['exposure']==rows[i]['exposure'],(arm,i)
 stats={}
 for arm,rows in exposures.items():
  sources=[s for r in rows for s in r['exposure']['source_indices']];events=[e['id'] for r in rows for es in r['exposure']['events'] for e in es];stats[arm]=dict(actual_steps=len(rows),source_visits=len(sources),unique_sources=len(set(sources)),event_visits=len(events),unique_events=len(set(events)),source_repeat_visits=len(sources)-len(set(sources)),valid_source_frames=sum(r['valid_source_frames'] for r in rows),timed_event_visits=sum(r['timed_events'] for r in rows),content_only_event_visits=sum(r['content_only_events'] for r in rows),non_null_before=sum(r['non_null_before'] for r in rows),non_null_after=sum(r['non_null_after'] for r in rows),active_semantic_samples=sum(r['active_semantic_samples'] for r in rows),semantic_gradient_updates=sum(any(v is not None and v>0 for v in r['semantic_grads'].values()) for r in rows),train_seconds=sum(r['seconds'] for r in rows))
 perturbations=[]
 for arm in ARMS[2:]:
  for kind in ['text','time']:
   p=OUT/'task_multitarget'/(label_for(arm,1000,kind)+'.json')
   if p.exists():
    r=read(p)['result'];d=r['paired_deltas'];perturbations.append(dict(arm=arm,kind=kind,measurable_sources=len(d),incorrect_minus_correct={k:float(np.mean([x[k] for x in d])) if d else None for k in ['FRC','paired_CCC_mean']},prediction_mean_abs_change=float(np.mean([x['mean_abs_prediction_change'] for x in r['perturbation_changes']])) if d else None,paired_deltas=d))
 complete=len(results)==4 and all(r['FRD_completed'] for r in results) and len(perturbations)==4 and all(stats[a]['actual_steps']==1000 for a in ARMS)
 if complete:
  b=results[0]
  for r in results:r['prespecified_screen_pass']=r['metrics']['FRC']>=b['metrics']['FRC'] and r['FRD20']<=1.02*b['FRD20'] and r['metrics']['smse']>=.95*b['metrics']['smse']
 write(OUT/'RESULTS.json',dict(complete=complete,main=results,learning_curves=curves,exposure=stats,paired_schedules_match=matched,interventions=perturbations,MAM_reference=protocol['MAM_reference']))
 text='# Frozen-BERT semantic controls\n\nStatus: '+('complete' if complete else 'in progress')+'. All main results use correct source-only inputs.\n\n| Arm | FRC80 | exact FRD20 | S-MSE80 | FRVar80 |\n|---|---:|---:|---:|---:|\n'
 for r in results:
  m=r['metrics'];frd=f"{r['FRD20']:.6f}" if r['FRD_completed'] else 'pending';text+=f"| {r['arm']} | {m['FRC']:.6f} | {frd} | {m['smse']:.6f} | {m['FRVar']:.6f} |\n"
 text+='\nRESULTS.json contains source-paired deltas, fixed-cache NULL/non-NULL/timed strata, session-equal summaries and session bootstrap intervals, target-best CCC coverage, DC/slow/fast decomposition, parameter/resource counts and separate fixed interventions. FRD20 is the fixed 20-source subset, all2000 pairs; it is not full80 FRD.\n\nLimits: 48 TRAIN sources repeated over 1000 updates, single seed123; offline event text, no online/causal claim; TRAIN/DEV proposal distributions remain different. P2-P1 changes pretrained resources and representation architecture. Byte768 is not the historical byte256 model. P0 instantiates fusion but does not use semantic content; active parameter counts differ. BERT features are deterministic across K and do not directly supply new random modes. No event type or planner is used. Unknown roles and missing inputs remain NULL, not no-event.\n'
 if complete:
  passing=[r['arm'] for r in results[1:] if r['prespecified_screen_pass']];text+='\nNext decision: '+('Review '+', '.join(passing)+' against the fixed text/time interventions before widening TRAIN coverage; passing tolerances is not a statistical superiority claim.' if passing else 'No semantic arm passes the prespecified joint screen. Inspect content/time exposure, perturbation evidence and overfitting before changing encoder or adding a planner.')+'\n'
 write(OUT/'completion.json',dict(complete=complete));(OUT/'RESULTS.md').write_text(text);print('report complete=',complete,flush=True)
if __name__=='__main__':main()
