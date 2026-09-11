"""Reproducible single-seed tables, paired session intervals and complete curves."""
import csv,json,os,sys,time
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/hirp25_final_matplotlib')
sys.path.insert(0,'/home/zhengshiyi/react2025_new')
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hirp.phase15_audit import sha256_file,canonical_hash
r=Path('runs/phase25/timescale_v1');final=r/'final_seed123_v1';out=final/'analysis';out.mkdir(exist_ok=False)
def write(name,rows):
 with (out/name).open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def save(name,value):
 with (out/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def interval(x):
 x=np.asarray(x,dtype=float);rng=np.random.default_rng(25003);means=x[rng.integers(0,len(x),(10000,len(x)))].mean(1)
 return float(x.mean()),*np.quantile(means,[.025,.975]).tolist()
# Full task curve, preserving each immutable result's own implementation identity.
files={}
for directory in (r/'task_multitarget',final/'task_multitarget'):
 for p in directory.glob('seed123_*.json'):
  if p.name in files:raise ValueError('duplicate task identity')
  files[p.name]=p
assert len(files)==18
manifest=json.load(open('runs/phase24/evaluation_v1/multitarget_development_manifest.json'))
plan=json.load(open('runs/phase22/replicated_v1/evaluation_plan/manifest.json'))
assert [s['clip_id'] for s in manifest['sources']]==plan['clip_ids']
sessions=[s['clip_id'].split('/')[1] for s in manifest['sources']];session_names=sorted(set(sessions))
task=[];per_input=[];task_raw={};numerical=[];resource=[];input_hashes=[]
for p in files.values():
 payload=json.load(open(p));assert canonical_hash(payload['result'])==payload['result_sha256'];v=payload['result'];arm=v['arm'];step=v['steps'];task_raw[arm,step]=v
 input_hashes.append(dict(path=str(p),sha256=sha256_file(p)))
 for i,row in enumerate(v['per_input']):
  m=np.array(row['CCC_candidate_target']);a,b=linear_sum_assignment(-m)
  per_input.append(dict(arm=arm,step=step,index=i,session=sessions[i],FRC=float(m.max(1).sum()),single_FRC=float(m[:,0].sum()),generated_side_mean=float(m.max(1).mean()),target_side_mean=float(m.max(0).mean()),one_to_one_mean=float(m[a,b].mean())))
 rr=per_input[-80:]
 task.append(dict(arm=arm,step=step,seed=123,FRC=v['multi_target']['FRC'],single_FRC=v['single_paired_FRC'],target_side_mean=np.mean([z['target_side_mean'] for z in rr]),one_to_one_mean=np.mean([z['one_to_one_mean'] for z in rr]),FRVar=v['multi_target']['FRVar'],S_MSE=v['multi_target']['smse'],temporal_S_MSE=v['multi_target']['temporal_smse'],TLCC_first_candidate=v['multi_target']['TLCC']))
 resource.append(dict(arm=arm,step=step,evaluation_seconds=v['seconds'],peak_bytes=v['peak_memory_bytes']))
 for i,e in enumerate(v['exports']):
  c=e['checks']
  numerical.append(dict(arm=arm,step=step,index=i,max_chunk_error=c['candidate_consistency_max_error'],old_strict_pass=c.get('original_strict_check_passed',True),fp64_chunk_max=c.get('fp64_chunk_max'),fp32_to_fp64_max=c.get('fp32_to_fp64_max'),multi_FRC_chunk_error=c.get('multi_FRC_chunk_abs_error'),single_FRC_chunk_error=c.get('single_FRC_chunk_abs_error'),boundary_abs_change=c['boundary_abs_change'],interior_abs_change=c['interior_abs_change']))
write('task_curve.csv',sorted(task,key=lambda x:(x['arm'],x['step'])));write('task_per_input.csv',per_input);write('task_resources.csv',resource);write('numerical_and_boundaries.csv',numerical)
# ES curves, bank0 explicitly separate from two-bank endpoint averages.
es=[];results={};channel=[]
for p in sorted((final/'distribution').glob('*_K32.json')):
 payload=json.load(open(p));assert canonical_hash(payload['result'])==payload['result_sha256'];v=payload['result'];g=v['aggregate'];key=(v['T'],v['arm'],v['step'],v['bank']);results[key]=v
 es.append(dict(T=v['T'],arm=v['arm'],step=v['step'],bank=v['bank'],K=32,conditional_ES=g['conditional']['loss'],conditional_cross=g['conditional']['cross_distance'],conditional_self=g['conditional']['self_distance'],group_ES=g['marginal']['loss'],group_cross=g['marginal']['cross'],group_self=g['marginal']['self'],correct_common_ES=g['conditional_correct_common']['loss'],shuffled_common_ES=g['conditional_shuffled']['loss'],shuffle_gap=g['conditional_shuffle_gap']['difference'],within=g['spread']['within_input_descriptor_self'],between=g['spread']['between_input_descriptor_distance'],seconds=v['evaluation_seconds'],peak_bytes=v['peak_allocated_bytes']))
 for group,stats in g['channels'].items():
  channel.append(dict(T=v['T'],arm=v['arm'],step=v['step'],bank=v['bank'],group=group,statistics=json.dumps(stats)))
 input_hashes.append(dict(path=str(p),sha256=sha256_file(p)))
assert len(es)==48
write('ES_all_banks.csv',es);write('channels.csv',channel)
metrics=['conditional_ES','conditional_cross','conditional_self','group_ES','group_cross','group_self','correct_common_ES','shuffled_common_ES','shuffle_gap','within','between']
endpoints=[]
for T in (750,128):
 for arm in ('C0','C1','C2'):
  for step in (2000,6000):
   rr=[x for x in es if x['T']==T and x['arm']==arm and x['step']==step]
   assert len(rr)==2
   endpoints.append(dict(T=T,arm=arm,step=step,K=32,bank_aggregation='mean of existing banks0 and1',**{k:float(np.mean([x[k] for x in rr])) for k in metrics}))
write('ES_endpoints_two_bank_mean.csv',endpoints)
# Session cluster bootstrap of paired differences, conditional on single seed/protocol.
paired=[]
for step in (2000,6000):
 for baseline in ('C0','C1'):
  for metric in ('FRC','single_FRC','target_side_mean','one_to_one_mean'):
   delta=[]
   for s in session_names:
    def mean(arm):return np.mean([x[metric] for x in per_input if x['arm']==arm and x['step']==step and x['session']==s])
    delta.append(mean('C2')-mean(baseline))
   mean,lo,hi=interval(delta);paired.append(dict(protocol='full_task_K10',step=step,comparison='C2-'+baseline,metric=metric,mean=mean,lower=lo,upper=hi,n_sessions=20,training_seeds=1))
  for T in (750,128):
   for metric in ('conditional_ES','group_ES','shuffle_gap'):
    delta=[]
    for s in session_names:
     def value(arm):
      vals=[]
      for bank in (0,1):
       result=results[T,arm,step,bank];row=next(x for x in result['per_session'] if x['session_id']==s)
       if metric=='conditional_ES':vals.append(row['conditional']['loss'])
       elif metric=='group_ES':vals.append(row['marginal']['loss'])
       else:
        for shuffle in result['extra_shuffles']:
         sr=next(x for x in shuffle['per_session'] if x['session_id']==s);vals.append(sr['gap'])
      return np.mean(vals)
     delta.append(value('C2')-value(baseline))
    mean,lo,hi=interval(delta);paired.append(dict(protocol=f'T{T}_K32_two_banks'+('_three_derangements' if metric=='shuffle_gap' else ''),step=step,comparison='C2-'+baseline,metric=metric,mean=mean,lower=lo,upper=hi,n_sessions=20,training_seeds=1))
write('paired_session_intervals.csv',paired)
# Full actual optimizer accounting, all gradient observations, all saturation observations.
training=[];grad=[];sat=[];checkpoints=[]
for arm in ('C0','C1','C2'):
 attempts=sorted((r/'seed_123'/arm).glob('attempt_*'));last=json.load(open(attempts[-1]/'summary.json'));rows=[json.loads(s) for s in (attempts[-1]/'training.jsonl').read_text().splitlines()]
 assert last['completed'] and last['actual_optimizer_steps']==last['training_rows']==len(rows)==6000
 training.append(dict(arm=arm,requested=6000,actual=6000,rows=6000,source_occurrences=last['source_occurrences'],source_frames=last['valid_source_frame_exposures'],pair_frames=last['valid_pair_frame_exposures'],tensor_frames=last['tensor_frame_exposures'],unique_sources=last['unique_sources'],reference_exposures=last['reference_exposures'],parameters=last['parameter_count'],cumulative_wall_seconds=last['cumulative_wall_seconds'],peak_bytes=max(json.load(open(a/'summary.json'))['peak_allocated_bytes'] for a in attempts),initialization_hash=last['initialization_hash'],consumed_hashes=json.dumps(last['consumed_hashes'],sort_keys=True),prior_unchanged=last['prior_unchanged']))
 for row in rows:
  if row['gradients']:
   g=row['gradients'];grad.append(dict(arm=arm,step=row['step'],conditional_norm=g['conditional']['total'],group_norm=g['group']['total'] if g['group'] else None,weighted_group_norm=g['weighted_group_norm'],ratio=g['weighted_group_over_conditional'],cosine=g['cosine'],finite=g['finite']))
 for a in attempts:
  for p in a.glob('diagnostic_*.json'):
   d=json.load(open(p));step=int(p.stem.split('_')[-1]);sat.append(dict(arm=arm,step=step,residual_saturation=d['residual']['tanh_saturation_fraction'],noise_VJP=json.dumps(d['noise_jacobian']),outputs=json.dumps(d['outputs']),path_gradients=json.dumps({k:{g:z['conditional_gradient_norm'] for g,z in v.items()} for k,v in d['paths'].items()})))
  for c in json.load(open(a/'summary.json'))['checkpoints']:
   assert sha256_file(c['path'])==c['sha256'];checkpoints.append(dict(arm=arm,**c))
assert len({x['initialization_hash'] for x in training})==1 and len({x['consumed_hashes'] for x in training})==1
write('training_accounting.csv',training);write('gradient_curve.csv',grad);write('saturation_curve.csv',sat);save('checkpoint_fingerprints.json',checkpoints)
# Fixed full-frame20-source exact FRD, only2000 checkpoints plus native archived MAM.
frd=[]
for label in ('seed123_C0_step2000','seed123_C1_step2000','seed123_C2_step2000','MAM_archive_offline'):
 p=sorted((r/'frd20'/label).glob('status_*.json'))[-1];v=json.load(open(p));assert v['completed'] and v['completed_pairs']==2000
 frd.append(dict(model=label,FRD=v['FRD'],pairs=v['completed_pairs'],sources=20,training_step=2000 if label.startswith('seed') else 'native archive'))
write('FRD20.csv',frd)
# Figures; no mixing bank0 curve and two-bank averages.
fig,axs=plt.subplots(1,3,figsize=(14,4))
for arm in ('C0','C1','C2'):
 rr=sorted([x for x in task if x['arm']==arm],key=lambda x:x['step']);axs[0].plot([x['step'] for x in rr],[x['FRC'] for x in rr],'o-',label=arm)
 rr=sorted([x for x in es if x['arm']==arm and x['T']==750 and x['bank']==0],key=lambda x:x['step'])
 for ax,metric in zip(axs[1:],('conditional_ES','group_ES')):ax.plot([x['step'] for x in rr],[x[metric] for x in rr],'o-',label=arm)
for ax,title in zip(axs,('Full-sequence K10 FRC','T750 conditional ES / K32 bank0','T750 group ES / K32 bank0')):ax.set_title(title);ax.set_xlabel('Actual optimizer steps');ax.legend()
fig.suptitle('seed123 / Development80 / frozen checkpoints');fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=170)
fig,axs=plt.subplots(1,2,figsize=(10,4))
for T,ax in zip((750,128),axs):
 for arm in ('C0','C1','C2'):
  rr=sorted([x for x in endpoints if x['arm']==arm and x['T']==T],key=lambda x:x['step']);ax.plot([x['conditional_ES'] for x in rr],[x['group_ES'] for x in rr],'o-',label=arm)
  for x in rr:ax.annotate(str(x['step']),(x['conditional_ES'],x['group_ES']))
 ax.set(title=f'T{T} / K32 / mean banks0+1',xlabel='Conditional ES',ylabel='Group ES');ax.legend()
fig.tight_layout();fig.savefig(out/'endpoint_tradeoff.png',dpi=170)
save('input_result_fingerprints.json',input_hashes)
save('summary.json',dict(task=task,endpoints=endpoints,paired=paired,training=training,frd=frd,numerical=dict(strict_exceptions=sum(not x['old_strict_pass'] for x in numerical),max_fp64_chunk=max([x['fp64_chunk_max'] or 0 for x in numerical]),max_FRC_chunk_error=max([x['multi_FRC_chunk_error'] or 0 for x in numerical])),scope='single training seed; intervals conditional on model/fixed banks/development sessions, not participant-independent or training-seed uncertainty'))
print('ANALYSIS COMPLETE',out)
