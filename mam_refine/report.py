import json,csv
from pathlib import Path
import numpy as np
from mam_refine.audit import ROOT,group_stats,csvwrite,load_verified,write

def main():
 base=json.loads((ROOT/'RESULTS_LATEST.json').read_text())['models'];new=[];group_rows=[]
 for arm in ('R2-cont','R3-cover'):
  for step in (6500,8000):
   p=ROOT/'task_multitarget'/f'seed123_{arm}_step{step}.json'
   if not p.exists():continue
   x=load_verified(p);v=x['result'];m=v['multi_target'];frd=None;pairs=0
   if step==8000:
    status=sorted((ROOT/'frd20'/f'seed123_{arm}_step{step}').glob('status_*.json'))
    if status:
     s=json.loads(status[-1].read_text());pairs=s['completed_pairs'];frd=s['FRD'] if s['completed'] else None
   row=dict(arm=arm,step=step,additional_steps=step-6000,FRC80=m['FRC'],FRC20=m['FRC20_diagnostic'],FRD20=frd,FRD_pairs=pairs,S_MSE80=m['smse'],FRVar80=m['FRVar'],target_side_mean=float(np.mean([r['target_side_mean'] for r in v['per_input']])),one_to_one_mean=float(np.mean([r['one_to_one_mean'] for r in v['per_input']])),mean_selected_distinct_reference_contents=float(np.mean([r['selected_target_contents'] for r in v['per_input']])),checkpoint_sha256=x['eval_identity']['checkpoint_sha256'],output_policy=v['output_policy'])
   for i,e in enumerate(v['exports']):
    p=np.load(e['path'])
    for group,a,b in [('AU',0,15),('VA',15,17),('expression',17,25)]:group_rows.append(dict(arm=arm,step=step,input_index=i,group=group,**group_stats(p[...,a:b])))
   for g in ('AU','VA','expression'):
    selected=[x for x in group_rows if x['arm']==arm and x['step']==step and x['group']==g]
    for name in ('candidate_mse_offdiag','centered_candidate_mse_offdiag','temporal_variance','speed_abs'):row[g+'_'+name]=float(np.mean([x[name] for x in selected]))
   new.append(row)
 if new:csvwrite(ROOT/'refinement_results.csv',new);csvwrite(ROOT/'refinement_channel_per_input.csv',group_rows)
 lines=['# R2-cont / R3-cover fixed-budget results','', 'Both inherit identical R2 step6000 model/optimizer and consume same new6001..8000 schedule, differing ONLY in coverage. Native continuous AU, K10; Development80, exact FRD only fixed20-source subset.','', '| Model | global step | FRC80 ↑ | FRC20 ↑ | exact FRD20 ↓ | S-MSE80 | FRVar80 |','|---|---:|---:|---:|---:|---:|---:|']
 for r in base+new:lines.append(f"| {r['arm']} | {r['step']} | {r['FRC80']:.9f} | {r['FRC20']:.9f} | {r['FRD20'] if r['FRD20'] is not None else 'not scheduled at interim / pending'} | {r['S_MSE80']:.9f} | {r['FRVar80']:.9f} |")
 final={r['arm']:r for r in new if r['step']==8000 and r['FRD20'] is not None}
 if len(final)==2:
  a=final['R2-cont'];b=final['R3-cover'];quality=b['FRC80']>.810962318916204 and b['FRD20']<172.57642347297096;spread=b['S_MSE80']>a['S_MSE80']
  lines+=['',f'R3 final quality thresholds satisfied: {quality}; S-MSE above equal-budget control: {spread}. These flags alone do not establish valid/high-quality diversity: inspect channel activity/coverage and all baselines. No extra seed or independent confirmation inference.',f'R3 minus R2-cont: FRC80={b["FRC80"]-a["FRC80"]:+.9f}, FRD20={b["FRD20"]-a["FRD20"]:+.9f}, S-MSE80={b["S_MSE80"]-a["S_MSE80"]:+.9f}.','', 'Fixed budget finished. No automatic extension, codec/flow, new loss or seed search.']
 lines+=['', 'Raw baseline, channel, target-slot and content-repeat diagnostics remain separate and immutable. Inherited cost is6000 steps/17,135,366 valid frames per branch; incremental actual steps/frames/seconds/peak memory and optimizer step are in each status.json. Checkpoint/policy identifiers appear in refinement_results.csv. AU native rounding/pretraining differ for archived MAM; this remains a native-system development comparison.']
 (ROOT/'REFINEMENT_REPORT.md').write_text('\n'.join(lines)+'\n')
 write(ROOT/'refinement_results.json',dict(baselines=base,refinements=new,completed_final_points=len(final)))
if __name__=='__main__':main()
