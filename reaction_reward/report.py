from .common import *
from .train import ARMS

def main():
 results=read(OUT/'RESULTS_RM.json');y=read(OUT/'probes/Y_only.json')['audit'];retention=read(OUT/'MULTIANSWER_RETENTION.json');lines=['# Stage A: conditional matching reward models','','All evidence is automatic/data-supported, not human preference. No generator update or RL.','', '| Arm | Selected step | Context accuracy | Active time accuracy | Unseen-generator accuracy | Weak retention |', '|---|---:|---:|---:|---:|---:|'];decisions={}
 for arm,r in results.items():
  t=r['tasks'];context=t['base_context'];temporal=t['temporal'];unseen=t['unseen_generator'];rr=retention[arm];ratio=rr['weak_accepted']/max(1,rr['weak_total']);base=retention['RM-Paired'];base_ratio=base['weak_accepted']/max(1,base['weak_total'])
  conditions=dict(temporal_accuracy=temporal.get('accuracy',0)>=.7,unseen_accuracy=unseen.get('accuracy',0)>=.65,conditional_gain=context.get('accuracy',0)-y.get('accuracy',0)>=.1,bootstrap=all(t[k].get('group_bootstrap_95',[0])[0]>.5 for k in ['base_context','temporal','unseen_generator']),weak_retention=ratio>=base_ratio)
  # Few independent groups and unvalidated accepted-mode semantics remain material.
  decisions[arm]=dict(thresholds=conditions,numerical_screen_pass=all(conditions.values()),verdict='inconclusive: assess small independent-group coverage and fixed-pool value; no automatic RL')
  vals=[t[k].get('accuracy') for k in ['base_context','temporal','unseen_generator']];lines.append(f"| {arm} | {r['selected_step']} | {vals[0]} | {vals[1]} | {vals[2]} | {ratio:.4f} |")
 lines+=['','Scope: 416 fixed TRAIN windows, recording-date grouped split. Exact candidate-window DTW is not full-recording FRD. Each fixed-pool selection consumes 16 generated candidates.','Independent listener-only and balanced interaction diagnostics are under probes/. Full-recording aggregation and weak-answer retention are under audit/.','Mode acceptance remains a score-relative diagnostic: no independent human or mode truth was created.']
 (OUT/'RESULTS_RM.md').write_text('\n'.join(lines)+'\n');write(OUT/'FINAL_DECISION.json',dict(stage='A',stage_B_started=False,generator_updated=False,arms=decisions,remaining='scientific review of fixed-pool utility and multi-answer rejection; numerical thresholds alone do not authorize RL'))
if __name__=='__main__':main()
