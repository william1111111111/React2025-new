from .common import *
def main():
 rows=[];lines=['# Fixed-P2 quality-guarded NFT','','TRAIN monitor is training control, not held-out validation. Last proposed and last accepted models are separate rows. P2 reuse is identified, never called a learned improvement.','','| Arm | Model | Proposed | Accepted | Rejected | FRC80 | exact FRD20 | S-MSE | FRVar |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for a in ARMS:
  c=read(OUT/'training'/a/'completion.json')
  for v in ['last-proposal','last-accepted']:
   label=a+'_'+v;m=read(OUT/'task_multitarget'/(label+'.json'))['result']['metrics'];f=read(sorted((OUT/'frd20'/label).glob('status_*.json'))[-1]);assert f['completed_pairs']==2000
   row=dict(arm=a,variant=v,**c,metrics=m,exact_FRD20=f['FRD']);rows.append(row)
   lines.append(f"| {a} | {v} | {c['proposed_steps']} | {c['accepted_steps']} | {c['rejected_steps']} | {m['FRC']:.9f} | {f['FRD']:.9f} | {m['smse']:.9f} | {m['FRVar']:.9f} |")
 write(OUT/'FINAL_RESULTS.json',rows);(OUT/'FINAL_RESULTS.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
