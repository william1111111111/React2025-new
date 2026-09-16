from .common import *
def main():
    lines=['# Full velocity NFT joint evaluation','','P2 reference: FRC80 0.937901101, exact FRD20 130.688674740, S-MSE80 0.067720607. Native source-only K10; no GT output selection.','','| Arm / step | FRC80 | S-MSE80 | FRVar80 | exact FRD20 |','|---|---:|---:|---:|---:|']
    rows=[]
    for a in ARMS:
        for step in [1000,4000]:
            p=OUT/'task_multitarget'/(a+f'_step{step}.json');payload=read(p);m=payload['result']['metrics'];f=None
            if step==4000:
                f=read(sorted((OUT/'frd20'/(a+'_step4000')).glob('status_*.json'))[-1]);assert f['completed_pairs']==2000 and f['completed']
            row=dict(arm=a,step=step,metrics=m,FRD20=None if f is None else f['FRD'],checkpoint_sha256=payload['eval_identity']['student_checkpoint_sha256']);rows.append(row)
            lines.append(f"| {a} / {step} | {m['FRC']:.9f} | {m['smse']:.9f} | {m['FRVar']:.9f} | {row['FRD20']} |")
    write(OUT/'JOINT_RESULTS.json',rows);(OUT/'JOINT_RESULTS.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
