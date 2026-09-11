"""Actual measurements only, one checkpoint and one native output policy per row."""
import json,csv
from pathlib import Path
import numpy as np
from .config import ROOT
from hirp.phase15_audit import sha256_file

def main():
    out=ROOT/'analysis';out.mkdir(exist_ok=True);rows=[]
    for path in sorted((ROOT/'task_multitarget').glob('*.json')):
        payload=json.loads(path.read_text());r=payload['result'];m=r['multi_target'];frd=None;count=0
        folder=ROOT/'frd20'/path.stem
        if folder.exists():count=len(list(folder.glob('[0-9]*.json')))
        for p in sorted(folder.glob('status_*.json')):
            s=json.loads(p.read_text())
            if s['completed'] and s['completed_pairs']==2000:frd=s['FRD']
        row=dict(model=r['arm'],flow_updates=r['steps'],warmstart_R2_updates=6000,NFE=16,FRC80=m['FRC'],FRC20=m['FRC20_diagnostic'],exact_FRD20=frd,FRD_pairs=count,S_MSE80=m['smse'],FRVar80=m['FRVar'],single_paired_FRC=r['single_paired_FRC'],target_side_CCC=float(np.mean([x['target_side_mean'] for x in r['per_input']])),output_policy=r['output_policy'],checkpoint_sha256=payload['eval_identity']['checkpoint_sha256'],task_sha256=sha256_file(path))
        for group in ('AU','VA','expression'):
            for k in r['exports'][0]['group_diagnostics'][group]:row[group+'_'+k]=float(np.mean([x['group_diagnostics'][group][k] for x in r['exports']]))
        rows.append(row)
    (out/'results.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    if rows:
        with (out/'results.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Direct conditional trajectory flow — actual status','', 'Development80, native continuous AU, K10, Euler16. FRD20 is the fixed20-source subset, not a full-population score.','', '| Model | Flow updates | NFE | FRC80 | FRC20 | exactFRD20 | S-MSE80 | FRVar80 |','|---|---:|---:|---:|---:|---:|---:|---:|', '| MAM native archive | different budget | n/a | .810962319 | .766297889 | 172.576423473 | .157203704 | .058911592 |','| R2 parent6000 | n/a | n/a | 1.181468685 | 1.134674340 | 133.409074259 | .043679938 | .039064668 |']
    for r in rows:
        v=[r['model'],str(r['flow_updates']),str(r['NFE'])]+['pending' if r[k] is None else f'{r[k]:.9f}' for k in ('FRC80','FRC20','exact_FRD20','S_MSE80','FRVar80')];lines.append('| '+' | '.join(v)+' |')
    lines+=['','Every flow row additionally inherits6000 R2 training updates through frozen stems/encoder, not its response head. No MAM weights or predictions are used in training.','', 'No source-only task result is claimed from coordinate roundtrip, FM training loss, or noisy-target reconstruction. No new solver/noise-bank/seed/threshold search.','', 'Phase B is prepared but starts only after A12000 sampling diagnostics are inspected; if A produces only noise, inspect the path/masks/integration before task adaptation.']
    (out/'RESULTS_LATEST.md').write_text('\n'.join(lines)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4))
    for phase in ('A','F-cont','F-task'):
        logs=sorted((ROOT/phase).glob('attempt_*/training.jsonl'))
        if logs:
            raw=[]
            for line in logs[-1].read_text().splitlines():
                try:raw.append(json.loads(line))
                except json.JSONDecodeError:pass
            if raw:ax.plot([x['global_step'] for x in raw],[x['FM'] for x in raw],label=phase,alpha=.7)
    ax.set(xlabel='Flow optimizer updates (excludes R2 warmstart)',ylabel='TRAIN valid-coordinate FM MSE');ax.legend();fig.tight_layout();fig.savefig(out/'FM_curve.png',dpi=130);plt.close(fig)
    for path in sorted((ROOT/'task_multitarget').glob('*.json')):
        if not any(x in path.name for x in ('step12000','step14000')):continue
        r=json.loads(path.read_text())['result']
        for index in (0,1,2):
            dest=out/f'{path.stem}_source{index:03d}_all10.png'
            if dest.exists():continue
            y=np.load(r['exports'][index]['path']);n=y.shape[1];smooth=np.stack([y[:,j:min(j+8,n)].mean(1) for j in range(0,n,8)],1)
            fig,axes=plt.subplots(25,2,figsize=(13,36))
            for c in range(25):
                for k in range(10):axes[c,0].plot(y[k,:,c],lw=.4,alpha=.6);axes[c,1].plot(np.arange(0,n,8),smooth[k,:,c],lw=.4,alpha=.6)
                for a in axes[c]:a.set_ylim((-1,1) if 15<=c<17 else (0,1));a.set_ylabel(str(c))
            fig.suptitle(path.stem+f' fixed source{index}, all10, raw / 8-frame means');fig.tight_layout();fig.savefig(dest,dpi=90);plt.close(fig)
if __name__=='__main__':main()
