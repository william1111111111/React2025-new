"""Read-only cached prediction analysis, equal Development80 source weights."""
import json,csv
from pathlib import Path
import numpy as np
import torch
from .dynamics import GROUPS,p8,decomposition
from hirp.phase15_audit import sha256_file
from mam_target.losses import ccc25
OUT=Path('runs/reaction_flow/task_dynamics_v1')
def main():
    paths=[Path('runs/phase24/evaluation_v1/task_multitarget/MAM_archive_offline.json'),Path('runs/mam_target/task_v1/task_multitarget/seed123_R2_step6000.json')]+sorted(Path('runs/reaction_flow/trajectory_v1/task_multitarget').glob('*.json'))
    target_index=json.loads(Path('runs/phase24/evaluation_v1/processed_targets/completed.json').read_text())['result']['files']
    rows=[];frcs=[]
    for path in paths:
        result=json.loads(path.read_text())['result'];label=path.stem
        for i,item in enumerate(result['exports']):
            assert sha256_file(item['path'])==item['sha256']
            y=torch.from_numpy(np.load(item['path'])).double();target=target_index[i]
            assert sha256_file(target['path'])==target['sha256']
            t=torch.from_numpy(np.load(target['path'])).double()
            for kind,v in [('prediction',y),('processed_targets',t)]:
                for name,a,b in GROUPS:
                    q=v[...,a:b];full,parts=decomposition(q);err=float(abs(full-sum(parts.values())))
                    assert err<1e-12
                    rows.append(dict(model=label,source=i,population=kind,group=name,total=float(full),**{k:float(z) for k,z in parts.items()},identity_error=err,speed=float(q.diff(dim=1).abs().mean()),acceleration=float(q.diff(dim=1).diff(dim=1).abs().mean())))
            def frc(p):return float(ccc25(p.float()[:,None],t.float()[None]).max(1).values.sum())
            raw=frc(y);assert abs(raw-result['per_input'][i]['FRC'])<3e-6
            frcs.append(dict(model=label,source=i,raw_FRC=raw,P8_prediction_only_FRC=frc(p8(y))))
            if i in (0,20,40,60):
                import matplotlib;matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig,axes=plt.subplots(5,5,figsize=(20,12),sharex=True);slow=p8(y).numpy();raw_y=y.numpy()
                for c,ax in enumerate(axes.flat):
                    for k in range(10):ax.plot(raw_y[k,:,c],alpha=.12,lw=.35);ax.plot(slow[k,:,c],alpha=.6,lw=.5)
                    for boundary in range(750,y.shape[1],750):ax.axvline(boundary,color='black',lw=.5,ls='--')
                    ax.set_title(str(c))
                fig.suptitle(f'{label} source{i}: all K10 raw faint / P8 solid; no selection');fig.tight_layout();fig.savefig(OUT/f'{label}_source{i:03d}.png',dpi=100);plt.close(fig)
        print(label,'done',flush=True)
    for name,values in [('temporal_spread_decomposition.csv',rows),('raw_vs_P8_diagnostic.csv',frcs)]:
        with (OUT/name).open('x') as f:w=csv.DictWriter(f,fieldnames=values[0]);w.writeheader();w.writerows(values)
    summary={}
    for path in paths:
        label=path.stem;q=[v for v in frcs if v['model']==label];summary[label]={k:float(np.mean([v[k] for v in q])) for k in ('raw_FRC','P8_prediction_only_FRC')}
    with (OUT/'decomposition_summary.json').open('x') as f:json.dump(summary,f,indent=2)
if __name__=='__main__':torch.set_num_threads(1);main()
