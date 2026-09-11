"""Only completed measurements enter the table; no partial FRD means."""
import json,csv
from pathlib import Path
import numpy as np
from mam_staged.common import ROOT
from hirp.phase15_audit import sha256_file

def main():
    baseline=json.loads((ROOT/'manifest.json').read_text());rows=[]
    for path in sorted((ROOT/'task_multitarget').glob('*.json')):
        payload=json.loads(path.read_text());r=payload['result'];m=r['multi_target'];label=path.stem
        complete=[]
        for st in sorted((ROOT/'frd20'/label).glob('status_*.json')):
            status=json.loads(st.read_text())
            if status['completed'] and status['completed_pairs']==2000:complete.append((st,status))
        frd=complete[-1][1]['FRD'] if complete else None
        row=dict(arm=r['arm'],total_step=r['steps'],new_step=r['additional_steps'],FRC80=m['FRC'],FRC20=m['FRC20_diagnostic'],FRD20=frd,S_MSE80=m['smse'],FRVar80=m['FRVar'],target_side=float(np.mean([p['target_side_mean'] for p in r['per_input']])),one_to_one=float(np.mean([p['one_to_one_mean'] for p in r['per_input']])),checkpoint_sha256=payload['eval_identity']['checkpoint_sha256'],output_policy=r['output_policy'],task_file=str(path),task_sha256=sha256_file(path),FRD_completed_pairs=2000 if complete else None,FRC_band=m['FRC']>=baseline['quality_band_FRC_min'],FRD_band=frd<=baseline['quality_band_FRD_max'] if frd is not None else None)
        for name in ('AU','VA','expression'):
            group=[e['group_diagnostics'][name] for e in r['exports']]
            for metric in group[0]:row[name+'_'+metric]=float(np.mean([x[metric] for x in group]))
        rows.append(row)
    out=ROOT/'analysis';out.mkdir(exist_ok=True)
    if rows:
        with (out/'tasks.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'tasks.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    text=['# Staged quality-protected refinement','', 'Development-80, seed123, native continuous AU, K10. Exact FRD belongs to the frozen 20-source subset; only complete 2000-pair results are reported.','', '| Model | Total steps | FRC80 ↑ | FRC20 ↑ | exact FRD20 ↓ | S-MSE80 | FRVar80 |','|---|---:|---:|---:|---:|---:|---:|','| MAM archive (native rounding; different training budget) | archive | 0.810962319 | 0.766297889 | 172.576423473 | 0.157203704 | 0.058911592 |','| R2 parent | 6000 | 1.181468685 | 1.134674340 | 133.409074259 | 0.043679938 | 0.039064668 |']
    for r in rows:
        vals=[r['arm'],str(r['total_step'])]+[f'{r[k]:.9f}' if r[k] is not None else 'pending' for k in ('FRC80','FRC20','FRD20','S_MSE80','FRVar80')];text.append('| '+' | '.join(vals)+' |')
    text+=['','Baseline audit and original fingerprints: ../refine_v1/RESULTS_LATEST.json (read-only).','','S0 includes the common score guard; it is not unchanged historical R2 continuation. S1−S0 tests the dispersion term, S2−S1 tests the update schedule. No task advantage is inferred from training losses alone.','', 'Final selection awaits the +2000 endpoint and complete FRD20 for each arm. The quality band is a development engineering criterion, not equivalence. Prior A/B remains paused; no additional seeds or architectures are launched.']
    (out/'RESULTS_LATEST.md').write_text('\n'.join(text)+'\n')
    try:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(13,3.5))
        for arm in ('S0_quality','S1_joint','S2_staged'):
            logs=sorted((ROOT/arm).glob('attempt_*/training.jsonl'))
            if not logs:continue
            samples=[]
            for line in logs[-1].read_text().splitlines():
                try:samples.append(json.loads(line))
                except json.JSONDecodeError:pass
            for ax,key in zip(axes,('quality','disp','guard_D')):
                ax.plot([r['additional_step'] for r in samples],[r[key] for r in samples],label=arm,alpha=.7);ax.set_title(key);ax.set_xlabel('new optimizer steps')
        axes[0].legend(fontsize=8);fig.suptitle('TRAIN losses, seed123; not development task scores');fig.tight_layout();fig.savefig(out/'training_curves.png',dpi=150);plt.close(fig)
        # Pre-fixed sources0/1/2, all ten candidates, all25 channels; no ranking.
        for path in sorted((ROOT/'task_multitarget').glob('*step8000.json')):
            result=json.loads(path.read_text())['result']
            for index in (0,1,2):
                dest=out/f'{path.stem}_source{index:03d}_all10.png'
                if dest.exists():continue
                prediction=np.load(result['exports'][index]['path']);n=prediction.shape[1]
                pooled=np.stack([prediction[:,i:min(i+8,n)].mean(1) for i in range(0,n,8)],1)
                fig,axes=plt.subplots(25,2,figsize=(13,36))
                for channel in range(25):
                    for k in range(10):
                        axes[channel,0].plot(np.arange(n),prediction[k,:,channel],linewidth=.4,alpha=.6)
                        axes[channel,1].plot(np.arange(0,n,8),pooled[k,:,channel],linewidth=.5,alpha=.7)
                    for ax in axes[channel]:ax.set_ylim((-1,1) if 15<=channel<17 else (0,1));ax.set_ylabel(str(channel))
                axes[0,0].set_title('Raw, all10 candidates');axes[0,1].set_title('8-frame means, all10 candidates')
                fig.suptitle(f'{path.stem}, fixed source {index}; channel0..24; continuous AU');fig.tight_layout();fig.savefig(dest,dpi=90);plt.close(fig)
    except ImportError:pass
if __name__=='__main__':main()
