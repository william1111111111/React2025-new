"""Actual checkpoint curves and joint endpoint decision, no new inference."""
import json,csv
from pathlib import Path
import numpy as np
import torch
from .balance_common import ROOT,ARMS
from .dynamics import decomposition,GROUPS
from hirp.phase15_audit import sha256_file
from hirp.phase24 import verify_files

def dump(name,rows):
    if rows:
        with (ROOT/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def main():
    metrics=[];parts=[];grad=[];bounds=[]
    for arm in ARMS:
        log=ROOT/arm/'attempt_000/training.jsonl'
        if log.exists():
            logs=[]
            for l in log.read_text().splitlines():
                try:logs.append(json.loads(l))
                except json.JSONDecodeError:pass
            for r in logs:
                if 'r_actual' in r:
                    grad.append(dict(arm=arm,step=r['phase_step'],ratio=r['r_actual'],lambda_task=r['lambda_task'],lambda_next=r['lambda_next'],lambda_goal=r['lambda_goal'],EMA_A=r['EMA_A'],EMA_B=r['EMA_B'],cap_hit=r['cap_hit'],FM_gradient=r['FM_gradient'],task_gradient=r['task_gradient'],weighted_task_gradient=r['weighted_task_gradient'],cosine=r['FM_task_cosine'],total_gradient=r['gradient_norm'],update_norm=r['optimizer_update_norm']))
            if logs:
                tail=[r for r in logs if r['phase_step']>1500 and 'r_actual' in r]
                bounds.append(dict(arm=arm,actual_steps=logs[-1]['phase_step'],cap_fraction=sum(r['cap_hit_applied'] for r in logs)/len(logs),late_median_ratio=float(np.median([r['r_actual'] for r in tail])) if tail else None,late_fraction_in_half_to_two=float(np.mean([.5<=r['r_actual']<=2 for r in tail])) if tail else None))
        for step in (16500,17000,18000):
            label=f'seed123_{arm}_step{step}';p=ROOT/'task_multitarget'/f'{label}.json'
            if not p.exists():continue
            payload=json.loads(p.read_text());x=payload['result'];m=x['multi_target'];cache=ROOT/f'decomposition_{arm}_{step}.json'
            if cache.exists():
                d=json.loads(cache.read_text());assert d['task_sha256']==sha256_file(p)
                rows=d['rows']
            else:
                rows=[]
                for i,item in enumerate(x['exports']):
                    verify_files([item]);y=torch.from_numpy(np.load(item['path'])).double()
                    for group,a,b in GROUPS:
                        full,components=decomposition(y[...,a:b]);assert abs(float(full-sum(components.values())))<1e-12
                        rows.append(dict(arm=arm,step=step-16000,source=i,group=group,weight=(b-a)/25,total=float(full),**{k:float(v) for k,v in components.items()}))
                cache.write_text(json.dumps(dict(task_sha256=sha256_file(p),rows=rows),indent=2))
            parts.extend(rows)
            fs=sorted((ROOT/'frd20'/label).glob('status_*.json'));frd=json.loads(fs[-1].read_text()) if fs else {}
            aggregate={k:sum(r[k]*r['weight'] for r in rows)/80 for k in ('total','DC','slow','fast')}
            metrics.append(dict(arm=arm,step=step-16000,global_step=step,FRC=m['FRC'],exact_FRD20=frd.get('FRD') if frd.get('completed') else None,FRD_pairs=frd.get('completed_pairs',0),S_MSE=m['smse'],FRVar=m['FRVar'],target_coverage=float(np.mean([r['target_side_mean'] for r in x['per_input']])),one_to_one=float(np.mean([r['one_to_one_mean'] for r in x['per_input']])),**aggregate,checkpoint_sha256=payload['eval_identity']['checkpoint_sha256']))
    dump('checkpoint_curves.csv',metrics);dump('gradient_ratio_curve.csv',grad);dump('decomposition_per_source.csv',parts)
    (ROOT/'controller_execution.json').write_text(json.dumps(bounds,indent=2))
    final=[r for r in metrics if r['step']==2000]
    (ROOT/'FINAL_RESULTS.json').write_text(json.dumps(final,indent=2))
    text='# Prior/task balance — actual progress and endpoints\n\nSingle GPU1, seed123. Both arms inherit identical G1-16000 weights/AdamW, rho=.05, 2000 new steps. TRAIN calibration16 + short smoke24 updates are disclosed engineering cost, not production steps. Native continuous AU, FP64 Euler16, Development80/K10; exactFRD20 is the fixed subset, complete2000pairs only. No new generation policy or metric definition.\n\n'
    text+='| System | new steps | FRC80 | exact FRD20 | S-MSE80 | FRVar80 | target coverage |\n|---|---:|---:|---:|---:|---:|---:|\n'
    text+='| MAM archive | — | .810962319 | 172.576423473 | .157203704 | .058911592 | .042801 |\n| T0 archive | — | .859622576 | 133.876667407 | .081215642 | .053756177 | .058758 |\n| G0 archive | — | .977341087 | 139.140884363 | .061500423 | .046456564 | .065558 |\n| G1 parent | — | .678053116 | 131.143595554 | .094315603 | .039878245 | .050465 |\n'
    for r in metrics:
        text+=f"| {r['arm']} | {r['step']} | {r['FRC']:.9f} | {r['exact_FRD20']} | {r['S_MSE']:.9f} | {r['FRVar']:.9f} | {r['target_coverage']:.6f} |\n"
    text+='\nController: goal ratio1 is prespecified, not a success metric. Per-node norms/cosine/update norms and cap flags are in gradient_ratio_curve.csv. The [0.5,2] diagnostic band is descriptive only and never changes weights, bounds, budget, or checkpoint choice.\n'
    for r in bounds:text+=f"\n{r['arm']}: actual steps={r['actual_steps']}; cap fraction={r['cap_fraction']:.3f}; last500-step median weighted ratio={r['late_median_ratio']}; fraction within[0.5,2]={r['late_fraction_in_half_to_two']}.\n"
    decision='Pending completed fixed endpoints; no success inferred from loss, ratio, or DC.'
    if len(final)==2 and all(r['FRD_pairs']==2000 and r['exact_FRD20'] is not None for r in final):
        a=next(r for r in final if r['arm']=='B0-fixed');b=next(r for r in final if r['arm']=='B1-ratio')
        gates=b['FRC']>.810962319 and b['exact_FRD20']<172.576423473
        combo=(gates and b['FRC']>a['FRC'] and b['exact_FRD20']<a['exact_FRD20'] and b['S_MSE']>=a['S_MSE'] and b['S_MSE']>.081215642 and b['target_coverage']>=a['target_coverage'])
        decision=('Retain ratio control as a candidate training configuration: endpoint improves the measured quality/diversity/coverage combination over B0. This is not a new method claim or independent confirmation; MAM diversity .157203704 remains the reference.' if combo else 'End this optimization hypothesis after the authorized2000steps: no measured joint improvement meeting the prespecified quality floor and preserving diversity/coverage over B0. Preserve all endpoints; no lambda/rho scan or added budget. Explicit global state or paired-time conditioning remains unimplemented.')
        if b['S_MSE']<.157203704:decision+=' The MAM S-MSE reference is not reached; the full objective is not declared achieved.'
    text+='\nDecision: '+decision+'\n\nDC/slow/fast use equal-source weighting and group weights15/25,2/25,8/25; algebraic total is separate from official FP32 S-MSE. DC is not a semantic mode count, and fast is not automatically noise. Target coverage uses the actual saved target-side max CCC over full target slots, not800 independent people. Reused development data; MAM output rounding and budget differ. Gradient norm ratios are not AdamW update contributions.\n'
    (ROOT/'RESULTS_LATEST.md').write_text(text)
    if grad or metrics:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(2,3,figsize=(12,7))
        for arm in ARMS:
            q=[r for r in grad if r['arm']==arm]
            for ax,key in zip(axs[0],['ratio','lambda_task','update_norm']):
                ax.plot([r['step'] for r in q],[r[key] for r in q],label=arm);ax.set_title(key)
            q=[r for r in metrics if r['arm']==arm]
            for ax,key in zip(axs[1],['FRC','S_MSE','DC']):
                ax.plot([r['step'] for r in q],[r[key] for r in q],marker='o',label=arm);ax.set_title(key)
        axs[0,0].axhline(1,color='gray',linestyle='--')
        for ax in axs.flat:ax.set_xlabel('additional steps');ax.legend();ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(ROOT/'curves.png',dpi=150);plt.close(fig)
if __name__=='__main__':torch.set_num_threads(1);main()
