"""Actual-only endpoint and decomposition report; no synthetic completion."""
import json,csv
from pathlib import Path
import numpy as np
import torch
from .prepare_shared import ROOT
from .dynamics import GROUPS,decomposition
from hirp.phase24 import verify_files

def main():
    rows=[];parts=[];blocks=[];quality=[]
    for arm in ('G0-local','G1-shared'):
        label=f'seed123_{arm}_step16000';path=ROOT/'task_multitarget'/f'{label}.json'
        if not path.exists():continue
        payload=json.loads(path.read_text());x=payload['result'];m=x['multi_target'];st=json.loads((ROOT/arm/'attempt_000/status.json').read_text());frds=sorted((ROOT/'frd20'/label).glob('status_*.json'));frd=json.loads(frds[-1].read_text()) if frds else {}
        rows.append(dict(model=arm,rho=payload['eval_identity']['initial_prior']['rho'],FRC80=m['FRC'],FRC20=m['FRC20_diagnostic'],exact_FRD20=frd.get('FRD') if frd.get('completed') else None,FRD_pairs=frd.get('completed_pairs',0),S_MSE80=m['smse'],FRVar80=m['FRVar'],actual_steps=st['actual'],valid_frames=st['valid_frames'],seconds=st['seconds'],checkpoint_sha256=payload['eval_identity']['checkpoint_sha256'],target_side_mean=float(np.mean([r['target_side_mean'] for r in x['per_input']])),paired_shuffle_gap=float(np.mean([r['gap'] for r in x['shuffle_common_mask']]))))
        for i,item in enumerate(x['exports']):
            verify_files([item]);y=torch.from_numpy(np.load(item['path'])).double()
            matrix=np.asarray(x['per_input'][i]['CCC_candidate_target'])
            for k,v in enumerate(matrix.max(1)):quality.append(dict(model=arm,source=i,candidate=k,best_CCC=float(v)))
            for group,a,b in GROUPS:
                q=y[...,a:b];total,components=decomposition(q);assert abs(float(total-sum(components.values())))<1e-12
                parts.append(dict(model=arm,source=i,group=group,total=float(total),**{k:float(v) for k,v in components.items()},weight=(b-a)/25,speed=float(q.diff(dim=1).abs().mean()),acceleration=float(q.diff(dim=1).diff(dim=1).abs().mean())))
                means=[q[:,start:start+750].mean(1) for start in range(0,q.shape[1],750)]
                for j in range(len(means)-1):
                    u,v=means[j],means[j+1];cu=u-u.mean(0);cv=v-v.mean(0)
                    blocks.append(dict(model=arm,source=i,group=group,left_block=j,right_block=j+1,left_frames=min(750,q.shape[1]-j*750),right_frames=min(750,q.shape[1]-(j+1)*750),mean_abs_change=float((v-u).abs().mean()),candidate_covariance=float((cu*cv).sum(0).mean()/9)))
    for name,vals in [('FINAL_RESULTS.csv',rows),('decomposition.csv',parts),('block_diagnostics.csv',blocks),('candidate_quality.csv',quality)]:
        if vals:
            with (ROOT/name).open('w') as f:w=csv.DictWriter(f,fieldnames=vals[0]);w.writeheader();w.writerows(vals)
    (ROOT/'FINAL_RESULTS.json').write_text(json.dumps(rows,indent=2))
    text='# Shared/local noise: actual results\n\nNative continuous AU, FP64 Euler16, Development80/K10; exactFRD20 only when all2000 pairs complete. Inherited T0-14000 and R2 encoder warmstart costs apply.\n\n| Model | FRC80 | exactFRD20 | S-MSE80 | FRVar80 |\n|---|---:|---:|---:|---:|\n| MAM native archive | .810962319 | 172.576423473 | .157203704 | .058911592 |\n| T0 parent14000 | .859622576 | 133.876667407 | .081215642 | .053756177 |\n'
    for r in rows:text+=f"| {r['model']} | {r['FRC80']:.9f} | {r['exact_FRD20']} | {r['S_MSE80']:.9f} | {r['FRVar80']:.9f} |\n"
    text+='\nNo missing endpoint is filled with prior-model metrics. Compare DC/slow/fast and block covariance jointly with quality and source/noise-factor probes; no DC value is a required human variance target. MAM native AU/budget differ. Single-seed repeated development evidence, not independent confirmation.\n'
    (ROOT/'RESULTS_LATEST.md').write_text(text)
if __name__=='__main__':torch.set_num_threads(1);main()
