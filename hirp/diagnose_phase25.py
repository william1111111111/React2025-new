"""Read-only matching diagnostics from frozen Phase24 CCC matrices."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from .phase15_audit import sha256_file
from .train_phase22 import write

def main():
    root=Path('runs/phase25/timescale_v1/coverage')
    root.mkdir(parents=True,exist_ok=False)
    summaries=[];raw=[];files=[]
    for path in sorted(Path('runs/phase24/evaluation_v1/task_multitarget').glob('*.json')):
        payload=json.loads(path.read_text())
        result=payload.get('result',payload)
        if 'per_input' not in result:continue
        files.append(dict(path=str(path),sha256=sha256_file(path)))
        rows=[]
        for item in result['per_input']:
            m=np.asarray(item['CCC_candidate_target'])
            assert m.shape==(10,10) and np.isfinite(m).all()
            r,c=linear_sum_assignment(-m)
            row=dict(model=path.stem,seed=result['seed'],arm=result['arm'],lambda_group=result['lambda_group'],
                     input_index=item['input_index'],official_FRC=m.max(1).sum(),
                     generated_side_mean=m.max(1).mean(),target_side_mean=m.max(0).mean(),
                     one_to_one_mean=m[r,c].mean())
            assert abs(row['official_FRC']-item['FRC'])<1e-7
            rows.append(row);raw.append(row)
        summaries.append({**{k:rows[0][k] for k in ('model','seed','arm','lambda_group')},
                          **{k:float(np.mean([r[k] for r in rows])) for k in ('official_FRC','generated_side_mean','target_side_mean','one_to_one_mean')}})
    assert len(summaries)==16
    for name,rows in [('per_input',raw),('per_model',summaries)]:
        with (root/(name+'.csv')).open('x') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(root/'provenance.json',dict(files=files,source='frozen Phase24 matrices; no new generation',
          targets='all 10 slots retained, including duplicates; uniform frozen slot weights',
          limitations='auxiliary matching is not semantic mode counting or conditional probability recovery'))

if __name__=='__main__':main()
