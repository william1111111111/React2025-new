"""Read-only verification of completed artifacts, histories and old paired scores."""
import json,math
from pathlib import Path
from .phase24 import ROOT,OLD,PLAN,dependencies
from .phase15_audit import sha256_file,canonical_hash
from .train_phase22 import write


def main():
    deps=json.loads((ROOT/'dependency_snapshot.json').read_text());assert deps['hirp']==dependencies()
    for name in ('mc/completed.json','task_multitarget/completed_HiRP.json','task_multitarget/completed_MAM.json'):
        r=json.loads((ROOT/name).read_text())
        for row in r['cases']:assert sha256_file(row['path'])==row['sha256']
    rows=[]
    for p in (ROOT/'task_multitarget').glob('seed*.json'):
        r=json.loads(p.read_text())['result']
        for key,value in r['multi_target'].items():assert math.isfinite(value),(p,key)
        if r['arm']!='C0' and r['lambda_group']!=.1:continue
        oldpath=OLD/f"task_development_full/seed{r['seed']}_{r['arm']}.json";old=json.loads(oldpath.read_text())['result']['metrics']
        errors={k:abs(r['single_paired'][k]-v) for k,v in old.items() if not k.endswith('_time')}
        assert max(errors.values())<1e-6
        rows.append(dict(seed=r['seed'],arm=r['arm'],errors=errors))
    assert len(rows)==9
    write(ROOT/'single_paired_regression.json',dict(cases=rows,maximum_error=max(max(r['errors'].values()) for r in rows)))
    historical=json.loads((OLD/'DELIVERY_INDEX.json').read_text());bad=[]
    for r in historical['files']:
        if sha256_file(r['path'])!=r['sha256']:bad.append(r['path'])
    assert not bad,bad
    write(ROOT/'historical_preservation.json',dict(phase23_files_verified=len(historical['files']),changed=bad,phase23_index_sha256=sha256_file(OLD/'DELIVERY_INDEX.json')))
    fingerprints=[]
    for p in sorted(ROOT.rglob('*')):
        if p.is_file():fingerprints.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha256_file(p),local_array=p.suffix in ('.npy','.npz','.pt','.pth')))
    write(ROOT/'artifact_fingerprints.json',dict(files=fingerprints,weights='existing checkpoints read only; no checkpoint created',arrays='local only, gitignored',new_optimizer_steps=0))
    print('VERIFIED 120 MC, 15 HiRP task, 1 MAM task, 9 single-paired regressions, history preserved',flush=True)


if __name__=='__main__':main()
