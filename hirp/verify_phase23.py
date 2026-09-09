"""Post-run provenance, frozen-history and numerical regression checks."""
import json
from pathlib import Path
from .phase15_audit import sha256_file,canonical_hash
from .train_phase22 import write

ROOT=Path('runs/phase23/tradeoff_v1')


def numbers(value,prefix=''):
    if isinstance(value,dict):
        return {key:val for k,v in value.items() for key,val in numbers(v,prefix+k+'.').items()}
    if isinstance(value,(float,int)):return {prefix:float(value)}
    return {}


def main():
    comparisons=[]
    for seed in (123,42,2026):
        for arm in ('C0','C1','C2'):
            for k in (32,64):
                for bank in (0,1):
                    old=Path(f'runs/phase22/replicated_v1/seed_{seed}/evaluation/{arm}_step2000_bank{bank}_K{k}.json')
                    new=ROOT/f'evaluation_v1/seed_{seed}/{arm}_lambda{0 if arm=="C0" else 0.1:g}_bank{bank}_K{k}.json'
                    a=json.loads(old.read_text());b=json.loads(new.read_text())['result'];aa=numbers(a['aggregate']);bb=numbers(b['aggregate']);assert aa.keys()==bb.keys()
                    differences={key:abs(aa[key]-bb[key]) for key in aa};maximum=max(differences.values());assert maximum<3e-6
                    comparisons.append(dict(seed=seed,arm=arm,K=k,bank=bank,maximum_aggregate_abs_error=maximum,old_sha256=sha256_file(old),new_sha256=sha256_file(new)))
    write(ROOT/'phase22_metric_regression.json',dict(cases=comparisons,max_abs_error=max(r['maximum_aggregate_abs_error'] for r in comparisons),scope='new local evaluation against preserved Phase22 logs; not independent external reproduction'))
    audit=json.loads((ROOT/'initial_audit.json').read_text());print('initial audit keys',list(audit))
    history=audit['historical_files'];changed=[]
    for path,digest in history.items():
        if sha256_file(path)!=digest:changed.append(path)
    assert not changed,changed
    write(ROOT/'historical_preservation.json',dict(files_verified=len(history),changed=changed,initial_audit_sha256=sha256_file(ROOT/'initial_audit.json')))
    manifests=[];checkpoints=[]
    for path in ROOT.rglob('manifest.json'):manifests.append(dict(path=str(path),sha256=sha256_file(path)))
    for seed in (123,42,2026):
        for weight in (.03,.3):
            for arm in ('C1','C2'):
                directory=ROOT/f'lambda_{weight:g}/seed_{seed}/{arm}/attempt_000';s=json.loads((directory/'summary.json').read_text())
                rows=[json.loads(line) for line in (directory/'training.jsonl').read_text().splitlines()]
                assert s['requested_steps']==s['actual_optimizer_steps']==s['training_rows']==len(rows)==2000
                for cp in s['checkpoints']:
                    assert sha256_file(cp['path'])==cp['sha256'];checkpoints.append(dict(seed=seed,arm=arm,lambda_group=weight,**cp))
    write(ROOT/'artifact_fingerprints.json',dict(manifests=manifests,new_checkpoints=checkpoints,reused_checkpoints_sha256=sha256_file(ROOT/'reused_checkpoints.json')))
    print('PROVENANCE VERIFIED')


if __name__=='__main__':main()
