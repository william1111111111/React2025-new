"""Bounded exact FRD, atomic candidate-target pairs, never partial means."""
import argparse,json,time,inspect
from pathlib import Path
import numpy as np
from tslearn.metrics import dtw
from hirp.phase15_audit import sha256_file,canonical_hash
from hirp.phase24 import verify_files,normalization
from hirp.phase23_cache import cached,save_cached
from hirp.train_phase22 import write

ROOT=Path('runs/mam_target/refine_v1')
OLD=Path('runs/phase24/evaluation_v1')

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',default='MAM_archive_offline');p.add_argument('--task',type=Path);p.add_argument('--seconds',type=float,default=7200);p.add_argument('--max-pairs',type=int,default=2000);a=p.parse_args()
    manifest=json.loads((OLD/'multitarget_development_manifest.json').read_text());normalization(manifest)
    selected={}
    for i,s in sorted(enumerate(manifest['sources']),key=lambda v:v[1]['clip_id']):selected.setdefault(s['clip_id'].split('/')[1],i)
    assert len(selected)==20
    out=ROOT/'frd20';out.mkdir(exist_ok=True)
    protocol=dict(version='mam-refine-exact-frd20-v1',source_indices=list(selected.values()),selection='lexicographically first source per session, no scores',source_manifest_sha256=sha256_file(OLD/'multitarget_development_manifest.json'),models=[f'seed123_{arm}_step{step}' for arm in ('R2-cont','R3-cover') for step in (8000,)],pairs_per_model=2000,dtws_per_pair=3,full_frames=True,seconds_per_invocation=7200,metric_source=dict(path='/home/zhengshiyi/react2025/framework/metrics/FRD.py',sha256=sha256_file('/home/zhengshiyi/react2025/framework/metrics/FRD.py')),dtw_source=dict(path=inspect.getfile(dtw),sha256=sha256_file(inspect.getfile(dtw))))
    protocol_path=out/'protocol.json'
    if protocol_path.exists():assert json.loads(protocol_path.read_text())==protocol
    else:write(protocol_path,protocol)
    task=a.task or OLD/'task_multitarget'/f'{a.model}.json';payload=json.loads(task.read_text())
    assert canonical_hash(payload['result'])==payload['result_sha256']
    assert a.model in protocol['models']
    assert payload['eval_identity_sha256']==canonical_hash(payload['eval_identity'])
    exports=payload['result']['exports'];targets=json.loads((OLD/'processed_targets/completed.json').read_text())
    assert canonical_hash(targets['result'])==targets['result_sha256']
    identity=dict(protocol_sha256=sha256_file(protocol_path),task_sha256=sha256_file(task),processor_index_sha256=sha256_file(OLD/'processed_targets/completed.json'),implementation_sha256=sha256_file(__file__))
    dest=out/a.model;dest.mkdir(exist_ok=True);start=time.perf_counter();done=0;new=0;values=[]
    for i in selected.values():
        verify_files([exports[i],targets['result']['files'][i]])
        pred=np.load(exports[i]['path']);target=np.load(targets['result']['files'][i]['path']);assert pred.shape==target.shape==(10,manifest['sources'][i]['length'],25)
        matrix=np.full((10,10),np.nan)
        for k in range(10):
            for j in range(10):
                path=dest/f'{i:03d}_{k:02d}_{j:02d}.json';ident=dict(**identity,source=i,candidate=k,target=j)
                result=cached(path,ident)
                if result is None:
                    if new>=a.max_pairs or time.perf_counter()-start>=a.seconds:continue
                    t=time.perf_counter();parts=[float(dtw(pred[k,:,s:e].astype(np.float32),target[j,:,s:e].astype(np.float32))) for s,e in ((0,15),(15,17),(17,25))]
                    result=dict(parts=parts,distance=parts[0]/15+parts[1]+parts[2]/8,seconds=time.perf_counter()-t)
                    save_cached(path,ident,result);new+=1
                    print(a.model,i,k,j,result['seconds'],flush=True)
                matrix[k,j]=result['distance'];done+=1
        if np.isfinite(matrix).all():values.append(float(matrix.min(1).sum()))
    summary=dict(model=a.model,completed_pairs=done,requested_pairs=2000,completed=done==2000,complete_source_count=len(values),FRD=float(np.mean(values)) if done==2000 else None,new_pairs=new,wall_seconds=time.perf_counter()-start)
    write(dest/f'status_{len(list(dest.glob("status_*.json"))):03d}.json',summary);print(summary,flush=True)

if __name__=='__main__':main()
