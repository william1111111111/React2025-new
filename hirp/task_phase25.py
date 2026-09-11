"""New checkpoints on the existing frozen full Development80 task protocol."""
import argparse,importlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from .phase22 import load_checkpoint
from .phase23_long import export_full
from .phase24 import normalization,verify_files,dependencies,legacy_snapshot,LEGACY
from .phase23_cache import cached,save_cached
from .phase15_audit import canonical_hash,sha256_file
from .task_phase24 import raw_metrics,check_prediction
from .train_phase21 import configure

ROOT=Path('runs/phase25/timescale_v1')
OLD=Path('runs/phase24/evaluation_v1')

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();configure()
    manifest=json.loads((OLD/'multitarget_development_manifest.json').read_text());normalization(manifest)
    processed=json.loads((OLD/'processed_targets/completed.json').read_text())
    assert processed['result_sha256']==canonical_hash(processed['result'])
    assert processed['eval_identity']['manifest_sha256']==sha256_file(OLD/'multitarget_development_manifest.json')
    frozen=json.loads((OLD/'dependency_snapshot.json').read_text());verify_files(frozen['legacy']);verify_files(processed['result']['files']);verify_files([manifest['noise']])
    model,meta=load_checkpoint(a.checkpoint,a.device)
    assert meta['train_config']['T']==750 and meta['train_config']['lambda_group']==.1 and meta['prior_mode']=='standard_normal'
    assert meta['arm'] in ('C0','C1','C2') and meta['train_config']['init_seed'] in (123,42,2026)
    assert meta['global_step'] in (128,500,1000,2000,4000,6000)
    assert meta['actual_optimizer_steps']==meta['global_step']==len(meta['training_rows'])
    assert meta['train_config']['B']==4 and meta['train_config']['K']==4
    label=f"seed{meta['train_config']['init_seed']}_{meta['arm']}_step{meta['global_step']}"
    out=ROOT/'task_multitarget';out.mkdir(exist_ok=True)
    identity=dict(version='phase25-task-v1',checkpoint_sha256=sha256_file(a.checkpoint),arm=meta['arm'],seed=meta['train_config']['init_seed'],step=meta['global_step'],scales=meta['scales'],dependencies=dependencies(),legacy=frozen['legacy'],target_manifest_sha256=sha256_file(OLD/'multitarget_development_manifest.json'),processed_index_sha256=sha256_file(OLD/'processed_targets/completed.json'))
    path=out/f'{label}.json'
    if cached(path,identity) is not None:return
    sys.path.insert(0,str(LEGACY));from framework.utils.compute_metrics import compute_metrics
    mean=torch.from_numpy(np.load('external/FaceVerse/mean_face.npy')).float();std=torch.from_numpy(np.load('external/FaceVerse/std_face.npy')).float().clamp_min(1e-8)
    noise=torch.from_numpy(np.load(manifest['noise']['path']))[0,:10].to(a.device)
    preds=[];targets=[];speakers=[];exports=[];started=time.perf_counter();torch.cuda.reset_peak_memory_stats(a.device)
    export_dir=ROOT/'exports'/label;export_dir.mkdir(parents=True,exist_ok=True)
    for i,s in enumerate(manifest['sources']):
        normalization(manifest);verify_files(list(s['files'].values())+s['targets'])
        streams={k:torch.from_numpy(np.load(v['path'])).float() for k,v in s['files'].items()}
        face=streams['speaker_3dmm'];streams['speaker_3dmm']=((face[:,0] if face.ndim==3 else face)-mean)/std
        ident=dict(identity_sha256=canonical_hash(identity),source=i);sidecar=export_dir/f'{i:03d}.json';item=cached(sidecar,ident)
        if item is None:
            kwargs={k:v.to(a.device) for k,v in streams.items()}
            pred=export_full(model,**kwargs,source_length=s['length'],noise=noise)
            repeat=export_full(model,**kwargs,source_length=s['length'],noise=noise,candidate_chunk=3)
            error=float((pred-repeat).abs().max());assert error<1e-5
            checks=check_prediction(pred,s['length']);checks['candidate_consistency_max_error']=error
            dest=export_dir/f'{i:03d}.npy'
            with dest.open('xb') as f:np.save(f,pred.numpy())
            item=dict(path=str(dest),sha256=sha256_file(dest),checks=checks);save_cached(sidecar,ident,item)
        else:verify_files([item]);pred=torch.from_numpy(np.load(item['path']))
        preds.append(pred);targets.append(torch.from_numpy(np.load(processed['result']['files'][i]['path'])));speakers.append(streams['speaker_emotion']);exports.append(item)
        print(label,i+1,flush=True)
    metrics=compute_metrics(speakers,preds,targets,threads=4,metric_names=manifest['metrics']);raw=raw_metrics(preds,targets,speakers)
    assert abs(metrics['FRC']-np.mean([r['FRC'] for r in raw]))<1e-6
    save_cached(path,identity,dict(arm=meta['arm'],seed=meta['train_config']['init_seed'],steps=meta['global_step'],lambda_group=0 if meta['arm']=='C0' else .1,multi_target=metrics,single_paired_FRC=float(np.mean([r['single_paired_FRC'] for r in raw])),per_input=raw,exports=exports,seconds=time.perf_counter()-started,peak_memory_bytes=torch.cuda.max_memory_allocated(a.device)))

if __name__=='__main__':main()
