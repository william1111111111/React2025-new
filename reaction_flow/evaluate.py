"""Versioned numerical validation; unchanged FP32 K10 production predictions."""
import argparse,importlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from reaction_flow.sampler import load_checkpoint
from reaction_flow.export import recording_noise
from reaction_flow.export import export_full
from hirp.phase24 import normalization,verify_files,dependencies,legacy_snapshot,LEGACY
from hirp.phase23_cache import cached,save_cached
from hirp.phase15_audit import canonical_hash,sha256_file
from hirp.task_phase24 import raw_metrics,check_prediction
from reaction_flow.train import configure_flow as configure

ROOT=Path('runs/reaction_flow/trajectory_v1')
OLD=Path('runs/phase24/evaluation_v1')

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();configure()
    manifest=json.loads((OLD/'multitarget_development_manifest.json').read_text());normalization(manifest)
    processed=json.loads((OLD/'processed_targets/completed.json').read_text())
    assert processed['result_sha256']==canonical_hash(processed['result'])
    assert processed['eval_identity']['manifest_sha256']==sha256_file(OLD/'multitarget_development_manifest.json')
    frozen=json.loads((OLD/'dependency_snapshot.json').read_text());verify_files(frozen['legacy']);verify_files(processed['result']['files']);verify_files([manifest['noise']])
    # Resource-only gate; never changes K, frames, checkpoint or metric.
    import subprocess,os
    physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    while True:
        available=float(subprocess.check_output(['nvidia-smi','--id='+physical_gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if available>=4096:break
        print('WAIT_GPU_FREE_MIB',physical_gpu,available,'required',4096,flush=True);time.sleep(30)
    model,meta=load_checkpoint(a.checkpoint,a.device)
    assert meta['phase'] in ('A','F-cont','F-task')
    assert meta['global_step'] in (2000,6000,12000,14000)
    assert len(meta['rows'])==meta['phase_step']
    run_manifest=json.loads((ROOT/'manifest.json').read_text())
    assert meta['manifest_sha256']==sha256_file(ROOT/'manifest.json')
    label=f"seed123_{meta['phase']}_step{meta['global_step']}"
    out=ROOT/'task_multitarget';out.mkdir(exist_ok=True)
    import hashlib
    noise_fingerprints=[dict(clip_id=s['clip_id'],length=s['length'],sha256=hashlib.sha256(recording_noise(model.config,s['clip_id'],s['length']).numpy().tobytes()).hexdigest()) for s in manifest['sources']]
    identity=dict(version='reaction-flow-native-math-v2',attention_backend='explicit torch SDPA math for both FP32 production and FP64 check',noise_content_sha256=canonical_hash(noise_fingerprints),runtime=dict(torch=torch.__version__,numpy=np.__version__),noise_protocol=dict(seed=model.config.evaluation_seed,key='clip_id, sample index, absolute750-block start; assembled full-recording noise',dimension=24,K=10),solver='Euler16',numerical_protocol_sha256=sha256_file(ROOT/'numerical_protocol.json'),checkpoint_sha256=sha256_file(a.checkpoint),arm=meta['phase'],seed=model.config.seed,step=meta['global_step'],coordinate_stats_sha256=meta['coordinate_stats_sha256'],dependencies=dict(hirp=dependencies(),task={str(p):sha256_file(p) for folder in ('mam_target','mam_refine','reaction_flow') for p in sorted(Path(folder).glob('*.py'))}),legacy=frozen['legacy'],target_manifest_sha256=sha256_file(OLD/'multitarget_development_manifest.json'),processed_index_sha256=sha256_file(OLD/'processed_targets/completed.json'))
    path=out/f'{label}.json'
    if cached(path,identity) is not None:return
    sys.path.insert(0,str(LEGACY));from framework.utils.compute_metrics import compute_metrics
    mean=torch.from_numpy(np.load('external/FaceVerse/mean_face.npy')).float();std=torch.from_numpy(np.load('external/FaceVerse/std_face.npy')).float().clamp_min(1e-8)
    preds=[];targets=[];speakers=[];exports=[];started=time.perf_counter();torch.cuda.reset_peak_memory_stats(a.device)
    export_dir=ROOT/'exports'/'math_v2'/label;export_dir.mkdir(parents=True,exist_ok=True)
    for i,s in enumerate(manifest['sources']):
        normalization(manifest);verify_files(list(s['files'].values())+s['targets'])
        noise=recording_noise(model.config,s['clip_id'],s['length']).to(a.device)
        streams={k:torch.from_numpy(np.load(v['path'])).float() for k,v in s['files'].items()}
        face=streams['speaker_3dmm'];streams['speaker_3dmm']=((face[:,0] if face.ndim==3 else face)-mean)/std
        ident=dict(identity_sha256=canonical_hash(identity),source=i);sidecar=export_dir/f'{i:03d}.json';item=cached(sidecar,ident)
        if item is None:
            kwargs={k:v.to(a.device) for k,v in streams.items()}
            pred=export_full(model,**kwargs,source_length=s['length'],noise=noise)
            repeat=export_full(model,**kwargs,source_length=s['length'],noise=noise,candidate_chunk=3)
            error=float((pred-repeat).abs().max())
            checks=check_prediction(pred,s['length']);checks['candidate_consistency_max_error']=error
            checks['original_strict_check_passed']=error<1e-5
            if error>=1e-5:
                # Precision diagnostic only: restore exact original FP32 weights
                # afterwards; pred remains the unmodified FP32 K10 output.
                model.double();double_kwargs={k:v.double() for k,v in kwargs.items()}
                ref=export_full(model,**double_kwargs,source_length=s['length'],noise=noise.double())
                ref_chunk=export_full(model,**double_kwargs,source_length=s['length'],noise=noise.double(),candidate_chunk=3)
                model.float()
                double_error=float((ref-ref_chunk).abs().max())
                reference_error=max(float((pred.double()-ref).abs().max()),float((repeat.double()-ref).abs().max()))
                if double_error>1e-10 or reference_error>1e-4:raise RuntimeError('unexplained numerical discrepancy')
                ccc=importlib.import_module('framework.metrics.FRC').concordance_correlation_coefficient
                y=torch.from_numpy(np.load(processed['result']['files'][i]['path']))
                matrices=[np.array([[float(ccc(t.numpy(),q.numpy())[0]) for t in y] for q in pp]) for pp in (pred,repeat)]
                multi_error=abs(float(matrices[0].max(1).sum()-matrices[1].max(1).sum()))
                single_error=abs(float(matrices[0][:,0].sum()-matrices[1][:,0].sum()))
                if max(multi_error,single_error)>1e-6:raise RuntimeError('numerical difference affects task score beyond protocol')
                checks.update(fp64_chunk_max=double_error,fp32_to_fp64_max=reference_error,multi_FRC_chunk_abs_error=multi_error,single_FRC_chunk_abs_error=single_error)

            dest=export_dir/f'{i:03d}.npy'
            with dest.open('xb') as f:np.save(f,pred.numpy())
            item=dict(path=str(dest),sha256=sha256_file(dest),checks=checks);save_cached(sidecar,ident,item)
        else:verify_files([item]);pred=torch.from_numpy(np.load(item['path']))
        target=torch.from_numpy(np.load(processed['result']['files'][i]['path']))
        groups={}
        for group,lo,hi in [('AU',0,15),('VA',15,17),('expression',17,25)]:
            q=pred[...,lo:hi];y=target[...,lo:hi];flat=q.flatten(1)
            distances=torch.cdist(flat,flat).square()/flat.shape[1]
            groups[group]=dict(candidate_pair_mse_offdiagonal=float(distances.sum()/90),temporal_variance=float(q.var(1).mean()),speed_abs=float(q.diff(dim=1).abs().mean()),acceleration_abs=float(q.diff(dim=1).diff(dim=1).abs().mean()),centered_candidate_pair_mse_offdiagonal=float(torch.cdist((q-q.mean(1,keepdim=True)).flatten(1),(q-q.mean(1,keepdim=True)).flatten(1)).square().sum()/(90*flat.shape[1])),target_temporal_variance=float(y.var(1).mean()),target_speed_abs=float(y.diff(dim=1).abs().mean()),minimum=float(q.min()),maximum=float(q.max()))
        item=dict(item,group_diagnostics=groups)
        preds.append(pred);targets.append(target);speakers.append(streams['speaker_emotion']);exports.append(item)
        print(label,i+1,flush=True)
    shuffle=[]
    if meta['global_step'] in (12000,14000):
        # One prespecified cyclic derangement per established session, never selected by scores.
        from hirp.losses.energy_score import paired_energy_score
        groups={}
        for i,s in enumerate(manifest['sources']):groups.setdefault(s['clip_id'].split('/')[1],[]).append(i)
        permutation={}
        for indices in groups.values():
            if len(indices)<2:raise ValueError('session cannot be deranged')
            for i,j in zip(indices,indices[1:]+indices[:1]):permutation[i]=j
        for i,j in sorted(permutation.items()):
            original=manifest['sources'][i];source=manifest['sources'][j];verify_files(list(source['files'].values()))
            inputs={k:torch.from_numpy(np.load(v['path'])).float() for k,v in source['files'].items()}
            face=inputs['speaker_3dmm'];inputs['speaker_3dmm']=((face[:,0] if face.ndim==3 else face)-mean)/std
            z=recording_noise(model.config,original['clip_id'],source['length']).to(a.device)
            shuffled=export_full(model,**{k:v.to(a.device) for k,v in inputs.items()},source_length=source['length'],noise=z)
            n=min(original['length'],source['length']);mask=torch.ones(1,n,dtype=torch.bool)
            def score(p):return float(paired_energy_score(p[None,:,:n],targets[i][0,None,:n],mask,channel_scale=torch.ones(25)))
            correct=score(preds[i]);wrong=score(shuffled)
            shuffle.append(dict(source=i,shuffled_source=j,common_length=n,correct_paired_ES=correct,shuffled_paired_ES=wrong,gap=wrong-correct))
            print('SHUFFLE',i,flush=True)
    metrics=compute_metrics(speakers,preds,targets,threads=4,metric_names=manifest['metrics']);raw=raw_metrics(preds,targets,speakers)
    assert abs(metrics['FRC']-np.mean([r['FRC'] for r in raw]))<1e-6
    from scipy.optimize import linear_sum_assignment
    subset=json.loads(Path('runs/mam_target/task_v1/frd20/protocol.json').read_text())['source_indices']
    for i,row in enumerate(raw):
        matrix=np.asarray(row['CCC_candidate_target']);rr,cc=linear_sum_assignment(-matrix);slots=matrix.argmax(1).tolist();refs=manifest['sources'][i]['targets']
        row.update(target_side_mean=float(matrix.max(0).mean()),one_to_one_mean=float(matrix[rr,cc].mean()),matched_target_slots=slots,unique_target_contents=len(set(r['sha256'] for r in refs)),selected_target_contents=len(set(refs[j]['sha256'] for j in slots)))
    metrics['FRC20_diagnostic']=float(np.mean([raw[i]['FRC'] for i in subset]))
    save_cached(path,identity,dict(parent_sha256=meta['parent_sha256'],additional_steps=meta['phase_step'],arm=meta['phase'],seed=model.config.seed,steps=meta['global_step'],NFE=16,shuffle_common_mask=shuffle,output_policy='continuous AU native',multi_target=metrics,single_paired_FRC=float(np.mean([r['single_paired_FRC'] for r in raw])),per_input=raw,exports=exports,seconds=time.perf_counter()-started,peak_memory_bytes=torch.cuda.max_memory_allocated(a.device)))

if __name__=='__main__':main()
