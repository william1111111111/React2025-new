"""Aligned multi-target and single-paired Development80, immutable target processing."""
import argparse,importlib,json,sys,time,math,inspect
import numpy as np
import torch
from .phase24 import *
from .phase23_long import export_full
from .train_phase21 import configure


def check_prediction(p,n):
    if p.shape!=(10,n,25) or not torch.isfinite(p).all():raise ValueError('prediction shape/finite failed')
    if not ((p[:,:,:15]>=0).all() and (p[:,:,:15]<=1).all() and (p[:,:,15:17].abs()<=1.000001).all()):raise ValueError('prediction domain failed')
    torch.testing.assert_close(p[:,:,17:].sum(-1),torch.ones(10,n),atol=3e-6,rtol=0)
    d=(p[:,1:]-p[:,:-1]).abs();boundary=torch.arange(1,n)%750==0
    return dict(frames=n,K=10,boundaries=torch.arange(1,n)[boundary].tolist(),boundary_abs_change=float(d[:,boundary].mean()) if boundary.any() else None,interior_abs_change=float(d[:,~boundary].mean()),finite=True,domain=True)


def processor_targets(manifest,plan,device,Processor):
    out=ROOT/'processed_targets';out.mkdir(exist_ok=True);identity=dict(manifest_sha256=sha256_file(ROOT/'multitarget_development_manifest.json'),dependencies_sha256=sha256_file(ROOT/'dependency_snapshot.json'),seed=24201)
    index=out/'completed.json'
    previous=guarded(plan,lambda:cached(index,identity))
    if previous is not None:
        verify_files(previous['files']);return [torch.from_numpy(np.load(r['path'])) for r in previous['files']]
    processor=Processor(cfg_dir=str(LEGACY),ckpt_dir=str(LEGACY/'pretrained_models/post_processor'),device=torch.device(device),clip_len_test=1000,num_preds=10)
    tests=[];wanted={'shorter':None,'longer':None,'multiple_segments':None}
    for source in manifest['sources']:
        n=source['length']
        for ref in source['targets']:
            if ref['length']<n and wanted['shorter'] is None:wanted['shorter']=(source,ref)
            if ref['length']>n and wanted['longer'] is None:wanted['longer']=(source,ref)
            if ref['length']!=n and max(n,ref['length'])>2000 and wanted['multiple_segments'] is None:wanted['multiple_segments']=(source,ref)
    for label,pair in wanted.items():
        if pair is None:raise ValueError('required real processor case missing: '+label)
        source,ref=pair;n=source['length'];target=torch.from_numpy(np.load(ref['path'])).float()
        with torch.random.fork_rng(devices=[torch.device(device).index]),torch.no_grad():
            torch.manual_seed(24201);a=processor.forward([torch.zeros(10,n,25)],[[target]])[0]
            torch.manual_seed(24201);b=processor.forward([torch.ones(10,n,25)],[[target]])[0]
        error=float((a-b).abs().max());assert error==0 and a.shape==(1,n,25) and torch.isfinite(a).all()
        tests.append(dict(case=label,source_id=source['clip_id'],target_path=ref['path'],source_length=n,target_length=ref['length'],processor_segments=math.ceil(max(n,ref['length'])/1000),prediction_value_max_error=error,shape=list(a.shape),rng_control='same explicit Processor RNG state; original random reparameterization unchanged'))
    if not (ROOT/'unequal_processor_tests.json').exists():write(ROOT/'unequal_processor_tests.json',dict(cases=tests,processor_source=inspect.getfile(Processor),max_value_dependency_error=max(x['prediction_value_max_error'] for x in tests)))
    torch.manual_seed(24201);rows=[];targets=[]
    for i,s in enumerate(manifest['sources']):
        normalization(plan);verify_files(s['targets']);raw=[torch.from_numpy(np.load(r['path'])).float() for r in s['targets']]
        with torch.no_grad():y=processor.forward([torch.zeros(10,s['length'],25)],[raw])[0]
        assert y.shape==(10,s['length'],25) and torch.isfinite(y).all();assert torch.equal(y[0],raw[0])
        path=out/f'{i:03d}.npy'
        with path.open('xb') as f:np.save(f,y.numpy())
        rows.append(dict(path=str(path),sha256=sha256_file(path)));targets.append(y);print('TARGET',i+1,flush=True)
    save_cached(index,identity,dict(files=rows,shared_across_models=True));del processor;torch.cuda.empty_cache();return targets


def raw_metrics(predictions,targets,speakers):
    frc=importlib.import_module('framework.metrics.FRC');tlcc=importlib.import_module('framework.metrics.TLCC');rows=[]
    for i,(p,y,x) in enumerate(zip(predictions,targets,speakers)):
        matrix=[[float(frc.concordance_correlation_coefficient(t.numpy(),q.numpy())[0]) for t in y] for q in p]
        flat=p.flatten(1);dist=torch.cdist(flat,flat).square()/flat.shape[1];center=p-p.mean(1,keepdim=True);cf=center.flatten(1);cd=torch.cdist(cf,cf).square()/cf.shape[1]
        rows.append(dict(input_index=i,CCC_candidate_target=matrix,FRC=float(np.max(matrix,axis=1).sum()),single_paired_FRC=float(np.array(matrix)[:,0].sum()),candidate_channel_temporal_variance=p.var(1).tolist(),S_MSE_candidate_pair=dist.tolist(),temporal_S_MSE_candidate_pair=cd.tolist(),TLCC_first_candidate=float(tlcc._func(p,x))))
    return rows


def mam_model(device):
    sys.path.insert(0,str(MAM))
    from regnn.conditional_model import ConditionalREGNN,ConditionalREGNNConfig
    from regnn.emotion_query_mamba import EmotionQueryMambaConfig,EmotionQueryResidualMamba
    from regnn.eval_query_mamba_official_test import generate_official_offline_query_batch
    path=MAM/'checkpoints/mam_reactor_offline_evidence_epoch0003.pth';m=torch.load(path,map_location='cpu',weights_only=False)
    anchor=ConditionalREGNN(ConditionalREGNNConfig(**m['anchor_model_config']));cfg=EmotionQueryMambaConfig.from_checkpoint_payload(dict(m['emotion_query_model_config']));model=EmotionQueryResidualMamba(anchor,cfg).to(device);model.load_state_dict(m['state_dict']);model.eval()
    return model,generate_official_offline_query_batch,dict(checkpoint=str(path),sha256=sha256_file(path),method=m.get('method'),anchor_config=m['anchor_model_config'],query_config=m['emotion_query_model_config'],actual_sources={inspect.getfile(x):sha256_file(inspect.getfile(x)) for x in (ConditionalREGNN,EmotionQueryResidualMamba,generate_official_offline_query_batch)},archive_evidence=[str(MAM/'PROVENANCE.md'),str(MAM/'infer.sh')],inference=dict(residual_scale=.95,style_residual_scale=.95,channel_multipliers=[1.8]*15+[.4]*2+[1.6]*8,data_clamp=True),comparison='one archived MAM seed, native fixed query set + AU rounding, pretrained anchor/warmstart/B_fit; unequal training budget vs scratch HiRP')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda:6');parser.add_argument('--mam-only',action='store_true');args=parser.parse_args();configure();plan,snapshot=runtime();device=args.device
    manifest=json.loads((ROOT/'multitarget_development_manifest.json').read_text());normalization(plan);verify_files([manifest['noise']])
    for s in manifest['sources']:verify_files(list(s['files'].values())+s['targets'])
    sys.path.insert(0,str(LEGACY))
    from framework.modules.post_processor import Processor
    from framework.utils.compute_metrics import compute_metrics
    targets=processor_targets(manifest,plan,device,Processor)
    mean=torch.from_numpy(np.load('external/FaceVerse/mean_face.npy')).float();std=torch.from_numpy(np.load('external/FaceVerse/std_face.npy')).float().clamp_min(1e-8)
    noise=torch.from_numpy(np.load(manifest['noise']['path']))[0,:10].to(device)
    candidates=[dict(arm='MAM',seed=1,lambda_group=None)] if args.mam_only else json.loads((ROOT/'candidate_manifest.json').read_text())
    out=ROOT/'task_multitarget';out.mkdir(exist_ok=True);completed=[]
    for point in candidates:
        mam=point['arm']=='MAM';label='MAM_archive_offline' if mam else f"seed{point['seed']}_{point['arm']}_lambda{point['lambda_group']:g}"
        if mam:
            model,generate,mam_meta=mam_model(device)
            if not (ROOT/'mam_integration.json').exists():write(ROOT/'mam_integration.json',mam_meta)
            model_identity=mam_meta
        else:model,meta=load_candidate(point,device);model_identity=point
        identity=dict(version='phase24-multitarget-v1',model=model_identity,manifest_sha256=sha256_file(ROOT/'multitarget_development_manifest.json'),dependencies_sha256=canonical_hash(snapshot),processor_cache_sha256=sha256_file(ROOT/'processed_targets/completed.json'),actual_metric_source=inspect.getfile(compute_metrics))
        path=out/f'{label}.json';prior=guarded(plan,lambda:cached(path,identity))
        if prior is not None:completed.append(dict(path=str(path),sha256=sha256_file(path)));continue
        export_dir=ROOT/'exports'/label;export_dir.mkdir(parents=True,exist_ok=True);preds=[];speakers=[];exports=[];started=time.perf_counter();torch.cuda.reset_peak_memory_stats(device)
        for i,s in enumerate(manifest['sources']):
            normalization(plan);verify_files(list(s['files'].values()));streams={key:torch.from_numpy(np.load(r['path'])).float() for key,r in s['files'].items()};face=streams['speaker_3dmm'];streams['speaker_3dmm']=((face[:,0] if face.ndim==3 else face)-mean)/std
            dest=export_dir/f'{i:03d}.npy';sidecar=export_dir/f'{i:03d}.json';item_identity=dict(case=canonical_hash(identity),source_index=i);item=guarded(plan,lambda:cached(sidecar,item_identity))
            if item is not None:
                verify_files([item]);pred=torch.from_numpy(np.load(dest));checks=item['checks']
            else:
                if mam:
                    pred=generate(model,[streams['speaker_audio']],[streams['speaker_emotion']],[streams['speaker_3dmm']],[s['length']],torch.device(device),750,1,10,.95,True,.95,[1.8]*15+[.4]*2+[1.6]*8)[0]
                    # Native fixed candidate order is preserved, never relabelled as iid samples.
                    repeat=generate(model,[streams['speaker_audio']],[streams['speaker_emotion']],[streams['speaker_3dmm']],[s['length']],torch.device(device),750,2,10,.95,True,.95,[1.8]*15+[.4]*2+[1.6]*8)[0]
                    chunk_error=float((pred-repeat).abs().max());assert chunk_error<1e-4
                else:
                    kwargs={k:x.to(device) for k,x in streams.items()};pred=guarded(plan,lambda:export_full(model,**kwargs,source_length=s['length'],noise=noise,candidate_chunk=10));repeat=export_full(model,**kwargs,source_length=s['length'],noise=noise,candidate_chunk=3);chunk_error=float((pred-repeat).abs().max());assert chunk_error<1e-5
                checks=check_prediction(pred,s['length']);checks['candidate_consistency_max_error']=chunk_error
                with dest.open('xb') as f:np.save(f,pred.numpy())
                save_cached(sidecar,item_identity,dict(path=str(dest),sha256=sha256_file(dest),checks=checks))
            preds.append(pred);speakers.append(streams['speaker_emotion']);exports.append(dict(path=str(dest),sha256=sha256_file(dest),checks=checks));print('EXPORT',label,i+1,flush=True)
        values=compute_metrics(speakers,preds,targets,threads=4,metric_names=manifest['metrics']);raw=raw_metrics(preds,targets,speakers)
        assert abs(np.mean([r['FRC'] for r in raw])-values['FRC'])<1e-6
        single=dict(values);single['FRC']=float(np.mean([r['single_paired_FRC'] for r in raw]));single.pop('FRC_time',None)
        result=dict(seed=point['seed'],arm=point['arm'],lambda_group=point['lambda_group'],multi_target=values,single_paired=single,per_input=raw,exports=exports,seconds=time.perf_counter()-started,peak_memory_bytes=torch.cuda.max_memory_allocated(device),protocol='Development80 full source, frozen official 1+9 selection/Processor/metrics; not hidden test',FRD_executed=False)
        save_cached(path,identity,result);completed.append(dict(path=str(path),sha256=sha256_file(path)));del model;torch.cuda.empty_cache();print('TASK COMPLETE',label,values,flush=True)
    complete(out/('completed_MAM.json' if args.mam_only else 'completed_HiRP.json'),completed)


if __name__=='__main__':main()
