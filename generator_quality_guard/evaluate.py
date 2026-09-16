"""Formal full DEV80: speaker-only policy+fixed generator; targets for metrics only."""
import argparse,time,sys
from pathlib import Path
import numpy as np
import torch
from hirp.phase24 import normalization,verify_files,LEGACY
from hirp.phase15_audit import canonical_hash
from hirp.phase23_cache import cached,save_cached
from hirp.task_phase24 import raw_metrics,check_prediction
from reaction_flow.export import recording_noise
from reaction_flow.train import configure_flow
from reaction_flow.dynamics import decomposition
from semantic_supervision.bert_experiments.common import Sources
from semantic_supervision.bert_experiments.evaluate import compact,summaries
from generator_reward_nft.public import Generator
from reward_policy.common import PARENT,read,sha
from .common import OUT
EVAL=Path('runs/phase24/evaluation_v1')

def main(arm,variant):
    configure_flow();torch.set_num_threads(4);gen=Generator();completion=read(OUT/'training'/arm/'completion.json');path=Path(completion['last_proposal' if variant=='last-proposal' else 'last_accepted']);saved=torch.load(path,map_location='cpu',weights_only=True)
    assert saved['arm']==arm and saved['protocol_sha256']==sha(OUT/'PROTOCOL.json')
    gen.model.base.velocity.load_state_dict(saved['student']);gen.model.eval().requires_grad_(False)
    for p,h in read(OUT/'CODE_IDENTITY.json').items():assert sha(p)==h
    sources=Sources('val');manifest=read(EVAL/'multitarget_development_manifest.json');normalization(manifest);tp=read(EVAL/'processed_targets/completed.json');assert canonical_hash(tp['result'])==tp['result_sha256'];verify_files(tp['result']['files']);dep=read(EVAL/'dependency_snapshot.json');verify_files(dep['legacy'])
    label=arm+'_'+variant;dest=OUT/'exports'/label;dest.mkdir(parents=True,exist_ok=True);task=OUT/'task_multitarget'/(label+'.json');task.parent.mkdir(exist_ok=True)
    identity=dict(version='quality-guarded-native-generator-v1',student_checkpoint_sha256=sha(path),parent_sha256=sha(PARENT),protocol_sha256=sha(OUT/'PROTOCOL.json'),source_manifest_sha256=sha(EVAL/'multitarget_development_manifest.json'),processed_index_sha256=sha(EVAL/'processed_targets/completed.json'),K=10,source_only=True,target_selection=False,legacy=dep['legacy'],precision='FP64 Euler16 -> FP32 official metrics')
    if cached(task,identity) is not None:return
    if variant=='last-accepted' and saved['accepted_steps']==0:
        from .common import digest
        parent=Generator();assert digest(parent.model.base.velocity)==digest(gen.model.base.velocity)
        oldpath=Path('runs/reaction_flow/bert_semantic_v1/task_multitarget/seed123_P2-bert_step15000.json');old=read(oldpath);assert old['eval_identity']['checkpoint_sha256']==sha(PARENT)
        verify_files(old['result']['exports']);result=dict(old['result']);result.update(model=label,reused_parent_evaluation=str(oldpath),reused_parent_sha256=sha(oldpath),accepted_steps=0)
        save_cached(task,identity,result);return

    mean=torch.from_numpy(np.load('external/FaceVerse/mean_face.npy')).float();std=torch.from_numpy(np.load('external/FaceVerse/std_face.npy')).float().clamp_min(1e-8)
    preds=[];targets=[];speakers=[];exports=[];extra=[];tick=time.time()
    for i,source in enumerate(manifest['sources']):
        verify_files(list(source['files'].values()));name=source['clip_id'];pts=sources.rows[name]['frame_pts'];n=source['length'];assert len(pts)==n
        streams={k:torch.from_numpy(np.load(v['path'])).float() for k,v in source['files'].items()};face=streams['speaker_3dmm'];streams['speaker_3dmm']=((face[:,0] if face.ndim==3 else face)-mean)/std
        side=dest/f'{i:03d}.json';ident=dict(evaluation_sha256=canonical_hash(identity),source=i);item=cached(side,ident)
        if item is None:
            begin=time.time();noise=recording_noise(gen.model.base.config,name,n).cuda().double();events=[sources.crop(name,s,min(750,n-s))[0] for s in range(0,n,750)]
            prediction=gen.sample([streams[k] for k in ('speaker_audio','speaker_emotion','speaker_3dmm')],pts,events,noise).float().cpu();checks=check_prediction(prediction,n)
            file=dest/f'{i:03d}.npy';tmp=file.with_suffix('.tmp')
            with tmp.open('wb') as f:np.save(f,prediction.numpy())
            tmp.replace(file);item=dict(path=str(file),sha256=sha(file),checks=checks,generation_seconds=time.time()-begin);save_cached(side,ident,item)
        else:verify_files([item]);prediction=torch.from_numpy(np.load(item['path']))
        preds.append(prediction);targets.append(torch.from_numpy(np.load(tp['result']['files'][i]['path'])));speakers.append(streams['speaker_emotion']);exports.append(item)
        groups={}
        for group,a,b in [('AU',0,15),('VA',15,17),('expression',17,25)]:
            total,parts=decomposition(prediction[...,a:b].double());groups[group]=dict(total=float(total),**{k:float(v) for k,v in parts.items()},native_speed_quantiles={str(q):float(torch.quantile(prediction[...,a:b].diff(dim=1).abs().flatten(),q)) for q in (.95,.99)},native_acceleration_quantiles={str(q):float(torch.quantile(prediction[...,a:b].diff(n=2,dim=1).abs().flatten(),q)) for q in (.95,.99)})
        extra.append(dict(index=i,groups=groups));print(label,i+1,'/80',flush=True)
    sys.path.insert(0,str(LEGACY));from framework.utils.compute_metrics import compute_metrics
    metrics=compute_metrics(speakers,preds,targets,threads=4,metric_names=manifest['metrics']);raw=raw_metrics(preds,targets,speakers);rows=compact(raw,preds,manifest['sources'])
    for r,x in zip(raw,extra):
        c=np.asarray(r['CCC_candidate_target']).max(1);x['candidate_quality_quantiles']={str(q):float(np.quantile(c,q)) for q in (0,.1,.25,.5)}
    save_cached(task,identity,dict(model=label,metrics=metrics,raw=raw,exports=exports,per_source=rows,subgroups=summaries(rows,dict(all=list(range(80)))),diagnostics=extra,seconds=time.time()-tick,source_only=True))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--variant',choices=['last-proposal','last-accepted'],required=True);a=p.parse_args();main(a.arm,a.variant)
