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
from reward_policy.generator import Generator
from reward_policy.policy import Policy
from reward_policy.common import PARENT,read,sha
from .audit import OUT
EVAL=Path('runs/phase24/evaluation_v1')

def main(arm):
    configure_flow();torch.set_num_threads(4);gen=Generator();policy=Policy().cuda().double().eval();path=OUT/'training'/arm/'checkpoints/step_100.pt';saved=torch.load(path,map_location='cpu',weights_only=True)
    assert saved['step']==100 and saved['arm']==arm and saved['protocol_sha256']==sha(OUT/'PROTOCOL.json');policy.load_state_dict(saved['policy'])
    for p,h in read(OUT/'CODE_IDENTITY.json').items():assert sha(p)==h
    sources=Sources('val');manifest=read(EVAL/'multitarget_development_manifest.json');normalization(manifest);tp=read(EVAL/'processed_targets/completed.json');assert canonical_hash(tp['result'])==tp['result_sha256'];verify_files(tp['result']['files']);dep=read(EVAL/'dependency_snapshot.json');verify_files(dep['legacy'])
    label=arm+'_step100';dest=OUT/'exports'/label;dest.mkdir(parents=True,exist_ok=True);task=OUT/'task_multitarget'/(label+'.json');task.parent.mkdir(exist_ok=True)
    identity=dict(version='explicit-noise-policy-eval-signal-v2',policy_sha256=sha(path),generator_sha256=sha(PARENT),protocol_sha256=sha(OUT/'PROTOCOL.json'),source_manifest_sha256=sha(EVAL/'multitarget_development_manifest.json'),processed_index_sha256=sha(EVAL/'processed_targets/completed.json'),K=10,source_only=True,target_selection=False,legacy=dep['legacy'],precision='FP64 Euler16 -> FP32 official metrics')
    if cached(task,identity) is not None:return
    mean=torch.from_numpy(np.load('external/FaceVerse/mean_face.npy')).float();std=torch.from_numpy(np.load('external/FaceVerse/std_face.npy')).float().clamp_min(1e-8)
    preds=[];targets=[];speakers=[];exports=[];extra=[];tick=time.time()
    for i,source in enumerate(manifest['sources']):
        verify_files(list(source['files'].values()));name=source['clip_id'];pts=sources.rows[name]['frame_pts'];n=source['length'];assert len(pts)==n
        streams={k:torch.from_numpy(np.load(v['path'])).float() for k,v in source['files'].items()};face=streams['speaker_3dmm'];streams['speaker_3dmm']=((face[:,0] if face.ndim==3 else face)-mean)/std
        side=dest/f'{i:03d}.json';ident=dict(evaluation_sha256=canonical_hash(identity),source=i);item=cached(side,ident)
        if item is None:
            begin=time.time();noise=recording_noise(gen.model.base.config,name,n).cuda().double();events=[sources.crop(name,s,min(750,n-s))[0] for s in range(0,n,750)]
            prediction=gen.sample([streams[k] for k in ('speaker_audio','speaker_emotion','speaker_3dmm')],pts,events,noise,policy).float().cpu();checks=check_prediction(prediction,n)
            file=dest/f'{i:03d}.npy';tmp=file.with_suffix('.tmp')
            with tmp.open('wb') as f:np.save(f,prediction.numpy())
            tmp.replace(file);item=dict(path=str(file),sha256=sha(file),checks=checks,generation_seconds=time.time()-begin);save_cached(side,ident,item)
        else:verify_files([item]);prediction=torch.from_numpy(np.load(item['path']))
        preds.append(prediction);targets.append(torch.from_numpy(np.load(tp['result']['files'][i]['path'])));speakers.append(streams['speaker_emotion']);exports.append(item)
        groups={}
        for group,a,b in [('AU',0,15),('VA',15,17),('expression',17,25)]:
            total,parts=decomposition(prediction[...,a:b].double());groups[group]=dict(total=float(total),**{k:float(v) for k,v in parts.items()})
        extra.append(dict(index=i,groups=groups));print(label,i+1,'/80',flush=True)
    sys.path.insert(0,str(LEGACY));from framework.utils.compute_metrics import compute_metrics
    metrics=compute_metrics(speakers,preds,targets,threads=4,metric_names=manifest['metrics']);raw=raw_metrics(preds,targets,speakers);rows=compact(raw,preds,manifest['sources'])
    for r,x in zip(raw,extra):
        c=np.asarray(r['CCC_candidate_target']).max(1);x['candidate_quality_quantiles']={str(q):float(np.quantile(c,q)) for q in (0,.1,.25,.5)}
    save_cached(task,identity,dict(model=label,metrics=metrics,raw=raw,exports=exports,per_source=rows,subgroups=summaries(rows,dict(all=list(range(80)))),diagnostics=extra,seconds=time.time()-tick,source_only=True))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);a=p.parse_args();main(a.arm)
