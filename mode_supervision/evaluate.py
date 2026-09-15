"""Full official DEV80, shared K10 output; oracle always a separate diagnostic."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from reaction_flow.sampler import load_checkpoint
from reaction_flow.export import recording_noise
from reaction_flow.data import draw
from reaction_flow.train import configure_flow
from reaction_flow.dynamics import decomposition
from hirp.phase24 import normalization,verify_files,LEGACY
from hirp.phase15_audit import canonical_hash
from hirp.phase23_cache import cached,save_cached
from hirp.task_phase24 import raw_metrics,check_prediction
from semantic_supervision.bert_experiments.evaluate import compact,summaries
from .prepare import OUT,PARENT,sha
from .models import PlanPrior,PlanInjection
from .plans import recording
EVAL=Path('runs/phase24/evaluation_v1')

def main(arm,step,kind='predicted'):
    assert step in (1000,3000,6000) and (kind=='predicted' or arm=='M1-mode')
    configure_flow();torch.set_num_threads(4)
    path=OUT/'training'/arm/'checkpoints'/f'step_{step:06d}.pt';s=torch.load(path,map_location='cpu',weights_only=True)
    assert s['step']==step and s['protocol_sha256']==sha(OUT/'PROTOCOL.json') and s['stats_sha256']==sha(OUT/'plan_stats.json')
    base,_=load_checkpoint(PARENT,'cuda');base.load_state_dict(s['base']);base.double().eval()
    prior=PlanPrior().cuda().double().eval();prior.load_state_dict(s['prior']);inject=PlanInjection().cuda().double().eval();inject.load_state_dict(s['inject'])
    stats=json.loads((OUT/'plan_stats.json').read_text());mu=torch.tensor(stats['mean'],device='cuda',dtype=torch.float64);scale=torch.tensor(stats['std'],device='cuda',dtype=torch.float64)
    manifest=json.loads((EVAL/'multitarget_development_manifest.json').read_text());normalization(manifest)
    tp=json.loads((EVAL/'processed_targets/completed.json').read_text());assert canonical_hash(tp['result'])==tp['result_sha256'];verify_files(tp['result']['files'])
    dep=json.loads((EVAL/'dependency_snapshot.json').read_text());verify_files(dep['legacy'])
    # PTS only, source video measurements already verified in prior experiment; no semantic labels.
    clocks_path=Path('runs/reaction_flow/bert_semantic_v1/dev_sources.json');clocks=json.loads(clocks_path.read_text())
    label=f'seed123_{arm}_step{14000+step}'+('' if kind=='predicted' else '_'+kind)
    dest=OUT/'exports'/label;dest.mkdir(parents=True,exist_ok=True);task=OUT/'task_multitarget'/(label+'.json');task.parent.mkdir(exist_ok=True)
    identity=dict(version='observable-mode-v1',checkpoint_sha256=sha(path),protocol_sha256=sha(OUT/'PROTOCOL.json'),plan_stats_sha256=sha(OUT/'plan_stats.json'),PTS_sha256=sha(clocks_path),target_manifest_sha256=sha(EVAL/'multitarget_development_manifest.json'),processed_index_sha256=sha(EVAL/'processed_targets/completed.json'),legacy=dep['legacy'],source_only=kind!='oracle',diagnostic_only=kind!='predicted',arm=arm,step=step,kind=kind,K=10,precision='FP64 Euler16 -> FP32 official metrics')
    if cached(task,identity) is not None:return
    mean=torch.from_numpy(np.load('external/FaceVerse/mean_face.npy')).float();std=torch.from_numpy(np.load('external/FaceVerse/std_face.npy')).float().clamp_min(1e-8)
    preds=[];targets=[];speakers=[];exports=[];extra=[];tick=time.time()
    for i,source in enumerate(manifest['sources']):
        verify_files(list(source['files'].values()));name=source['clip_id'];pts=clocks[name]['frame_pts'];n=source['length'];assert len(pts)==n
        streams={k:torch.from_numpy(np.load(v['path'])).float() for k,v in source['files'].items()};face=streams['speaker_3dmm'];streams['speaker_3dmm']=((face[:,0] if face.ndim==3 else face)-mean)/std
        target=torch.from_numpy(np.load(tp['result']['files'][i]['path']))
        side=dest/f'{i:03d}.json';ident=dict(evaluation_sha256=canonical_hash(identity),source=i);item=cached(side,ident)
        if item is None:
            begin=time.time();hs=[];masks=[];positions=[];plan_masks=[]
            with torch.no_grad():
                for start in range(0,n,750):
                    nn=min(750,n-start);inputs=[F.pad(streams[k][start:start+nn],(0,0,0,750-nn))[None].cuda().double() for k in ('speaker_audio','speaker_emotion','speaker_3dmm')]
                    h,m=base.condition(*inputs,torch.tensor([nn],device='cuda'));hs.append(h);masks.append(m)
                    positions.append([pts[start]/30,(pts[start+nn-1]-pts[start])/30]);plan_masks.append((torch.arange(4,device='cuda')[:,None].expand(4,24)<min(4,nn)).flatten())
                context=torch.stack([(h*m[...,None]).sum(1)[0]/m.sum() for h,m in zip(hs,masks)])[None]
                pm=torch.stack(plan_masks)[None];pos=torch.tensor([positions],device='cuda',dtype=torch.float64);L=len(hs)
                if arm=='M1-mode':
                    if kind=='oracle':
                        # Fixed target order, no quality-based choice; diagnostic only.
                        chosen=[target[k%len(target),:n].cuda().double() for k in range(10)]
                        pairs=[recording(base.transform(y),pts) for y in chosen];plans=torch.stack([(a-mu)/scale for a,_ in pairs]);pm=torch.stack([m for _,m in pairs])
                    else:
                        z=torch.stack([draw((1,L,96),8500123,i,f'plan_{k}') for k in range(10)]).reshape(10,L,96).cuda().double()
                        plans=prior.sample(z,context.expand(10,-1,-1),pos.expand(10,-1,-1),pm.expand(10,-1,-1))
                        if kind=='fixed_plan':plans=plans[:1].expand_as(plans)
                    pm=pm.expand(10,-1,-1)
                noise=recording_noise(base.config,name,n).cuda().double()
                if kind=='fixed_noise':noise=noise[:1].expand_as(noise)
                outputs=[]
                for bi,start in enumerate(range(0,n,750)):
                    nn=min(750,n-start);h,m=hs[bi],masks[bi];z=F.pad(noise[:,start:start+nn],(0,0,0,750-nn))
                    if arm=='M1-mode':
                        h=inject(h.expand(10,-1,-1),plans,pm,torch.full((10,),bi,device='cuda',dtype=torch.long));p=base.rollout(h,m.expand(10,-1),z[:,None],16,torch.full((10,),start,device='cuda'))[:,0,:nn]
                    else:p=base.rollout(h,m,z[None],16,torch.tensor([start],device='cuda'))[0,:,:nn]
                    outputs.append(p.float().cpu())
                prediction=torch.cat(outputs,1)
            checks=check_prediction(prediction,n);file=dest/f'{i:03d}.npy';tmp=file.with_suffix('.tmp')
            with tmp.open('wb') as f:np.save(f,prediction.numpy())
            tmp.replace(file);item=dict(path=str(file),sha256=sha(file),checks=checks,generation_seconds=time.time()-begin);save_cached(side,ident,item)
        else:verify_files([item]);prediction=torch.from_numpy(np.load(item['path']))
        preds.append(prediction);targets.append(target);speakers.append(streams['speaker_emotion']);exports.append(item)
        groups={}
        for group,a,z in [('AU',0,15),('VA',15,17),('expression',17,25)]:
            total,parts=decomposition(prediction[...,a:z].double());groups[group]=dict(total=float(total),**{k:float(v) for k,v in parts.items()})
        boundaries=list(range(750,n,750));extra.append(dict(index=i,groups=groups,boundary_mean_abs=float(torch.stack([(prediction[:,b]-prediction[:,b-1]).abs().mean() for b in boundaries]).mean()) if boundaries else None))
        print(label,i+1,'/80',flush=True)
    sys.path.insert(0,str(LEGACY));from framework.utils.compute_metrics import compute_metrics
    metrics=compute_metrics(speakers,preds,targets,threads=4,metric_names=manifest['metrics']);raw=raw_metrics(preds,targets,speakers);rows=compact(raw,preds,manifest['sources'])
    for r,x in zip(raw,extra):
        c=np.asarray(r['CCC_candidate_target']).max(1);x['candidate_best_target_CCC_quantiles']={str(q):float(np.quantile(c,q)) for q in (.0,.1,.25,.5)}
    save_cached(task,identity,dict(model=label,metrics=metrics,raw=raw,exports=exports,per_source=rows,subgroups=summaries(rows,{'all':list(range(80))}),diagnostics=extra,diagnostic_only=kind!='predicted',source_only=kind!='oracle',seconds=time.time()-tick,peak_gpu_bytes=torch.cuda.max_memory_allocated()))
    print(metrics,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--step',type=int,required=True);p.add_argument('--kind',default='predicted',choices=['predicted','oracle','fixed_plan','fixed_noise']);a=p.parse_args();main(a.arm,a.step,a.kind)
