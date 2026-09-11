import json,time
from pathlib import Path
import numpy as np
import torch
from .config import FlowConfig,ROOT,PARENT,PARENT_SHA,DATA_MANIFEST
from .reaction_transform import ReactionTransform
from .data import FlowData
from hirp.train_phase25 import TrainConfig,schedule
from hirp.phase15_audit import sha256_file,canonical_hash
from mam_target.losses import ccc25

def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)

def main():
    torch.set_num_threads(2);cfg=FlowConfig();assert sha256_file(PARENT)==PARENT_SHA
    old=json.loads(DATA_MANIFEST.read_text())
    for row in old['data_files']+old['normalization']:
        if sha256_file(row['path'])!=row['sha256']:raise ValueError('training provenance changed: '+row['path'])
    data=FlowData(cfg);pool=data.inner.pool
    records,_,_=schedule(pool,TrainConfig(max_steps=cfg.steps+cfg.adaptation_steps,T=750),32)
    target_rng=torch.Generator().manual_seed(cfg.target_seed)
    for r in records:r['target_slots']=torch.randint(4,(cfg.B,),generator=target_rng).tolist()
    write(ROOT/'schedule.json',dict(records=records,choice_law='iid uniform over four weighted slots; independent target generator; no model-dependent selection'))
    transform=ReactionTransform();mean=torch.zeros(24,dtype=torch.double);second=mean.clone();count=0;quantiles=[];crops=[];duplicates=0;checks=dict(frames=0,AU_outside=0,VA_outside=0,expression_negative=0,expression_sum_off=0);errors=[]
    started=time.time()
    for record in records[:cfg.statistics_batches]:
        batch,targets,lengths,ids,y,ns,slots=data.batch(record)
        for i,n0 in enumerate(ns):
            n=int(n0);raw=y[i,:n].double();checks['frames']+=n;checks['AU_outside']+=int(((raw[:,:15]<0)|(raw[:,:15]>1)).sum());checks['VA_outside']+=int(((raw[:,15:17]<-1)|(raw[:,15:17]>1)).sum());checks['expression_negative']+=int((raw[:,17:]<0).sum());checks['expression_sum_off']+=int(((raw[:,17:].sum(-1)-1).abs()>1e-5).sum())
            if checks['AU_outside'] or checks['VA_outside'] or checks['expression_negative'] or checks['expression_sum_off']:
                write(ROOT/'invalid_domain.json',checks);raise ValueError('TRAIN attributes violate declared domains; explicit new policy required')
            u=transform.coordinates(raw);assert torch.isfinite(u).all();mean+=u.mean(0);second+=u.square().mean(0);count+=1;quantiles.append(u[::max(1,n//32)])
            if record['step']<=16:
                reconstructed=transform.inverse(u);errors.append(dict(step=record['step'],source=i,rmse=float((reconstructed-raw).square().mean().sqrt()),max_error=float((reconstructed-raw).abs().max()),CCC_roundtrip=float(ccc25(reconstructed,raw)),CCC_self=float(ccc25(raw,raw))))
            duplicates+=4-len(set(ids[i]));crops.append(dict(step=record['step'],source_index=record['source_indices'][i],source_crop_start=int(batch['crop_start'][i]),source_length=int(batch['source_lengths'][i]),selected_slot=int(slots[i]),target_id=ids[i][int(slots[i])],all_target_slots=ids[i],selected_length=n,weight=1/cfg.statistics_batches/cfg.B))
        if record['step']%64==0:print('STATS',record['step'],time.time()-started,flush=True)
    mean/=count;std=(second/count-mean.square()).clamp_min(0).sqrt();q=torch.cat(quantiles)
    stats=dict(mean=mean.tolist(),std=std.clamp_min(cfg.std_floor).tolist(),raw_std=std.tolist(),std_floor=cfg.std_floor,floored_dimensions=(std<cfg.std_floor).nonzero().flatten().tolist(),eps=cfg.eps,Q=transform.Q.tolist(),fit='TRAIN first1024 schedule batches, one uniformly selected endpoint/source; equal occurrence weights, normalized over own valid frames',domain_counts=checks,duplicate_slot_fraction=duplicates/(count*4),crops=crops,transformed_quantile_levels=[0,.01,.5,.99,1],transformed_quantiles=torch.quantile(q,torch.tensor([0,.01,.5,.99,1],dtype=q.dtype),dim=0).tolist(),quantile_sampling='up to approximately33 evenly strided valid frames per endpoint',roundtrip_TRAIN16=errors,seconds=time.time()-started)
    write(ROOT/'coordinate_stats.json',stats)
    write(ROOT/'manifest.json',dict(config=cfg.dictionary(),parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,parent_reused_modules=['stems','encoder'],parent_training_steps=6000,parent_valid_frames=17135366,parent_origin='project R2 supervised training; no MAM weights/labels',data_manifest=str(DATA_MANIFEST),data_manifest_sha256=sha256_file(DATA_MANIFEST),split_hash=old['split_hash'],normalization=old['normalization'],artifact_hashes={n:sha256_file(ROOT/n) for n in ['schedule.json','coordinate_stats.json']},target_law='TRAIN paired+3 session alternatives; iid uniform slots; duplicate slots retain weights; weak not causal probability labels',noise_law='independent keyed normal full temporal state, separate streams for target-choice/noise/tau/rollout',statistics_batches=1024,physical_gpu=1,completed_training=False,prior_experiment_complete=json.loads(Path('runs/mam_target/staged_v2/queue_finished.json').read_text()),source_snapshot={str(p):sha256_file(p) for p in Path('reaction_flow').rglob('*.py')}))
    print('PREPARED',count,checks,flush=True)
if __name__=='__main__':main()
