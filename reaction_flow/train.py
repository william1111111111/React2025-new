"""14000-record finite runner: A=12000, then matched 2000-update adaptations."""
import argparse,json,time,random
from pathlib import Path
import numpy as np
import torch
from .config import FlowConfig,ROOT,PARENT,PARENT_SHA
from .sampler import ReactionFlow,load_checkpoint,metadata
from .data import resources,draw
from .flow_path import flow_path,flow_loss
from hirp.config import HiRPConfig
from hirp.paired_data import paired_model_inputs
from hirp.train_phase25 import rng_state,restore_rng
from hirp.train_phase21 import configure
from hirp.phase15_audit import sha256_file,canonical_hash
from mam_refine.train import tree_hash

def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)

def configure_flow():
    configure();torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)

def initial_model(cfg,stats,device):
    assert sha256_file(PARENT)==PARENT_SHA
    saved=torch.load(PARENT,map_location='cpu',weights_only=True)
    torch.manual_seed(cfg.seed)
    model=ReactionFlow(cfg,HiRPConfig(**saved['config']),stats['mean'],stats['std'])
    model.condition.from_parent(saved['model']);return model.to(device)

def fm_batch(model,batch,selected,lengths,record,cfg,device,h=None,source_valid=None):
    if h is None:h,source_valid=model.condition(**paired_model_inputs(batch))
    valid=torch.arange(cfg.T,device=device)[None]<lengths[:,None]
    with torch.no_grad():target=model.transform(selected)
    noise=draw(target.shape,cfg.noise_seed,record['step'],'FM_noise').to(device)
    tau=draw((cfg.B,),cfg.tau_seed,record['step'],'FM_tau',False).to(device)
    u,v=flow_path(target,noise,tau,valid)
    predicted=model.velocity(u,tau,h,valid,source_valid,batch['crop_start'])
    return flow_loss(predicted,v,valid),h,source_valid

def train(phase='A',stop=None,resume=None,parent=None,label=None,device='cuda:0'):
    cfg=FlowConfig();manifest,records,data=resources(cfg,device);stats=json.loads((ROOT/'coordinate_stats.json').read_text())
    offset=0 if phase=='A' else cfg.steps;budget=cfg.steps if phase=='A' else cfg.adaptation_steps;stop=stop or budget
    assert 0<stop<=budget and phase in ('A','F-cont','F-task')
    if phase!='A' and not parent:raise ValueError('adaptation requires explicit A12000 parent')
    if resume:model,saved=load_checkpoint(resume,device)
    elif parent:
        model,saved=load_checkpoint(parent,device);assert saved['phase']=='A' and saved['phase_step']==12000 and saved['completed']
    else:model=initial_model(cfg,stats,device);saved=None
    optimizer=torch.optim.AdamW(model.velocity.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    if saved:optimizer.load_state_dict(saved['optimizer'])
    if resume:
        assert saved['phase']==phase and saved['manifest_sha256']==sha256_file(ROOT/'manifest.json')
        u=saved['phase_step'];rows=saved['rows'];assert u==len(rows) and saved['schedule_position']==offset+u
    else:u=0;rows=[]
    cal=None
    if phase=='F-task':
        cal=json.loads((ROOT/'task_calibration.json').read_text());assert cal['parent_sha256']==sha256_file(parent)
    out=ROOT/(label or phase);out.mkdir(exist_ok=True);attempt=out/f'attempt_{len(list(out.glob("attempt_*"))):03d}';attempt.mkdir();(attempt/'checkpoints').mkdir()
    fixed_hash=tree_hash(model.condition.state_dict());initial_hash=tree_hash(model.velocity.state_dict())
    write(attempt/'initial.json',dict(phase=phase,parent_checkpoint=str(parent) if parent else str(PARENT),parent_sha256=sha256_file(parent) if parent else PARENT_SHA,condition_hash=fixed_hash,velocity_hash=initial_hash,optimizer_hash=tree_hash(optimizer.state_dict()),start_phase_step=u,physical_gpu=1))
    random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed);torch.cuda.manual_seed_all(cfg.seed)
    if saved:restore_rng(saved['rng'])
    probe_batch=data.batch(records[0])[0]
    probe_inputs={k:v[:1] for k,v in paired_model_inputs(probe_batch).items()}
    probe_h,probe_sm=model.condition(**probe_inputs);probe_offset=probe_batch['crop_start'][:1];probe_n=int(probe_batch['source_lengths'][0])
    model.train();started=time.time();torch.cuda.reset_peak_memory_stats(device);checkpoint_steps=cfg.checkpoints if phase=='A' else (2000,)
    def save():
        assert tree_hash(model.condition.state_dict())==fixed_hash
        payload=dict(**metadata(model),model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng_state(),phase=phase,phase_step=u,global_step=offset+u,schedule_position=offset+u,requested_phase_steps=budget,actual_optimizer_steps=offset+u,rows=rows,completed=u==budget,manifest_sha256=sha256_file(ROOT/'manifest.json'),coordinate_stats_sha256=manifest['artifact_hashes']['coordinate_stats.json'],parent_sha256=sha256_file(parent) if parent else PARENT_SHA,condition_hash=fixed_hash,split_hash=manifest['split_hash'],task_calibration=cal,training_seconds=time.time()-started)
        path=attempt/'checkpoints'/f'step_{offset+u:06d}.pt';assert not path.exists();tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.rename(path)
        write(path.with_suffix('.json'),dict(path=str(path),sha256=sha256_file(path),phase=phase,global_step=offset+u,phase_step=u))
    try:
        with (attempt/'training.jsonl').open('x') as log:
            for row in rows:log.write(json.dumps(row)+'\n')
            while u<stop:
                step_start=time.perf_counter();record=records[offset+u];assert record['step']==offset+u+1
                batch,targets,lengths,ids,selected,ns,slots=data.batch(record);optimizer.zero_grad(set_to_none=True)
                fm,h,sm=fm_batch(model,batch,selected,ns,record,cfg,device);total=fm;task=None
                if phase=='F-task':
                    from .finetune import task_loss
                    noise=draw((cfg.B,4,cfg.T,24),cfg.rollout_seed,record['step'],'task_rollout').to(device)
                    p=model.rollout(h,sm,noise,16,batch['crop_start'],gradient_checkpointing=True)
                    task=task_loss(p,batch,targets,lengths);total=total+cal['lambda_task']*task
                total.backward();assert torch.isfinite(total) and all(torch.isfinite(p.grad).all() for p in model.velocity.parameters() if p.grad is not None)
                norm=float(sum(p.grad.square().sum() for p in model.velocity.parameters() if p.grad is not None).sqrt());optimizer.step();u+=1
                row=dict(phase_step=u,global_step=offset+u,FM=float(fm),loss=float(total),task=None if task is None else float(task),gradient_norm=norm,source_indices=record['source_indices'],source_crop_start=batch['crop_start'].tolist(),source_lengths=batch['source_lengths'].tolist(),selected_lengths=ns.tolist(),slots=slots.tolist(),target_slots=ids,selected_target_ids=[ids[i][int(slots[i])] for i in range(cfg.B)],duplicate_slots=sum(4-len(set(x)) for x in ids),noise_key=[cfg.noise_seed,record['step'],'FM_noise'],step_seconds=time.perf_counter()-step_start)
                rows.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                if u==1 or u%100==0:print(phase,u,'FM',float(fm),'grad',norm,'elapsed',time.time()-started,flush=True)
                if u%500==0:
                    with torch.no_grad():
                        # Fixed TRAIN probe, fresh keyed temporal noise; does not consume training RNG.
                        probe_noise=draw((1,4,cfg.T,24),cfg.rollout_seed,0,'TRAIN_probe').to(device)
                        y=model.rollout(probe_h,probe_sm,probe_noise,16,probe_offset);y=y[0,:,:probe_n]
                        diag={name:dict(speed=float(y[...,a:b].diff(dim=1).abs().mean()),acceleration=float(y[...,a:b].diff(dim=1).diff(dim=1).abs().mean()),candidate_std=float(y[...,a:b].std(0).mean())) for name,a,b in [('AU',0,15),('VA',15,17),('expression',17,25)]}
                    write(attempt/f'diagnostic_{u:06d}.json',diag)
                if u in (*checkpoint_steps,stop):save()
        assert u==stop==len(rows)
        assert {int(s['step']) for s in optimizer.state.values()}=={offset+u}
        counts=np.bincount([j for r in rows for j in r['slots']],minlength=4).tolist()
        write(attempt/'status.json',dict(completed=u==budget,requested_phase_steps=budget,actual_phase_steps=u,global_step=offset+u,rows=len(rows),seconds=time.time()-started,peak_memory_bytes=torch.cuda.max_memory_allocated(device),new_valid_endpoint_frames=sum(sum(r['selected_lengths']) for r in rows),source_frame_exposures=sum(sum(r['source_lengths']) for r in rows),unique_sources=len({i for r in rows for i in r['source_indices']}),unique_endpoints=len({i for r in rows for i in r['selected_target_ids']}),target_slot_counts=counts,duplicate_slot_fraction=sum(r['duplicate_slots'] for r in rows)/(4*cfg.B*len(rows)),velocity_parameters=sum(p.numel() for p in model.velocity.parameters()),condition_parameters=sum(p.numel() for p in model.condition.parameters()),condition_unchanged=True))
    except BaseException as e:write(attempt/'failure.json',dict(error=repr(e),phase_step=u,completed=False));raise
    return attempt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['A','F-cont','F-task'],default='A');p.add_argument('--stop',type=int);p.add_argument('--resume');p.add_argument('--parent');p.add_argument('--label');a=p.parse_args();configure_flow();train(a.phase,a.stop,a.resume,a.parent,a.label)
