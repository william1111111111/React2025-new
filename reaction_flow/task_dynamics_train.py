"""Matched T0/T1 warm-start adaptation. Independent outputs; historical runner unchanged."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from .config import FlowConfig,ROOT as AROOT
from .data import resources,draw
from .flow_path import flow_path,flow_loss
from .sampler import load_checkpoint,metadata
from .train import configure_flow,write
from .finetune import WEIGHTS
from mam_target.losses import pair_costs,softmin,ccc25
from .dynamics import GROUPS,dynamic_loss,decomposition
from hirp.paired_data import paired_model_inputs
from hirp.phase15_audit import sha256_file
from hirp.train_phase25 import rng_state,restore_rng
from mam_refine.train import tree_hash
ROOT=Path('runs/reaction_flow/task_dynamics_v1')
PARENT=AROOT/'A/attempt_000/checkpoints/step_012000.pt'
def task_parts(pred,batch):
    c,d=pair_costs(pred,batch['_targets'],batch['source_lengths'],batch['_target_lengths'],32)
    cost=c+WEIGHTS['b']*d;valid=softmin(cost,2).mean();cover=softmin(cost,1).mean();paired=[]
    for i in range(len(pred)):
        n=int(batch['pair_lengths'][i]);p=pred[i,:,:n];y=batch['paired_target'][i,:n].expand_as(p)
        v=1-ccc25(p,y)+(p-y).abs().mean((-1,-2))
        if n>1:v=v+.1*((p[:,1:]-p[:,:-1])-(y[:,1:]-y[:,:-1])).abs().mean((-1,-2))
        paired.append(v.min())
    paired=torch.stack(paired).mean()
    return valid+.25*cover+.25*paired,dict(valid=float(valid.detach()),cover=float(cover.detach()),paired=float(paired.detach()),ccc_cost=float(c.detach().mean()),sdtw_cost=float(d.detach().mean()))

def losses(model,batch,selected,lengths,record,cfg,scales):
    h,sm=model.condition(**paired_model_inputs(batch));valid=torch.arange(cfg.T,device=h.device)[None]<lengths[:,None]
    target=model.transform(selected).detach();noise=draw(target.shape,cfg.noise_seed,record['step'],'FM_noise').to(h.device)
    tau=draw((cfg.B,),cfg.tau_seed,record['step'],'FM_tau',False).to(h.device)
    state,velocity=flow_path(target,noise,tau,valid);v=model.velocity(state,tau,h,valid,sm,batch['crop_start'])
    fm=flow_loss(v,velocity,valid)
    estimate=model.transform.inverse(state+(1-tau[:,None,None])*v)
    dyn=dynamic_loss(estimate,selected,lengths,tau,scales)
    z=draw((cfg.B,4,cfg.T,24),cfg.rollout_seed,record['step'],'task_rollout').to(h.device)
    pred=model.rollout(h,sm,z,16,batch['crop_start'],gradient_checkpointing=True)
    task,components=task_parts(pred,batch);batch['_task_components']=components
    return fm,task,dyn,pred,tau

def grad(loss,params,retain=True):
    gs=torch.autograd.grad(loss,params,retain_graph=retain,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(params,gs)])
def summary(g):
    return dict(norm=float(g.norm()),abs_quantiles=torch.quantile(g.abs(),torch.tensor([.5,.9,.99,1.],device=g.device)).tolist())
def get_batch(data,rec):
    b,t,l,ids,y,n,slots=data.batch(rec);b['_targets']=t;b['_target_lengths']=l
    return b,y,n,ids,slots

def calibrate():
    cfg=FlowConfig();manifest,records,data=resources(cfg,'cuda:0');model,meta=load_checkpoint(PARENT,'cuda:0');params=list(model.velocity.parameters())
    scale_values=[[[] for s in range(3)] for g in range(3)]
    for rec in records[12000:12016]:
        b,y,ns,ids,slots=get_batch(data,rec)
        for i,n in enumerate(ns.tolist()):
            for gi,(_,a,z) in enumerate(GROUPS):
                for si,s in enumerate((1,4,16)):
                    if n>s:scale_values[gi][si].append(float(((y[i,s:n,a:z]-y[i,:n-s,a:z])/s).abs().mean()))
    scales=[[max(1e-4,float(np.median([x for x in v if x>0]))) if any(x>0 for x in v) else 1e-4 for v in group] for group in scale_values]
    rows=[];ratios=[];dratios=[]
    for rec in records[12000:12016]:
        b,y,n,ids,slots=get_batch(data,rec);fm,task,dyn,p,tau=losses(model,b,y,n,rec,cfg,torch.tensor(scales,device='cuda'))
        gf=grad(fm,params);gt=grad(task,params);gd=grad(dyn,params)
        assert torch.isfinite(torch.cat((gf,gt,gd))).all() and gf.norm()>0 and gt.norm()>1e-12
        ratios.append(float(gf.norm()/gt.norm()))
        if gd.norm()>1e-12:dratios.append(float(.1*gf.norm()/gd.norm()))
        rows.append(dict(step=rec['step'],FM=summary(gf),task=summary(gt),dyn=summary(gd),FM_task_cosine=float(torch.nn.functional.cosine_similarity(gf,gt,dim=0)),FM_dyn_cosine=float(torch.nn.functional.cosine_similarity(gf,gd,dim=0)),tau=tau.tolist()))
        print('CAL',rec['step'],rows[-1],flush=True)
        del fm,task,dyn,p,gf,gt,gd
    if not dratios:raise RuntimeError('all dynamic gradients zero; do not launch')
    write(ROOT/'calibration.json',dict(parent_sha256=sha256_file(PARENT),lambda_ref=float(np.median(ratios)),lambda_dyn=float(np.median(dratios)),dynamic_scales=scales,scale_floor=1e-4,raw_scale_observations=scale_values,rows=rows,zero_dynamic_batches=16-len(dratios),source='TRAIN continuation records12001..12016; no DEV calibration'))

def train(arm,stop=2000,resume=None,label=None):
    assert arm in ('T0-task','T1-task-dynamics') and 0<stop<=2000
    cfg=FlowConfig();manifest,records,data=resources(cfg,'cuda:0');cal=json.loads((ROOT/'calibration.json').read_text());assert cal['parent_sha256']==sha256_file(PARENT)
    model,saved=load_checkpoint(resume or PARENT,'cuda:0');model.train();params=list(model.velocity.parameters())
    optimizer=torch.optim.AdamW(params,lr=2e-5,weight_decay=.01);optimizer.load_state_dict(saved['optimizer'])
    for group in optimizer.param_groups:group['lr']=2e-5
    restore_rng(saved['rng']);u=saved['phase_step'] if resume else 0;rows=saved['rows'] if resume else []
    if resume:assert saved['phase']==arm and u==len(rows)
    out=ROOT/(label or arm);out.mkdir(exist_ok=True);attempt=out/f'attempt_{len(list(out.glob("attempt_*"))):03d}';attempt.mkdir();(attempt/'checkpoints').mkdir()
    fixed=tree_hash(model.condition.state_dict());write(attempt/'initial.json',dict(parent_sha256=sha256_file(PARENT),velocity_hash=tree_hash(model.velocity.state_dict()),optimizer_hash=tree_hash(optimizer.state_dict()),condition_hash=fixed,LR=2e-5,start=u,optimizer_policy='inherit A AdamW states; both groups set lr2e-5',protocol_sha256=sha256_file(ROOT/'PROTOCOL.json')))
    scales=torch.tensor(cal['dynamic_scales'],device='cuda');started=time.time();torch.cuda.reset_peak_memory_stats()
    probe_batch=data.batch(records[12000])[0]
    probe_h,probe_mask=model.condition(**{k:v[:1] for k,v in paired_model_inputs(probe_batch).items()})
    probe_z=draw((1,4,750,24),cfg.rollout_seed,0,'fixed_dynamics_TRAIN_probe').cuda()
    def save():
        path=attempt/'checkpoints'/f'step_{12000+u:06d}.pt'
        assert tree_hash(model.condition.state_dict())==fixed
        payload=dict(**metadata(model),model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng_state(),phase=arm,phase_step=u,global_step=12000+u,rows=rows,completed=u==2000,schedule_position=12000+u,parent_sha256=sha256_file(PARENT),manifest_sha256=sha256_file(AROOT/'manifest.json'),coordinate_stats_sha256=manifest['artifact_hashes']['coordinate_stats.json'],adaptation_protocol_sha256=sha256_file(ROOT/'PROTOCOL.json'),calibration=cal)
        tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.rename(path);write(path.with_suffix('.json'),dict(path=str(path),sha256=sha256_file(path)))
    try:
        with (attempt/'training.jsonl').open('x') as log:
            for row in rows:log.write(json.dumps(row)+'\n')
            while u<stop:
                rec=records[12000+u];assert rec['step']==12000+u+1;b,y,n,ids,slots=get_batch(data,rec);optimizer.zero_grad(set_to_none=True);start=time.time()
                fm,task,dyn,p,tau=losses(model,b,y,n,rec,cfg,scales)
                ramp=min((u+1)/200,1);lt=cal['lambda_ref']*(.1+.9*ramp);ld=cal['lambda_dyn']*ramp if arm=='T1-task-dynamics' else 0.
                diagnostics={}
                if (u+1)%100==0 or u==0:
                    gf=grad(fm,params);gt=grad(task,params);gd=grad(dyn,params)
                    diagnostics=dict(FM_gradient=float(gf.norm()),task_gradient=float(gt.norm()),dyn_gradient=float(gd.norm()),FM_task_cosine=float(torch.nn.functional.cosine_similarity(gf,gt,dim=0)))
                    del gf,gt,gd
                total=fm+lt*task+ld*dyn;total.backward()
                assert torch.isfinite(total) and all(v.grad is None or torch.isfinite(v.grad).all() for v in params)
                norm=float(sum(v.grad.square().sum() for v in params if v.grad is not None).sqrt());optimizer.step();u+=1
                row=dict(phase_step=u,global_step=12000+u,FM=float(fm),task=float(task),dynamic=float(dyn),lambda_task=lt,lambda_dyn=ld,loss=float(total),gradient_norm=norm,source_indices=rec['source_indices'],crop_start=b['crop_start'].tolist(),slots=slots.tolist(),target_ids=ids,valid_frames=int(n.sum()),seconds=time.time()-start,task_components=b['_task_components'],**diagnostics)
                rows.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                if u%100==0 or u==1:
                    print(arm,u,row,flush=True)
                    with torch.no_grad():
                        probe=model.rollout(probe_h,probe_mask,probe_z,16,probe_batch['crop_start'][:1])[0,:,:int(probe_batch['source_lengths'][0])]
                        metrics={}
                        for g,a,z in GROUPS:
                            q=probe[...,a:z];full,parts=decomposition(q)
                            metrics[g]=dict(total=float(full),**{k:float(v) for k,v in parts.items()},speed=float(q.diff(dim=1).abs().mean()),acceleration=float(q.diff(dim=1).diff(dim=1).abs().mean()))
                    write(attempt/f'probe_{u:06d}.json',metrics)
                if u in (500,1000,2000,stop):save()
                del fm,task,dyn,p,total
        assert u==stop==len(rows) and {int(s['step']) for s in optimizer.state.values()}=={12000+u}
        write(attempt/'status.json',dict(completed=u==2000,requested=2000,actual=u,rows=len(rows),seconds=time.time()-started,peak_memory_bytes=torch.cuda.max_memory_allocated(),valid_frames=sum(r['valid_frames'] for r in rows),condition_unchanged=tree_hash(model.condition.state_dict())==fixed))
    except BaseException as e:
        if not (attempt/'checkpoints'/f'step_{12000+u:06d}.pt').exists():save()
        write(attempt/'failure.json',dict(step=u,error=repr(e),completed=False));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--calibrate',action='store_true');p.add_argument('--arm');p.add_argument('--stop',type=int,default=2000);p.add_argument('--resume');p.add_argument('--label');a=p.parse_args();configure_flow()
    if a.calibrate:calibrate()
    else:train(a.arm,a.stop,a.resume,a.label)
