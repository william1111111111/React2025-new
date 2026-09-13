"""T0 continuation with one prior-covariance change, fixed task weight."""
import argparse,json,time,hashlib
from pathlib import Path
import torch
from .config import FlowConfig,ROOT as AROOT,DATA_MANIFEST
from .data import FlowData,draw
from .shared_noise import training_noise,prior_metadata,GLOBAL_SEED
from .shared_model import load_checkpoint
from .sampler import metadata
from .prepare_shared import ROOT,PARENT
from .train import write,configure_flow
from .task_dynamics_train import get_batch,task_parts,grad
from .dynamics import GROUPS,decomposition
from .flow_path import flow_path,flow_loss
from hirp.paired_data import paired_model_inputs
from hirp.phase15_audit import sha256_file
from hirp.train_phase25 import rng_state,restore_rng
from mam_refine.train import tree_hash

def resources(cfg,device):
    protocol=json.loads((ROOT/'PROTOCOL.json').read_text())
    assert sha256_file(ROOT/'schedule.json')==protocol['schedule_sha256']
    assert sha256_file(DATA_MANIFEST)==protocol['data_manifest_sha256']
    assert sha256_file(AROOT/'coordinate_stats.json')==protocol['coordinate_stats_sha256']
    rows=json.loads((ROOT/'schedule.json').read_text())['records'];assert len(rows)==2000
    return json.loads((AROOT/'manifest.json').read_text()),rows,FlowData(cfg,device)

def losses(model,batch,selected,lengths,record,cfg,scales=None):
    h,sm=model.condition(**paired_model_inputs(batch));valid=torch.arange(cfg.T,device=h.device)[None]<lengths[:,None]
    target=model.transform(selected).detach();noise=training_noise(tuple(target.shape),cfg,record['step'],'FM',model.rho,valid)
    batch['_FM_initial_hash']=hashlib.sha256(noise.detach().cpu().numpy().tobytes()).hexdigest()
    tau=draw((cfg.B,),cfg.tau_seed,record['step'],'FM_tau',False).to(h.device)
    state,velocity=flow_path(target,noise,tau,valid);v=model.velocity(state,tau,h,valid,sm,batch['crop_start'])
    fm=flow_loss(v,velocity,valid)
    z=training_noise((cfg.B,4,cfg.T,24),cfg,record['step'],'task',model.rho,sm)
    batch['_task_initial_hash']=hashlib.sha256(z.detach().cpu().numpy().tobytes()).hexdigest()
    pred=model.rollout(h,sm,z,16,batch['crop_start'],gradient_checkpointing=True)
    task,components=task_parts(pred,batch);batch['_task_components']=components
    return fm,task,fm*0,pred,tau

def train(arm,stop=2000,resume=None,label=None):
    assert arm in ('G0-local','G1-shared') and 0<stop<=2000
    cfg=FlowConfig();manifest,records,data=resources(cfg,'cuda:0');cal=json.loads((ROOT/'PROTOCOL.json').read_text());assert cal['parent_sha256']==sha256_file(PARENT);rho=cal['arms'][arm]['rho']
    model,saved=load_checkpoint(resume or PARENT,'cuda:0',parent_rho=rho);model.train();params=list(model.velocity.parameters())
    optimizer=torch.optim.AdamW(params,lr=2e-5,weight_decay=.01);optimizer.load_state_dict(saved['optimizer'])
    for group in optimizer.param_groups:group['lr']=2e-5
    restore_rng(saved['rng']);u=saved['phase_step'] if resume else 0;rows=saved['rows'] if resume else []
    if resume:assert saved['phase']==arm and u==len(rows) and saved['schedule_position']==14000+u and saved['adaptation_protocol_sha256']==sha256_file(ROOT/'PROTOCOL.json')
    out=ROOT/(label or arm);out.mkdir(exist_ok=True);attempt=out/f'attempt_{len(list(out.glob("attempt_*"))):03d}';attempt.mkdir();(attempt/'checkpoints').mkdir()
    fixed=tree_hash(model.condition.state_dict());write(attempt/'initial.json',dict(parent_sha256=sha256_file(PARENT),velocity_hash=tree_hash(model.velocity.state_dict()),optimizer_hash=tree_hash(optimizer.state_dict()),condition_hash=fixed,LR=2e-5,start=u,optimizer_policy='inherit T0 AdamW states; both groups set lr2e-5',protocol_sha256=sha256_file(ROOT/'PROTOCOL.json')))
    scales=None;started=time.time();torch.cuda.reset_peak_memory_stats()
    probe_batch=data.batch(records[0])[0]
    probe_h,probe_mask=model.condition(**{k:v[:1] for k,v in paired_model_inputs(probe_batch).items()})
    probe_z=training_noise((1,4,750,24),cfg,14000,'task',rho,probe_mask)
    def save():
        path=attempt/'checkpoints'/f'step_{14000+u:06d}.pt'
        assert tree_hash(model.condition.state_dict())==fixed
        payload=dict(**metadata(model),model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng_state(),phase=arm,phase_step=u,global_step=14000+u,rows=rows,completed=u==2000,schedule_position=14000+u,parent_sha256=sha256_file(PARENT),manifest_sha256=sha256_file(AROOT/'manifest.json'),coordinate_stats_sha256=manifest['artifact_hashes']['coordinate_stats.json'],adaptation_protocol_sha256=sha256_file(ROOT/'PROTOCOL.json'),calibration=cal,initial_prior=prior_metadata(rho))
        payload['format_version']='reaction-flow-shared24-v1'
        tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.rename(path);write(path.with_suffix('.json'),dict(path=str(path),sha256=sha256_file(path)))
    try:
        with (attempt/'training.jsonl').open('x') as log:
            for row in rows:log.write(json.dumps(row)+'\n')
            while u<stop:
                rec=records[u];assert rec['step']==14000+u+1;b,y,n,ids,slots=get_batch(data,rec);optimizer.zero_grad(set_to_none=True);start=time.time()
                fm,task,dyn,p,tau=losses(model,b,y,n,rec,cfg,scales)
                lt=cal['lambda_task'];ld=0.
                diagnostics={}
                if (u+1)%100==0 or u==0:
                    gf=grad(fm,params);gt=grad(task,params);gd=grad(dyn,params)
                    diagnostics=dict(FM_gradient=float(gf.norm()),task_gradient=float(gt.norm()),dyn_gradient=float(gd.norm()),FM_task_cosine=float(torch.nn.functional.cosine_similarity(gf,gt,dim=0)))
                    del gf,gt,gd
                total=fm+lt*task+ld*dyn;total.backward()
                assert torch.isfinite(total) and all(v.grad is None or torch.isfinite(v.grad).all() for v in params)
                norm=float(sum(v.grad.square().sum() for v in params if v.grad is not None).sqrt());optimizer.step();u+=1
                row=dict(phase_step=u,global_step=14000+u,FM=float(fm),task=float(task),dynamic=float(dyn),lambda_task=lt,lambda_dyn=ld,loss=float(total),gradient_norm=norm,source_indices=rec['source_indices'],crop_start=b['crop_start'].tolist(),slots=slots.tolist(),target_ids=ids,valid_frames=int(n.sum()),seconds=time.time()-start,task_components=b['_task_components'],noise_keys=dict(local_FM=[cfg.noise_seed,rec['step'],'FM_noise'],local_task=[cfg.rollout_seed,rec['step'],'task_rollout'],global_seed=GLOBAL_SEED,global_step=rec['step'],rho=rho,FM_initial_sha256=b['_FM_initial_hash'],task_initial_sha256=b['_task_initial_hash']),**diagnostics)
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
        assert u==stop==len(rows) and {int(s['step']) for s in optimizer.state.values()}=={14000+u}
        write(attempt/'status.json',dict(completed=u==2000,requested=2000,actual=u,rows=len(rows),seconds=time.time()-started,peak_memory_bytes=torch.cuda.max_memory_allocated(),valid_frames=sum(r['valid_frames'] for r in rows),condition_unchanged=tree_hash(model.condition.state_dict())==fixed))
    except BaseException as e:
        if not (attempt/'checkpoints'/f'step_{14000+u:06d}.pt').exists():save()
        write(attempt/'failure.json',dict(step=u,error=repr(e),completed=False));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--stop',type=int,default=2000);p.add_argument('--resume');p.add_argument('--label');a=p.parse_args();configure_flow();train(a.arm,a.stop,a.resume,a.label)
