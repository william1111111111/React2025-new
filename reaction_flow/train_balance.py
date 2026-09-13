"""Two fixed-budget G1 continuations with serializable detached ratio control."""
import argparse,json,time,math
import torch
from .balance_common import ROOT,PARENT,PARENT_SHA,AROOT,resources,initialize,gradients
from .shared_noise import prior_metadata,GLOBAL_SEED
from .train_shared import losses
from .task_dynamics_train import get_batch
from .sampler import metadata
from .train import write,configure_flow
from hirp.phase15_audit import sha256_file,canonical_hash
from hirp.train_phase25 import rng_state
from mam_refine.train import tree_hash

def train(arm,stop=2000,resume=None,label=None,extra_checkpoints=()):
    assert 0<stop<=2000
    cfg,records,data,protocol=resources()
    model,opt,params,control,saved=initialize(arm,resume=resume)
    u=control.completed;rows=saved['rows'] if resume else []
    assert u<stop
    if resume:
        assert saved['schedule_position']==16000+u
        assert saved['schedule_sha256']==sha256_file(ROOT/'schedule.json')
        assert saved['consumed_prefix_sha256']==canonical_hash(records[:u])
    out=ROOT/(label or arm);out.mkdir(exist_ok=True)
    attempt=out/f'attempt_{len(list(out.glob("attempt_*"))):03d}';attempt.mkdir();(attempt/'checkpoints').mkdir()
    condition_hash=tree_hash(model.condition.state_dict())
    initial=dict(model_hash=tree_hash(model.state_dict()),optimizer_hash=tree_hash(opt.state_dict()),condition_hash=condition_hash,parent_sha256=PARENT_SHA,rho=model.rho,start=u,controller=control.state(arm))
    write(attempt/'initial.json',initial)
    started=time.time()
    def save():
        assert tree_hash(model.condition.state_dict())==condition_hash
        payload=dict(**metadata(model),model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),phase=arm,phase_step=u,global_step=16000+u,rows=rows,completed=u==2000,schedule_position=16000+u,parent_sha256=PARENT_SHA,manifest_sha256=sha256_file(AROOT/'manifest.json'),coordinate_stats_sha256=sha256_file(AROOT/'coordinate_stats.json'),adaptation_protocol_sha256=sha256_file(ROOT/'PROTOCOL.json'),calibration_sha256=sha256_file(ROOT/'calibration.json'),schedule_sha256=sha256_file(ROOT/'schedule.json'),consumed_prefix_sha256=canonical_hash(records[:u]),initial_prior=prior_metadata(.05),ratio_controller=control.state(arm),training_stage='prior-task-balance-v1')
        payload['format_version']='reaction-flow-shared24-v1'
        path=attempt/'checkpoints'/f'step_{16000+u:06d}.pt';tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)
        write(path.with_suffix('.json'),dict(path=str(path),sha256=sha256_file(path)))
    try:
        with (attempt/'training.jsonl').open('x') as log:
            for row in rows:log.write(json.dumps(row,allow_nan=False)+'\n')
            while u<stop:
                t=u+1;rec=records[u];assert rec['step']==16000+t
                batch,y,n,ids,slots=get_batch(data,rec);opt.zero_grad(set_to_none=True);start=time.time()
                fm,task,zero,pred,tau=losses(model,batch,y,n,rec,cfg)
                weight=control.weight(arm,t);diagnostic=t%20==0;old_state=control.state(arm)
                a=b=cos=None
                if diagnostic:a,b,cos=gradients(fm,task,params)
                loss=fm+weight*task
                if not torch.isfinite(loss):raise ValueError('nonfinite total loss')
                loss.backward()
                if not any(p.grad is not None for p in params):raise ValueError('empty total gradient')
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in params):raise ValueError('nonfinite total gradient')
                assert all(p.grad is None for p in model.condition.parameters())
                norm=float(sum(p.grad.square().sum() for p in params if p.grad is not None).sqrt())
                if not math.isfinite(norm):raise ValueError('nonfinite total gradient norm')
                before=[p.detach().clone() for p in params] if diagnostic else None
                opt.step();u=t
                next_weight=control.advance(arm,t,(a,b) if diagnostic else None)
                row=dict(phase_step=u,global_step=16000+u,FM=float(fm),task=float(task),loss=float(loss),lambda_task=weight,lambda_next=next_weight,lambda_goal=control.goal,lambda_goal_used=old_state['lambda_goal'],cap_hit=control.cap_hit,cap_hit_used=old_state['cap_hit'],cap_hit_applied=arm=='B1-ratio' and old_state['cap_hit'],EMA_A=control.A,EMA_B=control.B,gradient_norm=norm,valid_frames=int(n.sum()),seconds=time.time()-start,source_indices=rec['source_indices'],target_slots=slots.tolist(),crop_start=batch['crop_start'].tolist(),target_ids=ids,tau=tau.detach().cpu().tolist(),record_sha256=canonical_hash(rec),FM_initial_sha256=batch['_FM_initial_hash'],task_initial_sha256=batch['_task_initial_hash'],task_components=batch['_task_components'],noise_step=rec['step'],global_seed=GLOBAL_SEED)
                if diagnostic:
                    row.update(FM_gradient=a,task_gradient=b,weighted_task_gradient=weight*b,r_actual=weight*b/a,FM_task_cosine=cos,optimizer_update_norm=float(sum((p.detach()-q).square().sum() for p,q in zip(params,before)).sqrt()),EMA_before_A=old_state['A'],EMA_before_B=old_state['B'])
                rows.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                if diagnostic or u==1:print(arm,u,'lambda',weight,'loss',row['loss'],'ratio',row.get('r_actual'),flush=True)
                if u in (500,1000,2000,stop) or u in extra_checkpoints:save()
                del fm,task,zero,pred,loss,before
        assert len(rows)==u==control.completed
        assert {int(s['step']) for s in opt.state.values()}=={16000+u}
        write(attempt/'status.json',dict(completed=u==2000,actual=u,requested=2000,seconds=time.time()-started,valid_frames=sum(r['valid_frames'] for r in rows),condition_unchanged=tree_hash(model.condition.state_dict())==condition_hash))
    except BaseException as error:
        try:write(attempt/'failure.json',dict(step=u,error=repr(error),completed=False))
        finally:
            try:save()
            except Exception as save_error:print('CHECKPOINT_SAVE_FAILED',repr(save_error),flush=True)
        raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--stop',type=int,default=2000);p.add_argument('--resume');p.add_argument('--label');a=p.parse_args();configure_flow();train(a.arm,a.stop,a.resume,a.label)
