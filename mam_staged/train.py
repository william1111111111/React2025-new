"""Independent seed123 warm-start runner; 2000 new updates, new AdamW in all arms."""
import argparse,json,time
import torch
from mam_target.model import load_checkpoint,metadata
from mam_staged.common import ROOT,write,resources
from mam_staged.params import make_optimizer,apply_stage,partition
from mam_staged.numerics import consistent_attention
from mam_staged.losses import losses
from mam_refine.train import tree_hash
from hirp.train_phase25 import rng_state,restore_rng,stream_hashes
from hirp.train_phase21 import configure
from hirp.paired_data import paired_model_inputs
from hirp.phase15_audit import sha256_file
from hirp.train_phase1 import state_hash

ARMS=('S0_quality','S1_joint','S2_staged')

def train(arm,stop=2000,label=None,resume=None,device='cuda:0'):
    assert arm in ARMS and 1<=stop<=2000
    manifest,records,noises,data=resources(device)
    calibration=json.loads((ROOT/'calibration.json').read_text());calhash=sha256_file(ROOT/'calibration.json')
    model,saved=load_checkpoint(resume or manifest['parent_checkpoint'],device)
    parent,parent_saved=load_checkpoint(manifest['parent_checkpoint'],device)
    parent.requires_grad_(False);parent.eval();consistent_attention(parent);consistent_attention(model)
    optimizer=make_optimizer(model);assert len(optimizer.state)==0
    if resume:
        assert saved['arm']==arm and saved['parent_sha256']==manifest['parent_sha256'] and saved['calibration_sha256']==calhash
        optimizer.load_state_dict(saved['optimizer']);u=saved['additional_steps'];rows=saved['new_rows']
        assert saved['consumed_new_hashes']==stream_hashes(records[:u],noises[:u])
        assert saved['schedule_position']==u
    else:u=0;rows=[]
    assert len(rows)==u and stop>=u
    inherited=parent_saved['rows'];assert len(inherited)==parent_saved['step']==6000
    groups=partition(model);parent_groups=partition(parent)
    def verify_frozen(next_u):
        names=['prior']
        if arm=='S2_staged':names+=['source_base']+(['decoder_body'] if next_u<=500 else [])
        for name in names:
            for (n,p),(n0,p0) in zip(groups[name],parent_groups[name]):
                assert n==n0 and torch.equal(p,p0), 'frozen parameter changed: '+n
    verify_frozen(u)
    out=ROOT/(label or arm);out.mkdir(exist_ok=True)
    attempt=out/f'attempt_{len(list(out.glob("attempt_*"))):03d}';attempt.mkdir();(attempt/'checkpoints').mkdir()
    write(attempt/'initial.json',dict(arm=arm,initial_model_hash=state_hash(model),initial_optimizer_hash=tree_hash(optimizer.state_dict()),optimizer_reset=not bool(resume),parent_sha256=manifest['parent_sha256'],calibration_sha256=calhash,start_additional_steps=u,resume=str(resume) if resume else None,physical_gpu=1))
    restore_rng(saved['rng']);start=time.time();torch.cuda.reset_peak_memory_stats(device)
    def checkpoint():
        verify_frozen(u)
        payload=dict(**metadata(model),model=model.state_dict(),optimizer=optimizer.state_dict(),step=6000+u,arm=arm,seed=123,weights=manifest['weights'],rows=inherited+rows,new_rows=rows,additional_steps=u,requested_additional_steps=2000,parent_sha256=manifest['parent_sha256'],calibration_sha256=calhash,rng=rng_state(),scales=manifest['scales'],split_hash=manifest['split_hash'],consumed_new_hashes=stream_hashes(records[:u],noises[:u]),schedule_position=u,stage=phase,learning_rates={g['name']:g['lr'] for g in optimizer.param_groups},active_parameters=[n for n,p in model.named_parameters() if p.requires_grad],completed=u==2000,training_policy='new AdamW warmstart; dropout off; common unfused attention; score guard only')
        path=attempt/'checkpoints'/f'step_{6000+u:06d}.pt'
        assert not path.exists();temporary=path.with_suffix('.tmp');torch.save(payload,temporary);temporary.rename(path)
        write(path.with_suffix('.json'),dict(path=str(path),sha256=sha256_file(path),step=6000+u,additional_steps=u,arm=arm))
    try:
        with (attempt/'training.jsonl').open('x') as log:
            for row in rows:log.write(json.dumps(row)+'\n')
            while u<stop:
                phase=apply_stage(model,optimizer,arm,u);optimizer.zero_grad(set_to_none=True)
                rec=records[u];assert rec['step']==6001+u
                batch,_,targets,lengths,ids=data.task_batch(rec);inputs=paired_model_inputs(batch)
                diagnostic=u==0 or (u+1)%50==0
                noise=noises[u].to(device).requires_grad_(diagnostic)
                with torch.no_grad():p0=parent(**inputs,sample_count=4,noise=noise)
                pred=model(**inputs,sample_count=4,noise=noise)
                terms,stats=losses(pred,p0,batch,targets,lengths,ids,manifest['weights'],calibration,u,arm,common_parent_step=(u==0))
                diag={}
                if diagnostic:
                    with torch.no_grad():other=model(**inputs,sample_count=4,noise=-noise)
                    valid=torch.arange(pred.shape[2],device=device)[None]<batch['source_lengths'][:,None]
                    for name,a,b in [('AU',0,15),('VA',15,17),('expression',17,25)]:
                        q=pred[...,a:b];deriv=torch.autograd.grad((q*torch.linspace(.1,1,b-a,device=device)).sum(),noise,retain_graph=True)[0]
                        delta=(q.detach()-other[...,a:b]).square().mean((1,3))
                        diag[name]=dict(noise_vjp_norm=float(deriv.norm()),fixed_noise_vs_negative_mse=float(delta[valid].mean()),sample_std=float(q.detach().std(1)[valid].mean()))
                    diag['spread']={k:v.detach().cpu().tolist() for k,v in stats.items()}
                terms['loss'].backward()
                assert torch.isfinite(terms['loss']) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                if diagnostic:
                    diag['parameter_groups']={k:dict(active=sum(p.numel() for _,p in group if p.requires_grad),gradient_norm=float(sum((p.grad.square().sum() for _,p in group if p.grad is not None),torch.zeros((),device=device)).sqrt())) for k,group in groups.items()}
                assert all(p.grad is None for p in parent.parameters())
                optimizer.step();u+=1
                row=dict(step=6000+u,additional_step=u,stage=phase,**{k:float(v) for k,v in terms.items()},source_indices=rec['source_indices'],targets=ids,crop_start=batch['crop_start'].tolist(),source_lengths=batch['source_lengths'].tolist(),pair_lengths=batch['pair_lengths'].tolist(),learning_rates={g['name']:g['lr'] for g in optimizer.param_groups})
                rows.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                if diagnostic:write(attempt/f'diagnostic_{u:06d}.json',diag)
                if u%50==0:print(arm,u,float(terms['loss']),time.time()-start,flush=True)
                if u in (500,1000,1500,2000,stop):checkpoint()
        assert u==stop==len(rows)
        step_counts={}
        for name,group in groups.items():
            counts=sorted({int(optimizer.state[p]['step']) for _,p in group if p in optimizer.state})
            expected=u if arm!='S2_staged' or name=='stochastic' else max(0,u-500) if name=='decoder_body' else 0
            if name=='prior':expected=0
            assert counts==([expected] if expected else []),(name,counts,expected)
            step_counts[name]=counts
        write(attempt/'status.json',dict(completed=u==2000,requested_additional_steps=2000,additional_optimizer_steps=u,actual_global_step=6000+u,new_training_rows=len(rows),new_valid_frames=sum(sum(r['source_lengths']) for r in rows),inherited_valid_frames=manifest['inherited_valid_frames'],seconds=time.time()-start,peak_memory_bytes=torch.cuda.max_memory_allocated(device),parameters=sum(p.numel() for p in model.parameters()),optimizer_step_values=step_counts,frozen_checks_passed=True))
    except BaseException as error:
        write(attempt/'failure.json',dict(error=repr(error),additional_steps=u,completed=False));raise
    return attempt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--stop',type=int,default=2000);p.add_argument('--label');p.add_argument('--resume');a=p.parse_args()
    configure();torch.use_deterministic_algorithms(True);train(a.arm,a.stop,a.label,a.resume)
