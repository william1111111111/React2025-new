import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from mam_target.model import load_checkpoint,metadata
from mam_target.data import TaskData
from mam_refine.losses import objective
from mam_refine.audit import ROOT,write
from hirp.train_phase25 import TrainConfig,rng_state,restore_rng,stream_hashes
from hirp.train_phase21 import configure
from hirp.paired_data import paired_model_inputs
from hirp.phase15_audit import sha256_file
from hirp.train_phase1 import state_hash

def tree_hash(x):
 h=hashlib.sha256()
 def add(v):
  if torch.is_tensor(v):h.update(str((v.dtype,tuple(v.shape))).encode());h.update(v.detach().cpu().numpy().tobytes())
  elif isinstance(v,dict):
   for k in sorted(v,key=str):h.update(str(k).encode());add(v[k])
  elif isinstance(v,(list,tuple)):
   for t in v:add(t)
  else:h.update(repr(v).encode())
 add(x);return h.hexdigest()

def train(arm,stop=2000,label=None,resume=None,device='cuda:0'):
 manifest=json.loads((ROOT/'manifest.json').read_text());mode=manifest['arms'][arm]
 assert sha256_file(manifest['parent_checkpoint'])==manifest['parent_sha256']
 assert sha256_file(manifest['data_manifest'])==manifest['data_manifest_sha256']
 assert sha256_file(ROOT/'noise.npy')==manifest['noise_sha256']
 records=json.loads((ROOT/'schedule.json').read_text())['records'];noises=torch.from_numpy(np.load(ROOT/'noise.npy'))
 assert len(records)==len(noises)==2000 and stream_hashes(records,noises)==manifest['schedule_hashes']
 data=TaskData(json.loads(Path(manifest['data_manifest']).read_text()),TrainConfig(max_steps=8000),'C0',device)
 m,s=load_checkpoint(resume or manifest['parent_checkpoint'],device);opt=torch.optim.AdamW(m.parameters(),lr=manifest['lr'],weight_decay=manifest['weight_decay']);opt.load_state_dict(s['optimizer'])
 if resume:
  assert s['arm']==arm and s['parent_sha256']==manifest['parent_sha256'] and s['coverage_mode']==mode
  additional=s['additional_steps'];rows=s['new_rows'];inherited=s['rows'][:6000]
  assert stream_hashes(records[:additional],noises[:additional])==s['consumed_new_hashes']
 else:additional=0;rows=[];inherited=s['rows'];assert s['step']==len(inherited)==6000
 for group in opt.param_groups:assert group['lr']==1e-4 and group['weight_decay']==.01
 initial_model_hash=state_hash(m);initial_optimizer_hash=tree_hash(opt.state_dict());prior=tree_hash(m.prior.state_dict())
 out=ROOT/(label or arm);out.mkdir(exist_ok=True);attempt=out/f'attempt_{len(list(out.glob("attempt_*"))):03d}';attempt.mkdir();(attempt/'checkpoints').mkdir()
 write(attempt/'initial.json',dict(arm=arm,coverage_mode=mode,parent_sha256=manifest['parent_sha256'],resumed_from=str(resume) if resume else None,initial_model_hash=initial_model_hash,initial_optimizer_hash=initial_optimizer_hash,start_additional_steps=additional,physical_gpu=1))
 m.train();restore_rng(s['rng']);start=time.time();torch.cuda.reset_peak_memory_stats(device)
 try:
  with (attempt/'training.jsonl').open('x') as log:
   for row in rows:log.write(json.dumps(row)+'\n')
   while additional<stop:
    rec=records[additional];assert rec['step']==6001+additional
    batch,_,targets,lengths,ids=data.task_batch(rec);opt.zero_grad(set_to_none=True);diagnostic=additional==0 or (additional+1)%100==0
    noise=noises[additional].to(device).requires_grad_(diagnostic)
    p=m(**paired_model_inputs(batch),sample_count=4,noise=noise)
    v=objective(p,batch,targets,lengths,manifest['weights'],mode)
    diag={}
    if diagnostic:
     valid=torch.arange(p.shape[2],device=device)[None]<batch['source_lengths'][:,None]
     for group,a,b in [('AU',0,15),('VA',15,17),('expression',17,25)]:
      q=p[...,a:b];weights=torch.linspace(.1,1,b-a,device=device)
      derivative=torch.autograd.grad((q*weights).sum(),noise,retain_graph=True)[0]
      diag[group]=dict(noise_vjp_norm=float(derivative.norm()),sample_std=float(q.detach().std(1)[valid].mean()),mean=float(q.detach().permute(0,2,1,3)[valid].mean()))
    v['loss'].backward()
    assert torch.isfinite(v['loss']) and all(torch.isfinite(t.grad).all() for t in m.parameters() if t.grad is not None)
    opt.step();additional+=1;global_step=6000+additional
    row=dict(step=global_step,additional_step=additional,**{k:float(t) for k,t in v.items()},source_indices=rec['source_indices'],targets=ids,crop_start=batch['crop_start'].tolist(),source_lengths=batch['source_lengths'].tolist(),pair_lengths=batch['pair_lengths'].tolist(),coverage_mode=mode)
    rows.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
    if diagnostic:write(attempt/f'diagnostic_{additional:06d}.json',diag)
    if additional%50==0:print(arm,additional,float(v['loss']),time.time()-start,flush=True)
    if additional in (500,2000,stop):
     assert tree_hash(m.prior.state_dict())==prior
     saved=dict(**metadata(m),model=m.state_dict(),optimizer=opt.state_dict(),step=global_step,arm=arm,seed=123,weights=manifest['weights'],rows=inherited+rows,new_rows=rows,additional_steps=additional,requested_additional_steps=2000,parent_sha256=manifest['parent_sha256'],coverage_mode=mode,rng=rng_state(),scales=manifest['scales'],split_hash=manifest['split_hash'],consumed_new_hashes=stream_hashes(records[:additional],noises[:additional]),completed=additional==2000)
     path=attempt/'checkpoints'/f'step_{global_step:06d}.pt';tmp=path.with_suffix('.tmp');torch.save(saved,tmp);tmp.rename(path)
     write(path.with_suffix('.json'),dict(path=str(path),sha256=sha256_file(path),step=global_step,additional_steps=additional,arm=arm))
  assert additional==stop==len(rows)
  steps={int(v['step']) for v in opt.state.values() if 'step' in v};assert steps=={6000+additional}
  write(attempt/'status.json',dict(completed=additional==2000,additional_optimizer_steps=additional,requested_additional_steps=2000,actual_global_step=6000+additional,new_training_rows=len(rows),new_valid_frames=sum(sum(r['source_lengths']) for r in rows),inherited_valid_frames=manifest['inherited_valid_frames'],seconds=time.time()-start,peak_memory_bytes=torch.cuda.max_memory_allocated(device),parameters=sum(p.numel() for p in m.parameters()),optimizer_step_values=list(steps),prior_unchanged=True))
 except BaseException as e:write(attempt/'failure.json',dict(error=repr(e),additional_steps=additional,completed=False));raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['R2-cont','R3-cover'],required=True);p.add_argument('--stop',type=int,default=2000);p.add_argument('--label');p.add_argument('--resume');a=p.parse_args();configure();torch.use_deterministic_algorithms(True);train(a.arm,a.stop,a.label,a.resume)
