"""Matched 1000-step continuation, inherited T0 optimizer, explicit resume and RNG."""
import argparse,json,time,random,shutil
import numpy as np,torch
from reaction_flow.config import FlowConfig
from reaction_flow.data import FlowData,draw
from reaction_flow.flow_path import flow_path,flow_loss
from reaction_flow.sampler import load_checkpoint,metadata
from reaction_flow.task_dynamics_train import get_batch,task_parts
from reaction_flow.train import configure_flow
from hirp.train_phase25 import rng_state,restore_rng
from .common import *
from .model import BertSemanticFlow

def build(arm,device='cuda:0'):
 protocol=verify();base,parent=load_checkpoint(PARENT,device);assert (base.config.d_model,base.config.heads,base.config.T,base.config.B)==(256,8,750,4)
 model=BertSemanticFlow(base,arm,ContentCache()).to(device).train();opt=torch.optim.AdamW(base.velocity.parameters(),lr=2e-5,weight_decay=.01);opt.load_state_dict(parent['optimizer'])
 for g in opt.param_groups:g['lr']=2e-5;g['weight_decay']=.01
 opt.add_param_group(dict(params=list(model.semantic_parameters()),lr=1e-4,weight_decay=.01));return model,opt,parent

def batch_inputs(data,sources,rec):
 b,y,n,ids,slots=get_batch(data,rec);events=[];frames=[];durations=[]
 for i,clip in enumerate(b['clip_id']):
  start=int(b['crop_start'][i]);length=int(b['source_lengths'][i]);assert int(b['source_total_length'][i])==len(sources.rows[clip]['frame_pts']);es,q,d=sources.crop(clip,start,length);events.append(es);frames.append(q);durations.append(d)
 return b,y,n,ids,slots,events,torch.tensor(np.stack(frames),device=data.device,dtype=torch.float32),durations

def update(model,opt,parent,batch,rec,drop,j):
 b,y,n,ids,slots,events,frames,durations=batch;cfg=model.base.config;device=y.device
 # RNG is also checkpointed; keyed source/noise/dropout streams do not depend on model construction.
 opt.param_groups[-1]['lr']=1e-4*min(j/100,1);opt.zero_grad(set_to_none=True)
 h,sm=model.condition(b['speaker_audio'],b['speaker_emotion'],b['speaker_3dmm'],b['source_lengths'],events,frames,durations,drop)
 valid=torch.arange(cfg.T,device=device)[None]<n[:,None];target=model.base.transform(y).detach();noise=draw(target.shape,cfg.noise_seed,rec['step'],'FM_noise').to(device);tau=draw((cfg.B,),cfg.tau_seed,rec['step'],'FM_tau',False).to(device);u,v=flow_path(target,noise,tau,valid)
 fm=flow_loss(model.base.velocity(u,tau,h,valid,sm,b['crop_start']),v,valid);z=draw((cfg.B,4,cfg.T,24),cfg.rollout_seed,rec['step'],'task_rollout').to(device);pred=model.base.rollout(h,sm,z,16,b['crop_start'],gradient_checkpointing=True);task,parts=task_parts(pred,b);weight=parent['calibration']['lambda_ref'];loss=fm+weight*task;loss.backward()
 assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
 norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.))
 grads={name:float(p.grad.norm()) if p.grad is not None else None for name,p in model.named_parameters() if name.startswith(('fusion','byte_embedding'))}
 opt.step()
 # Fingerprints are verification only; no listener identifiers enter model.condition.
 hsh=lambda x:hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
 exposure=dict(source_indices=rec['source_indices'],crop_occurrences=rec['crop_occurrences'],crop_start=b['crop_start'].tolist(),target_slots=slots.tolist(),target_ids=ids,source_lengths=b['source_lengths'].tolist(),selected_target_lengths=n.tolist(),source_tensors_sha256={k:hsh(b[k]) for k in ['speaker_audio','speaker_emotion','speaker_3dmm']},target_tensor_sha256=hsh(b['_targets']),FM_noise_sha256=hsh(noise),tau_sha256=hsh(tau),rollout_noise_sha256=hsh(z),semantic_dropout=drop,events=[[dict(id=e.event_id,text_sha256=text_key(e.text),interval=e.interval) for e in es or []] for es in events])
 # Some dataset target IDs are tuples; JSON normalizes them without model access.
 row=dict(step=j,global_step=14000+j,FM=float(fm),task=float(task),loss=float(loss),lambda_ref=weight,gradient_norm_before_clip=norm,semantic_lr=opt.param_groups[-1]['lr'],semantic_grads=grads,exposure=exposure,
 valid_source_frames=int(b['source_lengths'].sum()),valid_target_frames=int(n.sum()),non_null_before=sum(bool(e) for e in events),non_null_after=sum(bool(e) and not drop[i] for i,e in enumerate(events)),active_semantic_samples=sum(bool(e) and not drop[i] for i,e in enumerate(events)) if model.arm!='P0-null' else 0,
 timed_events=sum(e.interval is not None for es in events for e in es or []),content_only_events=sum(e.interval is None for es in events for e in es or []))
 return row

def save(path,model,opt,parent,j,rows,diagnostic):
 payload=dict(run_identity_sha256=sha(OUT/'RUN_IDENTITY.json') if (OUT/'RUN_IDENTITY.json').exists() else None,format_version='bert-semantic-controls-v1',arm=model.arm,step=j,schedule_cursor=j,base_metadata=metadata(model.base),model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),parent_sha256=PARENT_SHA,protocol_sha256=sha(OUT/'PROTOCOL.json'),content_index_sha256=sha(OUT/'content/index.json'),schedule_sha256=sha(OUT/'schedule.json'),dropout_sha256=sha(OUT/'dropout.json'),calibration=parent['calibration'],rows=rows,diagnostic=diagnostic,code_identity_sha256=sha(OUT/'CODE_IDENTITY.json') if (OUT/'CODE_IDENTITY.json').exists() else None)
 tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path);write(path.with_suffix('.json'),dict(sha256=sha(path),step=j,arm=model.arm,diagnostic=diagnostic))

def train(arm,stop=1000,diagnostic=None):
 assert arm in ARMS and 1<=stop<=1000;configure_flow();torch.set_num_threads(4);random.seed(123);np.random.seed(123);torch.manual_seed(123)
 if (OUT/'CODE_IDENTITY.json').exists():
  for p,h in read(OUT/'CODE_IDENTITY.json').items():assert sha(p)==h,'code changed '+p
 model,opt,parent=build(arm);sources=Sources('train');data=FlowData(model.base.config,'cuda:0');records=read(OUT/'schedule.json');drops=read(OUT/'dropout.json');assert len(records)==len(drops)==1000
 dest=OUT/('diagnostics/'+diagnostic if diagnostic else 'training/'+arm);dest.mkdir(parents=True,exist_ok=True);(dest/'checkpoints').mkdir(exist_ok=True)
 import fcntl
 lock=open(dest/'train.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 initial=dict(arm=arm,parent_sha256=PARENT_SHA,base_hash=state_hash(model.base.state_dict()),fusion_hash=state_hash(model.fusion.state_dict()),optimized_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),frozen_in_model=sum(p.numel() for p in model.parameters() if not p.requires_grad),total_in_model=sum(p.numel() for p in model.parameters()),external_frozen_BERT_parameters=model.content_cache.meta['frozen_parameters'] if arm in ('P2-bert','P3-bert-time') else 0,initial_optimizer_base_sha256=state_hash({str(i)+'.'+k:v for i,st in opt.state_dict()['state'].items() for k,v in st.items() if isinstance(v,torch.Tensor)}))
 if not (dest/'initial.json').exists():write(dest/'initial.json',initial)
 else:assert read(dest/'initial.json')==initial
 checkpoints=sorted((dest/'checkpoints').glob('step_*.pt'));j=0;rows=[]
 if checkpoints:
  saved=torch.load(checkpoints[-1],map_location='cpu',weights_only=True);assert saved['arm']==arm and saved['protocol_sha256']==sha(OUT/'PROTOCOL.json') and saved['content_index_sha256']==sha(OUT/'content/index.json') and saved['diagnostic']==diagnostic
  if diagnostic is None:assert saved['run_identity_sha256']==sha(OUT/'RUN_IDENTITY.json')
  model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng']);j=saved['step'];rows=json.loads(json.dumps(saved['rows']));assert len(rows)==saved['schedule_cursor']==j
  if j>stop:raise ValueError('cannot rewind completed steps')
  if j==stop:return
  lp=dest/'training.jsonl'
  if lp.exists():
   existing=[json.loads(l) for l in lp.read_text().splitlines()];assert existing[:j]==rows
   if len(existing)>j:
    archive=dest/f'interrupted_log_{time.time_ns()}.jsonl';shutil.copyfile(lp,archive)
   lp.write_text(''.join(json.dumps(r)+'\n' for r in rows))
 else:save(dest/'checkpoints/step_000000.pt',model,opt,parent,0,[],diagnostic)
 tick=time.time()
 with (dest/'training.jsonl').open('a') as log:
  for step in range(j+1,stop+1):
   begin=time.time();batch=batch_inputs(data,sources,records[step-1]);row=update(model,opt,parent,batch,records[step-1],drops[step-1],step);torch.cuda.synchronize();row['seconds']=time.time()-begin;row['peak_gpu_bytes']=torch.cuda.max_memory_allocated();row=json.loads(json.dumps(row));rows.append(row);log.write(json.dumps(row)+'\n');log.flush();write(dest/'monitor_latest.json',{k:v for k,v in row.items() if k not in ('semantic_grads','exposure')})
   if step%25==0 or step==j+1:print(arm,step,float(row['loss']),row['seconds'],flush=True)
   if step in (100,500,1000) or step==stop:save(dest/'checkpoints'/f'step_{step:06d}.pt',model,opt,parent,step,rows,diagnostic)
 assert len(rows)==stop;write(dest/'status.json',dict(arm=arm,actual_steps=stop,requested_steps=stop,completed=stop==1000,invocation_seconds=time.time()-tick,diagnostic=diagnostic))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--stop',type=int,default=1000);p.add_argument('--diagnostic');a=p.parse_args();train(a.arm,a.stop,a.diagnostic)
