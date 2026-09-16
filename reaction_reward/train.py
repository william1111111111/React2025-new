import argparse,random,time,copy
import numpy as np
import torch
import torch.nn.functional as F
from .common import *
from .data import Data
from .model import Judge
ARMS={'RM-Paired':[1.,.5,0.,0.],'RM-Multi':[1.,.5,.25,0.],'RM-Gen':[1.,.5,.25,.5]}
FAMILIES=['base_context','temporal','weak_context','generated'];SLOTS=[16,8,4,4]
def score(model,data,rows,key):return model(*data.batch(rows,key))
def difference(model,data,rows,family):
 a=score(model,data,rows,'A');b=score(model,data,rows,'B');d=a-b;return d[:,0] if family in ['base_context','weak_context'] else d[:,1] if family=='temporal' else d.sum(-1)
def save(path,state):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)
@torch.no_grad()
def evaluate(model,data,records):
 model.eval();out=[]
 for family in FAMILIES:
  rs=[r for r in records if r['family']==family]
  for start in range(0,len(rs),8):
   part=rs[start:start+8];ds=difference(model,data,part,family).cpu().tolist()
   for r,d in zip(part,ds):out.append(dict(family=family,group=r['group'],state=r['state'],weight=r['weight'],difference=d,shift_seconds=r.get('shift_seconds'),window=r['window'],unseen=r.get('unseen',False)))
 return out

def main(arm):
 torch.set_num_threads(3);torch.manual_seed(123);np.random.seed(123);random.seed(123);data=Data();model=Judge().cuda();opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01);dest=OUT/'training'/arm;dest.mkdir(parents=True,exist_ok=True)
 records=read(OUT/'REAL_EVIDENCE.json')+read(OUT/'GENERATED_EVIDENCE.json');pools={f:[r for r in records if r['split']=='RM_fit' and r['family']==f] for f in FAMILIES};cal=[r for r in records if r['split']=='RM_cal'];start=0;history=[]
 initial_hash=hashlib.sha256(b''.join(x.detach().cpu().numpy().tobytes() for x in model.state_dict().values())).hexdigest();write(dest/'initial.json',dict(seed=123,initial_sha256=initial_hash,trainable=sum(p.numel() for p in model.parameters()),pretrained_weights=False,weights=ARMS[arm]))
 if (dest/'latest.pt').exists():
  state=torch.load(dest/'latest.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);torch.set_rng_state(state['rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);start=state['step'];history=state['history']
 if start in [1000,2000,4000] and not (dest/f'cal_{start:06d}.json').exists():
  write(dest/f'cal_{start:06d}.json',evaluate(model,data,cal))
 for step in range(start,4000):
  tick=time.time();model.train();opt.zero_grad(set_to_none=True);losses={};exposure={};lr=1e-4*min(1,(step+1)/200)
  for g in opt.param_groups:g['lr']=lr
  for fam,slots,mult in zip(FAMILIES,SLOTS,ARMS[arm]):
   torch.manual_seed(seed(f'dropout|123|{step}|{fam}'));rng=random.Random(seed(f'123|{step}|{fam}'));rows=rng.choices(pools[fam],k=slots) if pools[fam] else [];active=[r for r in rows if r['weight']>0];exposure[fam]=dict(scheduled=len(rows),active=len(active) if mult else 0,unknown=sum(r['weight']==0 for r in rows));den=sum(r['weight'] for r in active);value=0.
   if not active or mult==0:losses[fam]=None;continue
   for i in range(0,len(active),4):
    batch=active[i:i+4];d=difference(model,data,batch,fam);w=d.new_tensor([r['weight'] for r in batch]);loss=(F.softplus(-d)*w).sum()/den;assert torch.isfinite(loss);(mult*loss).backward();value+=float(loss.detach())
   losses[fam]=value
  norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));assert np.isfinite(norm);opt.step();row=dict(step=step+1,losses=losses,exposure=exposure,gradient_norm=norm,lr=lr,seconds=time.time()-tick);history.append(row)
  with (dest/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  write(dest/'status.json',dict(stage='RM_training',arm=arm,step=step+1,total=4000,latest=row,time=time.time()))
  if (step+1)%25==0 or step==0:
   state=dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),step=step+1,history=history,initial_hash=initial_hash);save(dest/'latest.pt',state);print(arm,'step',step+1,'loss',losses,flush=True)
  if step+1 in [1000,2000,4000]:
   save(dest/'checkpoints'/f'step_{step+1:06d}.pt',state);pred=evaluate(model,data,cal);write(dest/f'cal_{step+1:06d}.json',pred)
 values={}
 for s in [1000,2000,4000]:
  rs=read(dest/f'cal_{s:06d}.json');tasks={}
  for fam in ['base_context','temporal','generated']:
   vals=[np.logaddexp(0,-r['difference']) for r in rs if r['family']==fam and r['weight']>0]
   if vals:tasks[fam]=float(np.mean(vals))
  values[str(s)]=dict(macro_NLL=float(np.mean(list(tasks.values()))) if tasks else None,tasks=tasks)
 valid=[int(s) for s,v in values.items() if v['macro_NLL'] is not None];assert valid
 chosen=min(valid,key=lambda s:(values[str(s)]['macro_NLL'],s));write(dest/'SELECTION.json',dict(selected_step=chosen,selected_checkpoint=str(dest/'checkpoints'/f'step_{chosen:06d}.pt'),calibration=values,selection_split='RM_cal',audit_not_read=True,time=time.time()))
 print(arm,'SELECTED',chosen,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);main(p.parse_args().arm)
