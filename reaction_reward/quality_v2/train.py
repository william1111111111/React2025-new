import argparse,random,time
import numpy as np
import torch
import torch.nn.functional as F
from reaction_reward.common import ROOT,OUT as OLD,read,write,sha,seed
from reaction_reward.data import Data
from reaction_reward.train import save
from .model import Judge
OUT=ROOT/'runs/reaction_reward/quality_v2';ARMS=['A-repaired-RMGen','B-quality-heads'];STEPS=[250,500,1000,2000,4000]
def pair_difference(model,data,rows,family,arm,selection=False):
 d=model(*data.batch(rows,'A'))-model(*data.batch(rows,'B'))
 if family in ['base_context','weak_context']:return d[:,0]
 if family=='temporal':return d[:,1]
 return d[:,2]-d[:,3] if selection and arm==ARMS[1] else d[:,:2].sum(-1)
@torch.no_grad()
def evaluate(m,data,records,arm):
 m.eval();out=[]
 for f in ['base_context','temporal','weak_context','generated']:
  rs=[r for r in records if r['family']==f]
  for s in range(0,len(rs),4):
   batch=rs[s:s+4];ds=pair_difference(m,data,batch,f,arm,True).cpu().tolist()
   for r,d in zip(batch,ds):out.append(dict(family=f,difference=d,group=r['group'],source=r['source'],state=r['state'],weight=r['weight'],window=r['window'],shift_seconds=r.get('shift_seconds'),unseen=r.get('unseen',False)))
 return out

def main(arm):
 torch.set_num_threads(3);torch.manual_seed(123);np.random.seed(123);random.seed(123);data=Data();m=Judge().cuda()
 if arm==ARMS[0]:m.quality_C.requires_grad_(False);m.quality_D.requires_grad_(False)
 opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=.01);dest=OUT/'training'/arm;dest.mkdir(parents=True,exist_ok=True);records=read(OUT/'RECORDS.json');schedule=read(OUT/'SCHEDULE.json');qr=read(OUT/'QUALITY_RECORDS.json');qs=read(OUT/'QUALITY_SCHEDULE.json');cal=[r for r in records if r['split']=='RM_cal'];history=[];start=0
 import hashlib
 h=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in m.state_dict().values())).hexdigest();write(dest/'initial.json',dict(sha256=h,fresh_seed=123,pretrained=False,quality_weight=0 if arm==ARMS[0] else 1,total_parameters=sum(p.numel() for p in m.parameters()),quality_head_control='heads instantiated in both arms; A heads unused and get no gradients'))
 if (dest/'latest.pt').exists():
  s=torch.load(dest/'latest.pt',map_location='cpu',weights_only=False);m.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);start=s['step'];history=s['history'];torch.set_rng_state(s['rng']);torch.cuda.set_rng_state_all(s['cuda_rng'])
 if start in STEPS and not (dest/f'cal_{start:06d}.json').exists():write(dest/f'cal_{start:06d}.json',evaluate(m,data,cal,arm))
 for step in range(start,4000):
  tick=time.time();m.train();opt.zero_grad(set_to_none=True);losses={};exposure={}
  for g in opt.param_groups:g['lr']=1e-4*min(1,(step+1)/200)
  for f,mult in [('base_context',1.),('temporal',.5),('weak_context',.25),('generated',.5)]:
   torch.manual_seed(seed(f'quality_v2_dropout|123|{step}|{f}'));rs=[records[i] for i in schedule[step][f]];assert all(r['weight']>0 and r['state']=='PREFERRED' and r['split']=='RM_fit' for r in rs);assert len({r['source'] for r in rs})==len(rs);den=sum(r['weight'] for r in rs);value=0.
   for s in range(0,len(rs),4):
    b=rs[s:s+4];d=pair_difference(m,data,b,f,arm);loss=(F.softplus(-d)*d.new_tensor([r['weight'] for r in b])).sum()/den;(mult*loss).backward();value+=float(loss.detach())
   losses[f]=value if rs else None;exposure[f]=len(rs)
  torch.manual_seed(seed(f'quality_regression|123|{step}'));quality=[qr[i] for i in qs[step]];qloss=np.zeros(2)
  for s in range(0,8,4):
   b=quality[s:s+4];targets=torch.tensor([r['target'] for r in b],device='cuda')
   if arm==ARMS[0]:
    with torch.no_grad():pred=m(*data.batch(b,'A'))[:,2:];ls=F.huber_loss(pred,targets,reduction='none',delta=1.).mean(0);qloss+=ls.cpu().numpy()/2
   else:
    pred=m(*data.batch(b,'A'))[:,2:];ls=F.huber_loss(pred,targets,reduction='none',delta=1.).mean(0);(ls.sum()/2).backward();qloss+=ls.detach().cpu().numpy()/2
  norm=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.));assert np.isfinite(norm);opt.step();row=dict(step=step+1,losses=losses,quality_C_loss=float(qloss[0]),quality_D_loss=float(qloss[1]),quality_supervision_active=arm==ARMS[1],exposure=exposure,quality_candidates=8,gradient_norm=norm,seconds=time.time()-tick);history.append(row)
  with (dest/'training.jsonl').open('a') as f:f.write(__import__('json').dumps(row)+'\n')
  write(dest/'status.json',dict(step=step+1,total=4000,arm=arm,time=time.time(),latest=row))
  if step==0 or (step+1)%25==0:
   state=dict(model=m.state_dict(),optimizer=opt.state_dict(),step=step+1,history=history,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all());save(dest/'latest.pt',state);print(arm,step+1,losses,qloss.tolist(),flush=True)
  if step+1 in STEPS:
   save(dest/'checkpoints'/f'step_{step+1:06d}.pt',state);write(dest/f'cal_{step+1:06d}.json',evaluate(m,data,cal,arm))
 results={}
 for step in STEPS:
  rs=read(dest/f'cal_{step:06d}.json');tasks={f:float(np.mean([np.logaddexp(0,-r['difference']) for r in rs if r['family']==f and r['weight']>0])) for f in ['base_context','temporal','generated'] if any(r['family']==f and r['weight']>0 for r in rs)};results[str(step)]=dict(tasks=tasks,macro_NLL=float(np.mean(list(tasks.values()))))
 chosen=min(STEPS,key=lambda s:(results[str(s)]['macro_NLL'],s));write(dest/'SELECTION.json',dict(step=chosen,path=str(dest/'checkpoints'/f'step_{chosen:06d}.pt'),calibration=results,source='RM_cal only',audit_scope='future reused development diagnostic'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);main(p.parse_args().arm)
