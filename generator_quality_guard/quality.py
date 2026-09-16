"""Native hard-DTW active-path local gradient and official-style CCC."""
import numpy as np
import torch
import torch.nn.functional as F
from tslearn.metrics import dtw_path

def ccc(pred,target):
 n=min(len(pred),len(target));assert n>=2
 x=pred[:n];y=target[:n].to(x)
 # Algebraic CCC uses population covariance; avoids sqrt(0) derivatives for constants.
 # Official NumPy FP32 reduction roundoff is bounded by preflight tests.
 xd=x.double();yd=y.double();mx=xd.mean(0);my=yd.mean(0)
 cov=((xd-mx)*(yd-my)).mean(0);vx=(xd-mx).square().mean(0);vy=(yd-my).square().mean(0)
 return (2*cov/(vx+vy+(mx-my).square()+1e-8)).mean()

def active_dtw(pred,target):
 assert len(pred)>0 and len(target)>0
 path,value=dtw_path(pred.detach().cpu().numpy(),target.detach().cpu().numpy())
 ij=torch.tensor(path,device=pred.device,dtype=torch.long);difference=pred.double()[ij[:,0]]-target.to(pred.device).double()[ij[:,1]]
 distance=torch.linalg.vector_norm(difference)
 return distance,float(value)

def scores(predictions,targets):
 cs=[];ds=[]
 for pred in predictions:
  c=[];d=[]
  for target in targets:
   target=target.to(pred);n=min(len(pred),len(target));assert n>=2
   c.append(ccc(pred[:n],target[:n]));parts=[active_dtw(pred[:n,a:b],target[:n,a:b])[0]*w for a,b,w in [(0,15,1/15),(15,17,1),(17,25,1/8)]];d.append(sum(parts))
  cs.append(torch.stack(c));ds.append(torch.stack(d))
 C=torch.stack(cs);D=torch.stack(ds)
 return C.max(1).values.mean(),D.min(1).values.mean(),C,D

def rollout_student_with_grad(models,e,noise):
 outputs=[];n=e['n'];mask=e['mask'];h=e['h'];offset=torch.tensor([e['start']],device='cuda')
 for k in range(len(noise)):
  u=F.pad(noise[k:k+1,e['start']:e['start']+n],(0,0,0,750-n)).detach().masked_fill(~mask[...,None],0)
  for step in range(16):
   tau=u.new_full((1,),step/16)
   u=(u+models.student(u,tau,h,mask,mask,offset,gradient_checkpointing=True)/16).masked_fill(~mask[...,None],0)
  outputs.append(models.env.generator.model.base.transform.inverse(u)[0,:n].float())
 return outputs

def hinges(c,d,c0,d0,scales):
 return torch.relu((c0-c)/scales['c']).square(),torch.relu((d-d0)/scales['d']).square()
