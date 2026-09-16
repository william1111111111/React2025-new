"""MAM-style independent chunk batching, unchanged native FP64 Euler16."""
import torch
import torch.nn.functional as F
from generator_reward_nft.public import Generator
class BatchedGenerator(Generator):
 @torch.no_grad()
 def sample(self,streams,pts,events_by_block,noise):
  hs,ms,_=self.features(streams,pts,events_by_block);predictions=[]
  for first in range(0,len(hs),2):
   indices=list(range(first,min(first+2,len(hs))));starts=[i*750 for i in indices];lengths=[min(750,len(pts)-s) for s in starts]
   h=torch.cat([hs[i] for i in indices]);mask=torch.cat([ms[i] for i in indices]);z=torch.stack([F.pad(noise[:,s:s+n],(0,0,0,750-n)) for s,n in zip(starts,lengths)])
   y=self.model.base.rollout(h,mask,z,16,torch.tensor(starts,device=noise.device))
   predictions.extend([y[j,:,:n] for j,n in enumerate(lengths)])
  return torch.cat(predictions,1)
