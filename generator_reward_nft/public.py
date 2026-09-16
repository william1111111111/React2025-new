"""Native source-only P2 path with a post-trained full velocity; no noise policy."""
import torch
from reward_policy.generator import Generator as ParentGenerator
class Generator(ParentGenerator):
    @torch.no_grad()
    def sample(self,streams,pts,events_by_block,noise):
        hs,ms,_=self.features(streams,pts,events_by_block)
        return torch.cat([self.block(h,m,noise,s,min(750,len(pts)-s)) for h,m,s in zip(hs,ms,range(0,len(pts),750))],1)
