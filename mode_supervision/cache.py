"""Frozen legal source summaries and TRAIN-only paired plan normalization."""
import json,time
import torch
from reaction_flow.sampler import load_checkpoint
from reaction_flow.train import configure_flow
from .prepare import OUT,PARENT,sha,write
from .data import ModeData

def main():
    configure_flow();torch.set_num_threads(4);base,_=load_checkpoint(PARENT,'cuda:0');data=ModeData(base)
    sums=torch.zeros(96,dtype=torch.float64);squares=sums.clone();counts=sums.clone();tick=time.time()
    for i,row in enumerate(data.rows):
        data.context(i)
        y=data.aligned(i,row['clip_id'].replace('speaker/','listener/',1),True);a,m=data.plan(i,y);a=a.double().cpu();m=m.cpu()
        sums+=(a*m).sum(0);squares+=(a.square()*m).sum(0);counts+=m.sum(0)
        if (i+1)%25==0:write(OUT/'cache_status.json',dict(completed=i+1,total=len(data.rows),seconds=time.time()-tick));print('cached',i+1,flush=True)
    mean=sums/counts.clamp_min(1);std=(squares/counts.clamp_min(1)-mean.square()).clamp_min(0).sqrt().clamp_min(.001)
    write(OUT/'plan_stats.json',dict(mean=mean.tolist(),std=std.tolist(),counts=counts.tolist(),split='train',population='one real paired trajectory per source, all valid generation blocks; shared across arms; weak alternatives not used to fit scale',parent_sha256=sha(PARENT)))
    write(OUT/'cache_status.json',dict(completed=len(data.rows),total=len(data.rows),complete=True,seconds=time.time()-tick,peak_gpu_bytes=torch.cuda.max_memory_allocated()))
if __name__=='__main__':main()
