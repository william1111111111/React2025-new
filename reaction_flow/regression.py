"""Real TRAIN integration and exact restart probe; no DEV solver selection."""
import json,time
from pathlib import Path
import torch
from .train import train,configure_flow,write
from .sampler import load_checkpoint
from .config import ROOT,FlowConfig
from .data import resources,draw
from mam_refine.train import tree_hash
from hirp.paired_data import paired_model_inputs

def main():
    configure_flow();start=time.time()
    split=train(stop=1,label='smoke_split')
    resume=train(stop=2,resume=split/'checkpoints/step_000001.pt',label='smoke_resume')
    full=ROOT/'smoke_continuous/attempt_000/checkpoints/step_000002.pt';rest=resume/'checkpoints/step_000002.pt'
    a=torch.load(full,map_location='cpu',weights_only=True);b=torch.load(rest,map_location='cpu',weights_only=True)
    error=max(float((a['model'][k]-b['model'][k]).abs().max()) for k in a['model']);assert error==0 and tree_hash(a['optimizer'])==tree_hash(b['optimizer']) and tree_hash(a['rng'])==tree_hash(b['rng'])
    # Wall times differ; all actual sampling choices/losses match.
    assert [{k:v for k,v in r.items() if k!='step_seconds'} for r in a['rows']]==[{k:v for k,v in r.items() if k!='step_seconds'} for r in b['rows']]
    cfg=FlowConfig();_,records,data=resources(cfg,'cuda:0');batch,*_=data.batch(records[0]);m,_=load_checkpoint(full,'cuda:0');inputs={k:v[:1] for k,v in paired_model_inputs(batch).items()}
    noise=draw((1,10,750,24),cfg.rollout_seed,0,'integration_probe').cuda();t0=time.time()
    p=m.sample(**inputs,sample_count=10,noise=noise,integration_steps=16);chunk=m.sample(**inputs,sample_count=10,noise=noise,integration_steps=16,candidate_chunk=3)
    chunk_error=float((p-chunk).abs().max());assert chunk_error<1e-4
    p32=m.sample(**inputs,sample_count=10,noise=noise,integration_steps=32);n=int(inputs['lengths'][0]);difference=(p[:,:,:n]-p32[:,:,:n]).abs()
    assert torch.isfinite(p).all() and torch.isfinite(p32).all()
    # No objective-quality claim for a two-update model.
    result=dict(resume_max_parameter_error=error,optimizer_equal=True,RNG_equal=True,loss_and_draw_rows_equal=True,TRAIN_source=records[0]['source_indices'][0],Euler16_32_MAE=float(difference.mean()),Euler16_32_max=float(difference.max()),candidate_chunk_max_error=chunk_error,solver='Euler16 remains fixed; TRAIN-only untrained numerical diagnostic, no DEV choice',probe_seconds=time.time()-t0,seconds=time.time()-start,full_T=750,K_probe=10,checkpoint=str(full))
    write(ROOT/'gpu_regression.json',result);print(result,flush=True)
if __name__=='__main__':main()
