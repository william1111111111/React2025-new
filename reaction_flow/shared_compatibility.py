"""Real parent forward/sample/loader compatibility, no Development scoring."""
import copy,json,torch
from .sampler import load_checkpoint as old_load
from .shared_model import upgrade,load_checkpoint
from .shared_noise import make_initial_noise,recording_bases
from .prepare_shared import ROOT,PARENT
from .train_shared import resources
from .config import FlowConfig
from .train import configure_flow,write
from hirp.paired_data import paired_model_inputs

def main():
    configure_flow();cfg=FlowConfig();_,records,data=resources(cfg,'cuda:0');b=data.batch(records[0])[0];inputs={k:v[:1] for k,v in paired_model_inputs(b).items()};local,g=recording_bases(cfg,'TRAIN-only-compatibility',750,4);local=local[None].cuda().double();g=g[None].cuda().double()
    model,_=old_load(PARENT,'cuda:0');model.double();inputs={k:v.double() if v.is_floating_point() else v for k,v in inputs.items()}
    old=model.sample(**inputs,sample_count=4,noise=local)
    new=upgrade(model,0.);same=new.sample(**inputs,sample_count=4,noise=local,global_noise=g);error=float((old-same).abs().max());assert error==0
    shared,saved=load_checkpoint(ROOT/'smoke_full/attempt_000/checkpoints/step_014002.pt','cuda:0');shared.double();a=shared.sample(**inputs,sample_count=4,noise=local,global_noise=g);c=shared.sample(**inputs,sample_count=4,noise=local,global_noise=g,candidate_chunk=1)
    err=float((a-c).abs().max());assert err<1e-10
    assert saved['initial_prior']['rho']==.05 and shared.rho==.05
    write(ROOT/'compatibility.json',dict(real_T0_rho0_error=error,G1_loaded_candidate_chunk_error=err,T=750,precision='FP64 Euler16',rho_checkpoint=.05,public_input='base local plus base global; combined exactly once'))
if __name__=='__main__':main()
