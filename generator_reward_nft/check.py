"""Real detached TRAIN endpoints and optimizer/teacher ownership acceptance."""
import copy
import torch
from reaction_flow.train import configure_flow
from reaction_flow.data import draw
from reaction_flow.flow_path import flow_path
from .common import *
from .model import Models
from .objective import nft,core

def main():
    configure_flow();torch.set_num_threads(4);torch.manual_seed(123)
    # Analytical core checks use repository implementation rather than vendor example.
    old=torch.randn(2,3,24,dtype=torch.double);target=torch.randn_like(old);mask=torch.ones(2,3,dtype=torch.bool)
    for reward in (0.,.5,1.):
        new=old.clone().requires_grad_();r=torch.full((2,),reward,dtype=torch.double)
        loss=core(new,old,target,r,mask);g=torch.autograd.grad(loss,new)[0]
        expected=2*(2*reward-1)*(old-target)/old.numel();assert torch.allclose(g,expected,atol=1e-14)
    m=Models();e=m.env.episode(read(OLD/'episodes.json')['calibration'][0]);noise=m.env.noise(e,2,'action');U,pred=m.generate(e,noise,m.old)
    assert torch.equal(pred,m.env.generate(e,noise))
    refhash=digest(m.env.generator.model);oldhash=digest(m.old);params=dict(m.student.named_parameters());opt=torch.optim.AdamW(params.values(),lr=2e-5,weight_decay=.01)
    assert all(p.requires_grad for p in params.values())
    assert all(any(name.startswith(f'blocks.{i}.') for name in params) for i in range(8))
    results=[]
    for i,rvalue in enumerate((.9,.1)):
        clean=U[i:i+1].cuda().detach();h=e['h'];mask=e['mask'];eps=draw(clean.shape,123,i,'acceptance_independent_epsilon').cuda().double();tau=clean.new_tensor([.5]);u,_=flow_path(clean,eps,tau,mask);offset=torch.tensor([e['start']],device='cuda')
        with torch.no_grad():vold=m.old(u,tau,h,mask,mask,offset)
        new=m.student(u,tau,h,mask,mask,offset,gradient_checkpointing=True);loss=nft(new,vold,u,clean,tau,clean.new_tensor([rvalue]),mask)
        before={k:p.detach().clone() for k,p in params.items()};opt.zero_grad();loss.backward()
        groups={str(b):sum(float(p.grad.square().sum()) for name,p in params.items() if name.startswith(f'blocks.{b}.') and p.grad is not None)**.5 for b in range(8)}
        assert all(v>0 for v in groups.values());assert params['output.weight'].grad.norm()>0
        norm=float(torch.nn.utils.clip_grad_norm_(params.values(),1.));opt.step();delta=sum(float((p.detach()-before[k]).square().sum()) for k,p in params.items())**.5;assert delta>0
        assert digest(m.old)==oldhash and digest(m.env.generator.model)==refhash
        assert all(p.grad is None for p in m.old.parameters()) and all(p.grad is None for p in m.env.generator.model.parameters())
        results.append(dict(r=rvalue,loss=float(loss),gradient_norm=norm,actual_delta=delta,block_gradients=groups,detached_clean=not clean.requires_grad))
    _,after=m.generate(e,noise,m.student);change=float((after-pred).square().mean().sqrt());assert change>0
    with torch.no_grad():vafter=m.student(u,tau,h,mask,mask,offset)
    vector_change=float((vafter-vold).square().mean().sqrt());assert vector_change>0
    write(OUT/'ACCEPTANCE.json',dict(passed=True,core_analytic_rewards=[0,.5,1],real_TRAIN_source=e['clip_id'],native_clean_inverse_exact=True,optimizer_names=list(params),parameter_count=sum(p.numel() for p in params.values()),full_velocity_blocks=8,frozen_source_path=True,ref_hash=refhash,old_hash=oldhash,teacher_hashes_unchanged=True,steps=results,same_noise_prediction_RMS_change=change,vector_field_RMS_change=vector_change,extra_rollouts=6,formal_training_initialized_separately=True))
    print('ACCEPTANCE passed',change,vector_change,flush=True)
if __name__=='__main__':main()
