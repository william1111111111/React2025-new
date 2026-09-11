import importlib.util
import numpy as np
import torch
from mam_target.model import make_model,metadata,load_checkpoint
from mam_target.losses import ccc25,soft_dtw,divergence,pair_costs,objective
from hirp.config import HiRPConfig

def test_ccc_official():
    spec=importlib.util.spec_from_file_location('official','/home/zhengshiyi/react2025/framework/metrics/FRC.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    g=torch.Generator().manual_seed(1);p=torch.randn(50,25,generator=g,dtype=torch.double);y=torch.randn(50,25,generator=g,dtype=torch.double)
    p[:,0]=y[:,0]=0;y[:,1]=1
    assert abs(float(ccc25(p,y))-m.concordance_correlation_coefficient(y.numpy(),p.numpy())[0])<1e-12

def test_sdtw_grad():
    x=torch.randn(2,4,3,dtype=torch.double,requires_grad=True);y=torch.randn(2,5,3,dtype=torch.double)
    assert torch.autograd.gradcheck(lambda z:divergence(z,y),(x,),atol=1e-4)
    torch.testing.assert_close(divergence(x,x),torch.zeros(2,dtype=torch.double),atol=1e-12,rtol=0)

def test_head_range_and_contract(tmp_path):
    m=make_model(config=HiRPConfig(d_model=16,nhead=4,num_layers=1,dim_feedforward=32,dropout=0,dilations=(1,2))).eval()
    h=torch.zeros(1,8,16);d=torch.randn(1,2,8,16)
    with torch.no_grad():
        m.output_head.base.weight.zero_();m.output_head.base.bias.zero_();m.output_head.stochastic.weight.zero_();m.output_head.stochastic.bias.zero_()
        m.output_head.stochastic.weight[:15,0]=10
        d[:,0,:,0]=-10;d[:,1,:,0]=10
    y=m.output_head(h,d,torch.ones(1,8,dtype=torch.bool));assert (y[0,1,:,:15]-y[0,0,:,:15]).min()>.95
    inp=[torch.randn(1,8,c) for c in (768,25,58)];length=torch.tensor([6]);noise=torch.randn(1,4,32,requires_grad=True)
    y=m(*inp,length,4,noise);torch.testing.assert_close(y[:,:2],m.sample(*inp,length,2,noise[:,:2]))
    assert y[:,:,6:].abs().sum()==0
    v=torch.autograd.grad(y[...,:15].sum(),noise)[0];assert torch.isfinite(v).all() and v.abs().sum()>0
    path=tmp_path/'m.pt';torch.save(dict(**metadata(m),model=m.state_dict()),path);loaded,_=load_checkpoint(path)
    torch.testing.assert_close(y,loaded.sample(*inp,length,4,noise))

def test_all_samples_and_padding():
    pred=torch.rand(1,4,8,25,requires_grad=True);targets=torch.rand(1,4,8,25)
    batch=dict(source_lengths=torch.tensor([6]),pair_lengths=torch.tensor([5]),paired_target=targets[:,0])
    lengths=torch.tensor([[5,6,6,6]]);w=dict(a=1,b=.1,beta=.25,gamma=.25,eta=.5,grid=4)
    a=objective(pred,batch,targets,lengths,w)['loss'];g=torch.autograd.grad(a,pred)[0]
    assert (g[:,:,:6].abs().sum((2,3))>0).all() and g[:,:,6:].abs().sum()==0
    q=pred.detach().clone();q[:,:,6:]=100;z=targets.clone();z[:,:,6:]=-100
    torch.testing.assert_close(a,objective(q,batch,z,lengths,w)['loss'])

def test_sdtw_reference_and_grid():
    from tslearn.metrics import soft_dtw as reference
    from mam_target.losses import grid
    x=torch.randn(2,5,3,dtype=torch.double);y=torch.randn(2,7,3,dtype=torch.double)
    actual=soft_dtw(x,y)
    expected=torch.tensor([reference(x[i].numpy(),y[i].numpy(),gamma=.1) for i in range(2)],dtype=torch.double)
    torch.testing.assert_close(actual,expected,atol=1e-10,rtol=0)
    old=torch.nn.functional.interpolate(x.transpose(-1,-2),size=32,mode='linear',align_corners=True).transpose(-1,-2)
    torch.testing.assert_close(grid(x),old,atol=1e-12,rtol=0)

def test_permutation_target_isolation():
    import inspect
    from mam_target.model import TaskHiRP
    assert not any(k in inspect.signature(TaskHiRP.forward).parameters for k in ['target','targets','session_id','reference'])
    m=make_model(config=HiRPConfig(d_model=16,nhead=4,num_layers=1,dim_feedforward=32,dropout=0,dilations=(1,2))).eval()
    x=[torch.randn(1,9,c) for c in (768,25,58)];n=torch.tensor([7]);z=torch.randn(1,4,32);order=[3,1,0,2]
    p=m.sample(*x,n,4,z);q=m.sample(*x,n,4,z[:,order]);torch.testing.assert_close(q,p[:,order])
    for t in x:t[:,7:]=100
    torch.testing.assert_close(p,m.sample(*x,n,4,z))
