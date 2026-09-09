from copy import deepcopy
import json
import numpy as np
import pytest
import torch
from hirp.phase23_cache import cached,save_cached,complete
from hirp.phase23_long import chunks,export_full
from hirp.phase23_aux import exact_decomposition
from hirp.phase22 import make_model,eval_adapter
from hirp import HiRPConfig


def test_full_identity_reuse_and_mutations(tmp_path):
    identity=dict(checkpoint='sha',global_step=2000,arm='C2',seed=123,noise='n',plan='p',source='s',paired='y',reference='q',scaler='scale',K=32,T=128,
        permutations='perm',mask='common',aggregate='session',metric_version='v23')
    path=tmp_path/'result.json';save_cached(path,identity,dict(score=.123))
    assert cached(path,identity)==dict(score=.123)
    for key in identity:
        changed=deepcopy(identity);changed[key]=str(changed[key])+'different'
        with pytest.raises(ValueError,match='identity'):cached(path,changed)
    legacy=tmp_path/'legacy.json';legacy.write_text(json.dumps(dict(checkpoint='sha',score=.123)))
    with pytest.raises(ValueError):cached(legacy,identity)
    rows=[dict(path=str(path),identity=identity)]
    assert complete(tmp_path/'completed.json',rows)=='created'
    assert complete(tmp_path/'completed.json',rows)=='reused'
    with pytest.raises(ValueError):complete(tmp_path/'completed.json',rows+rows)
    edited=json.loads(path.read_text());edited['result']['score']=9.;path.write_text(json.dumps(edited))
    with pytest.raises(ValueError,match='content'):cached(path,identity)


def test_expectation_decomposition():
    g=torch.Generator().manual_seed(231)
    support=torch.randn(7,3,generator=g,dtype=torch.float64)
    p=torch.rand(4,7,generator=g,dtype=torch.float64);p=p/p.sum(1,keepdim=True)
    w=torch.tensor([.1,.2,.3,.4],dtype=torch.float64);q=torch.rand(7,generator=g,dtype=torch.float64);q=q/q.sum()
    left,marginal,extra=exact_decomposition(p,w,q,support)
    torch.testing.assert_close(left,marginal+extra,rtol=0,atol=1e-12)
    assert extra>=0


@pytest.mark.parametrize('length',[1,128,256,750,751,1500])
def test_official_padding_and_sample_index(length):
    c=HiRPConfig(d_model=16,nhead=2,num_layers=1,dim_feedforward=32,dilations=(1,),dropout=0.)
    model=make_model(123,c).eval();noise=torch.randn(10,32)
    inputs=dict(speaker_audio=torch.randn(length,768),speaker_emotion=torch.randn(length,25),speaker_3dmm=torch.randn(length,58))
    stream,valid=chunks(inputs['speaker_audio'],length,750)
    assert int(valid.sum())==length
    if length%750==0:assert valid[-1]==0
    seen=[]
    hook=model.register_forward_pre_hook(lambda m,args,kwargs:seen.append(kwargs['noise'].detach().clone()),with_kwargs=True)
    try:a=export_full(model,**inputs,source_length=length,noise=noise,chunk_T=750,candidate_chunk=10)
    finally:hook.remove()
    b=export_full(model,**inputs,source_length=length,noise=noise,chunk_T=750,candidate_chunk=3)
    torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
    assert a.shape==(10,length,25)
    assert all(torch.equal(n,noise[None]) for n in seen)
    assert (a[:,:,:15]>=0).all() and (a[:,:,:15]<=1).all() and (a[:,:,15:17].abs()<=1).all()
    torch.testing.assert_close(a[:,:,17:].sum(-1),torch.ones(10,length))


def test_actual_identity_content_changes(tmp_path):
    from hirp.phase23_cache import eval_identity
    from hirp.phase15_audit import sha256_file
    paths={}
    for name in ('speaker/s/a','listener/s/a'):
        path=tmp_path/'facial-attributes'/f'{name}.npy';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(name.encode());paths[name]=path
    cp=tmp_path/'cp';cp.write_bytes(b'checkpoint')
    bank=tmp_path/'noise';bank.write_bytes(b'noise')
    scaler=tmp_path/'scaler';scaler.write_bytes(b'scale')
    plan_path=tmp_path/'plan';plan_path.write_text('frozen plan')
    plan=dict(data_files=[dict(path=str(p),sha256=sha256_file(p)) for p in paths.values()],normalization=[],clip_ids=['speaker/s/a'],scaler_sha256=sha256_file(scaler),permutations=[0],T=128)
    bank_info=dict(path=str(bank),sha256=sha256_file(bank))
    meta=dict(global_step=2000,arm='C1',train_config=dict(init_seed=123,lambda_group=.1),prior_mode='standard_normal',output_config={},scales={'channel':[1]*25})
    def identity(k=32):return eval_identity(cp,meta,plan_path,plan,bank_info,k,scaler,[],{'version':'test'})
    original=identity();output=tmp_path/'result.json';save_cached(output,original,{'value':1})
    assert cached(output,identity())=={'value':1}
    with pytest.raises(ValueError):cached(output,identity(64))
    for path in (bank,scaler,paths['speaker/s/a']):
        before=path.read_bytes();path.write_bytes(before+b'changed')
        with pytest.raises(ValueError):identity()
        path.write_bytes(before)
    plan_path.write_text('different plan')
    with pytest.raises(ValueError):cached(output,identity())
    plan_path.write_text('frozen plan')
    bank.write_bytes(b'new legitimate bank');bank_info['sha256']=sha256_file(bank)
    with pytest.raises(ValueError):cached(output,identity())
