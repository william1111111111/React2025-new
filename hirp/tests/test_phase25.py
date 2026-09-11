"""Temporal support is measured before AdamW; no weight-decay inference."""
from dataclasses import asdict
import json
import torch
import pytest
from hirp import HiRPConfig
from hirp.model.temporal_decoder import TemporalBlock
from hirp.phase22 import make_model,load_checkpoint,model_metadata,eval_adapter

@pytest.mark.parametrize('length',[128,750])
@pytest.mark.parametrize('kind',['block','hirp'])
def test_dilation_data_gradient(length,kind,tmp_path):
    config=HiRPConfig()
    if kind=='block':
        model=TemporalBlock(config,128)
        x=torch.randn(1,length,256);z=torch.randn(1,32);mask=torch.ones(1,length,dtype=torch.bool)
        y=model(x,z,mask);weight=model.depthwise.weight
    else:
        model=make_model(123).eval()
        inputs=dict(speaker_audio=torch.randn(1,length,768),speaker_emotion=torch.randn(1,length,25),speaker_3dmm=torch.randn(1,length,58),lengths=torch.tensor([length]))
        noise=torch.randn(1,2,32)
        y=model(**inputs,sample_count=2,noise=noise);weight=model.decoder.blocks[-1].depthwise.weight
    # Nonuniform linear data objective avoids cancellation, without optimizer/decay.
    (y*torch.randn_like(y)).sum().backward()
    norms=weight.grad.abs().sum((0,1))
    assert torch.isfinite(norms).all() and norms[1]>0
    if length==128:assert norms[0]==0 and norms[2]==0
    else:assert norms[0]>0 and norms[2]>0
    print(json.dumps(dict(kind=kind,T=length,data_gradient_l1=norms.tolist())))

def test_long_mask_sample_loader_prefix(tmp_path):
    model=make_model(123).eval();t=750
    inputs=dict(speaker_audio=torch.randn(1,t,768),speaker_emotion=torch.randn(1,t,25),speaker_3dmm=torch.randn(1,t,58),lengths=torch.tensor([431]))
    noise=torch.randn(1,4,32)
    with torch.no_grad():p=model(**inputs,sample_count=4,noise=noise)
    torch.testing.assert_close(p,model.sample(**inputs,sample_count=4,noise=noise),rtol=0,atol=0)
    torch.testing.assert_close(p[:,:1],model.sample(**inputs,sample_count=1,noise=noise[:,:1]),rtol=2e-5,atol=2e-6)
    changed={k:v.clone() for k,v in inputs.items()}
    for k in ('speaker_audio','speaker_emotion','speaker_3dmm'):changed[k][:,431:]=12345
    torch.testing.assert_close(p,model.sample(**changed,sample_count=4,noise=noise),rtol=0,atol=0)
    assert (p[:,:,431:]==0).all()
    scales=dict(channel_scale=[1.]*25,descriptor_scaler_sha256='test',descriptor_dimension=75,training_T=750)
    path=tmp_path/'model.pt';torch.save(dict(**model_metadata(model,scales),model=model.state_dict()),path)
    restored,_=load_checkpoint(path)
    torch.testing.assert_close(p,eval_adapter(restored,**inputs,noise=noise),rtol=0,atol=0)

def test_temporal_block_mask_before_after():
    block=TemporalBlock(HiRPConfig(),128)
    x=torch.randn(1,750,256);z=torch.randn(1,32);mask=torch.arange(750)[None]<450
    y=block(x,z,mask);x[:,450:]=12345
    torch.testing.assert_close(y,block(x,z,mask),rtol=0,atol=0)
    assert (y[:,450:]==0).all()

def test_padded_long_tensor_does_not_fake_valid_support():
    block=TemporalBlock(HiRPConfig(),128)
    x=torch.randn(1,750,256);z=torch.randn(1,32);mask=torch.arange(750)[None]<128
    y=block(x,z,mask);(y*torch.randn_like(y)).sum().backward()
    norms=block.depthwise.weight.grad.abs().sum((0,1))
    assert norms[0]==0 and norms[2]==0 and norms[1]>0

@pytest.mark.parametrize('arm',['C0','C1','C2'])
def test_occurrence_resume(tmp_path,arm):
    from types import SimpleNamespace
    from hirp.train_phase25 import TrainConfig,schedule,train
    from hirp.group_features import DescriptorScaler
    from hirp.train_phase1 import state_hash
    config=HiRPConfig(d_model=32,nhead=4,num_layers=1,dim_feedforward=64,dropout=.1,dilations=(1,128))
    cfg=TrainConfig(max_steps=4,T=260,diagnostic_interval=4,checkpoint_interval=2,eval_interval=2,fixed_checkpoints=(2,4))
    pool=SimpleNamespace(sources={'s':[0,1,2,3]},references={'s':['a','b','c']})
    records,noise,_=schedule(pool,cfg,32)
    scaler=DescriptorScaler(torch.zeros(75),torch.ones(75),torch.ones(75))
    def batch(record):
        g=torch.Generator().manual_seed(record['crop_occurrences'][0])
        b=dict(speaker_audio=torch.randn(4,260,768,generator=g),speaker_emotion=torch.randn(4,260,25,generator=g),speaker_3dmm=torch.randn(4,260,58,generator=g),source_lengths=torch.tensor([260,250,200,190]),pair_lengths=torch.tensor([250,240,190,180]),paired_target=torch.rand(4,260,25,generator=g),crop_start=torch.tensor(record['crop_occurrences']))
        refs=None if arm=='C0' else (torch.rand(3,75,generator=g),torch.ones(3,75,dtype=torch.bool))
        return b,refs
    data=SimpleNamespace(scaler=scaler,batch=batch)
    manifest=dict(model_config=asdict(config),scales=dict(channel_scale=[1.]*25,descriptor_scaler_sha256='test',descriptor_dimension=75,training_T=260),split_hash='test',population=dict(source_population=4))
    a,ma,oa=train(tmp_path/'full',cfg,arm,records,noise,data,manifest,'cpu',model_config=config)
    first,_,_=train(tmp_path/'resume',cfg,arm,records,noise,data,manifest,'cpu',stop_after=2,model_config=config)
    b,mb,ob=train(tmp_path/'resume',cfg,arm,records,noise,data,manifest,'cpu',resume=first['checkpoints'][-1]['path'],model_config=config)
    assert state_hash(ma)==state_hash(mb)
    assert a['actual_optimizer_steps']==b['actual_optimizer_steps']==4
    aa=torch.load(a['checkpoints'][-1]['path'],weights_only=True);bb=torch.load(b['checkpoints'][-1]['path'],weights_only=True)
    assert aa['training_rows']==bb['training_rows']
    for key,value in oa.state_dict()['state'].items():
        for k,v in value.items():torch.testing.assert_close(v,ob.state_dict()['state'][key][k],rtol=0,atol=0)
    changed=[dict(r) for r in records];changed[0]['crop_seed']+=1
    with pytest.raises(ValueError,match='prefix mismatch'):
        train(tmp_path/'bad',cfg,arm,changed,noise,data,manifest,'cpu',resume=first['checkpoints'][-1]['path'],model_config=config)
