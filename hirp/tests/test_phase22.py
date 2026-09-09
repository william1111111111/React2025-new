import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from hirp import HiRPConfig
from hirp.phase21 import make_model as old_model, forward_a0, OutputConfig
from hirp.phase22 import make_model, model_metadata, load_checkpoint, eval_adapter, official_adapter
from hirp.train_phase22 import TrainConfig, schedule, train, rng_state
from hirp.group_features import DescriptorScaler
from hirp.paired_data import paired_model_inputs
from hirp.train_phase1 import state_hash

CONFIG=HiRPConfig(d_model=32,nhead=4,num_layers=1,dim_feedforward=64,dropout=.1,dilations=(1,2))
SCALES=dict(channel_scale=[1.]*25,descriptor_scaler_sha256='test-scaler',descriptor_dimension=75,training_T=12)


def batch():
    g=torch.Generator().manual_seed(73)
    return dict(speaker_audio=torch.randn(4,12,768,generator=g),speaker_emotion=torch.randn(4,12,25,generator=g),
        speaker_3dmm=torch.randn(4,12,58,generator=g),source_lengths=torch.tensor([12,10,8,6]),
        pair_lengths=torch.tensor([10,9,7,5]),paired_target=torch.rand(4,12,25,generator=g))


@pytest.mark.parametrize('k',[1,4,10,32])
def test_public_paths_roundtrip_and_legacy_equivalence(tmp_path,k):
    b=batch();inputs=paired_model_inputs(b);noise=torch.randn(4,k,32)
    old=old_model(config=CONFIG,output_config=OutputConfig(head_pre_norm=True)).eval()
    new=make_model(123,CONFIG).eval();new.load_state_dict(old.state_dict())
    with torch.no_grad():
        a=forward_a0(old,**inputs,sample_count=k,noise=noise)
        p=new(**inputs,sample_count=k,noise=noise)
    torch.testing.assert_close(a,p,rtol=0,atol=0)
    path=tmp_path/'model.pt';torch.save(dict(**model_metadata(new,SCALES),model=new.state_dict()),path)
    rng=torch.get_rng_state();restored,_=load_checkpoint(path)
    assert torch.equal(rng,torch.get_rng_state())
    for q in [new.sample(**inputs,sample_count=k,noise=noise),restored.sample(**inputs,sample_count=k,noise=noise),eval_adapter(restored,**inputs,noise=noise)]:
        torch.testing.assert_close(p,q,rtol=0,atol=0)
    perm=torch.randperm(k)
    torch.testing.assert_close(new.sample(**inputs,sample_count=k,noise=noise[:,perm]),p[:,perm],atol=2e-6,rtol=2e-5)
    chunks=torch.cat([new.sample(**inputs,sample_count=min(3,k-i),noise=noise[:,i:i+3]) for i in range(0,k,3)],1)
    torch.testing.assert_close(chunks,p,atol=2e-6,rtol=2e-5)
    prefix=new.sample(**inputs,sample_count=1,noise=noise[:,:1])
    torch.testing.assert_close(prefix,p[:,:1],atol=2e-6,rtol=2e-5)
    if k==10:
        torch.testing.assert_close(official_adapter(new,**{k:v for k,v in b.items() if k.startswith('speaker_') or k=='source_lengths'},noise=noise),p,rtol=0,atol=0)


def test_historical_bare_sample_mismatch_is_real():
    b=batch();old=old_model(config=CONFIG,output_config=OutputConfig(head_pre_norm=True)).eval();noise=torch.randn(4,4,32)
    hooked=forward_a0(old,**paired_model_inputs(b),noise=noise)
    bare=old.sample(**paired_model_inputs(b),noise=noise)
    assert (hooked-bare).abs().max()>0


def test_standard_normal_does_not_call_prior_and_conditional_does():
    b=batch();noise=torch.randn(4,4,32)
    for mode in ('standard_normal','conditional_gaussian'):
        model=make_model(123,CONFIG,prior_mode=mode).eval();calls=[]
        hook=model.prior.register_forward_hook(lambda *args:calls.append(1))
        try:aux=model(**paired_model_inputs(b),noise=noise,return_aux=True)
        finally:hook.remove()
        assert bool(calls)==(mode=='conditional_gaussian')
        if mode=='standard_normal':assert torch.equal(aux['latent_z'],noise)
        else:assert not torch.equal(aux['latent_z'],noise)


def test_loader_rejects_missing_semantics(tmp_path):
    model=make_model(123,CONFIG);full=dict(**model_metadata(model,SCALES),model=model.state_dict())
    for key in ('prior_mode','output_config','config','scales','latent_dim'):
        p=tmp_path/(key+'.pt');torch.save({k:v for k,v in full.items() if k!=key},p)
        with pytest.raises(ValueError):load_checkpoint(p)


@pytest.mark.parametrize('arm',['C0','C1','C2'])
def test_exact_resume_and_step_budget(tmp_path,arm):
    cfg=TrainConfig(max_steps=6,T=12,eval_interval=3,checkpoint_interval=3,diagnostic_interval=3,fixed_checkpoints=(3,6))
    pool=SimpleNamespace(sources={'s':[0,1,2,3,4]},references={'s':['a','b','c']})
    records,noise,_=schedule(pool,cfg,32)
    scaler=DescriptorScaler(torch.zeros(75),torch.ones(75),torch.ones(75))
    g=torch.Generator().manual_seed(33);refs=(torch.rand(3,75,generator=g),torch.ones(3,75,dtype=torch.bool))
    data=SimpleNamespace(scaler=scaler,batch=lambda r:(batch(),None if arm=='C0' else refs))
    manifest=dict(model_config=asdict(CONFIG),scales=SCALES,split_hash='test-split',population=dict(source_population=5))
    a,ma,oa=train(tmp_path/'full',cfg,arm,records,noise,data,manifest,'cpu',model_config=CONFIG)
    stopped,_,_=train(tmp_path/'resume',cfg,arm,records,noise,data,manifest,'cpu',stop_after=3,model_config=CONFIG)
    assert not stopped['completed'] and stopped['actual_optimizer_steps']==3
    b,mb,ob=train(tmp_path/'resume',cfg,arm,records,noise,data,manifest,'cpu',resume=stopped['checkpoints'][-1]['path'],model_config=CONFIG)
    assert a['completed'] and b['completed'] and b['actual_optimizer_steps']==b['requested_steps']==6
    assert state_hash(ma)==state_hash(mb)
    sa=torch.load(a['checkpoints'][-1]['path'],weights_only=True);sb=torch.load(b['checkpoints'][-1]['path'],weights_only=True)
    assert sa['training_rows']==sb['training_rows']
    for key,value in oa.state_dict()['state'].items():
        for k,v in value.items():
            torch.testing.assert_close(v,ob.state_dict()['state'][key][k],rtol=0,atol=0)
    with pytest.raises(ValueError,match='schedule length'):
        train(tmp_path/'bad',replace(cfg,max_steps=2000),arm,records,noise,data,manifest,'cpu',model_config=CONFIG)


def test_long_schedule_seed_independence_and_extension():
    pool=SimpleNamespace(sources={'s':[0,1,2],'t':[3,4]},references={'s':['a','b'],'t':['c','d']})
    configs=[TrainConfig(init_seed=s,sampler_seed=s,noise_seed=s) for s in (123,42,2026)]
    runs=[schedule(pool,c,32) for c in configs]
    assert all(len(r)==len(n)==2000 for r,n,h in runs)
    assert len({h['source_hash'] for r,n,h in runs})==len({h['noise_hash'] for r,n,h in runs})==3
    short,noise,_=schedule(pool,replace(configs[0],max_steps=128),32)
    assert short==runs[0][0][:128] and torch.equal(noise,runs[0][1][:128])
    assert not torch.equal(runs[0][1][:128],runs[0][1][128:256])
    assert len({state_hash(make_model(c.init_seed,CONFIG)) for c in configs})==3


def test_real_phase21_checkpoint_equivalence():
    path=Path('runs/phase21/conditional_score_v0/checkpoints/C0.pt')
    if not path.exists():pytest.skip('actual Phase 2.1 checkpoint missing')
    model,saved=load_checkpoint(path,legacy_manifest=path.parent.parent/'training_manifest.json')
    old=old_model(config=HiRPConfig(**saved['config']),output_config=OutputConfig(**saved['output_config'])).eval()
    old.load_state_dict(saved['model']);b=batch();noise=torch.randn(4,4,32)
    with torch.no_grad():
        p=forward_a0(old,**paired_model_inputs(b),noise=noise)
        q=model.sample(**paired_model_inputs(b),noise=noise)
    torch.testing.assert_close(p,q,rtol=0,atol=0)


def test_evaluation_uses_public_sampler_and_common_masks():
    from hirp.evaluate_phase22 import evaluate_model,extra_shuffles
    from hirp.evaluate_phase21 import evaluate_model as old_evaluate
    from hirp.group_features import DescriptorScaler
    b=batch();b['clip_id']=['a','b','c','d'];b['session_id']=['s']*4
    m=make_model(123,CONFIG).eval();scaler=DescriptorScaler(torch.zeros(75),torch.ones(75),torch.ones(75))
    refs={'s':(torch.rand(3,75),torch.ones(3,75,dtype=torch.bool))};bank=torch.randn(1,4,32);perm=torch.tensor([1,2,3,0])
    r=evaluate_model(m,b,refs,scaler,scaler,bank,perm,'cpu',4)
    original=old_evaluate(m,b,refs,scaler,scaler,bank,perm,'cpu',4)
    assert r['aggregate']==original['aggregate']
    rows=extra_shuffles(r['all_predictions'],b,[dict(seed=0,donors=perm.tolist())])[0]['per_input']
    for row,i in zip(rows,range(4)):
        assert row['common_length']==min(int(b['pair_lengths'][i]),int(b['source_lengths'][perm[i]]))
        assert row['correct']==r['per_input'][i]['conditional_correct_common']['loss']
    with pytest.raises(TypeError):m(**paired_model_inputs(b),target=b['paired_target'])
    with pytest.raises(TypeError):eval_adapter(m,**paired_model_inputs(b),noise=bank.expand(4,-1,-1),session_id=b['session_id'])
