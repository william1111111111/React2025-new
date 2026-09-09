import inspect
import pytest
import torch
from hirp import HiRPConfig
from hirp.model.temporal_decoder import FiLM
from hirp.phase21 import (OutputConfig, make_model, forward_a0, conditional_score,
                          losses_for_batch, CHANNEL_SCALE)
from hirp.phase21_diagnostics import output_path_diagnostic, component_gradients
from hirp.group_features import DescriptorScaler
from hirp.paired_data import paired_model_inputs, pair_valid_mask
from hirp.losses import paired_energy_score
from hirp.toy_phase21 import run_toy


def setup_batch():
    batch=dict(speaker_audio=torch.randn(4,12,768),speaker_emotion=torch.randn(4,12,25),
               speaker_3dmm=torch.randn(4,12,58),source_lengths=torch.tensor([12,9,8,7]),
               paired_target=torch.rand(4,12,25),pair_lengths=torch.tensor([8,7,6,4]))
    scaler=DescriptorScaler(torch.zeros(75),torch.ones(75),torch.ones(75))
    return batch,scaler,torch.randn(4,4,32)


def small_model(pre_norm=False):
    return make_model(output_config=OutputConfig(head_pre_norm=pre_norm),
                      config=HiRPConfig(d_model=32,nhead=4,num_layers=1,dim_feedforward=64,dropout=0.,dilations=(1,2))).eval()


def test_all_samples_receive_conditional_gradient():
    batch,_,_=setup_batch()
    p=torch.randn(4,4,12,25,requires_grad=True)
    d=conditional_score(p,batch)
    d['loss'].backward()
    assert (p.grad.square().sum((2,3))>0).all()
    assert p.grad[0,:,8:].count_nonzero()==0
    for k in range(4):
        changed=p.detach().clone();changed[:,k,:4]+=1
        assert not torch.equal(conditional_score(changed,batch)['loss'],d['loss'])


def test_common_conditional_objective_and_generator_isolation():
    batch,scaler,noise=setup_batch();model=small_model()
    refs=(torch.rand(3,75),torch.ones(3,75,dtype=torch.bool))
    seen=[]
    hook=model.register_forward_pre_hook(lambda m,args,kwargs:seen.append(set(kwargs)),with_kwargs=True)
    try:
        results=[losses_for_batch(model,batch,noise,scaler,arm,None if arm=='C0' else refs)
                 for arm in ('C0','C1','C2')]
    finally:hook.remove()
    for result in results:
        assert torch.equal(result['conditional']['loss'],results[0]['conditional']['loss'])
        exact=paired_energy_score(result['predictions'],batch['paired_target'],pair_valid_mask(batch),
                                  channel_scale=CHANNEL_SCALE)
        assert torch.equal(exact,result['conditional']['loss'])
    assert torch.equal(results[0]['total'],results[0]['conditional']['loss'])
    assert len(seen)==3 and all('target' not in str(keys) and 'ref' not in str(keys) for keys in seen)
    for fn in (model.forward,forward_a0):
        assert not any('target' in k or 'ref' in k for k in inspect.signature(fn).parameters)
    with pytest.raises(ValueError,match='C0'):
        losses_for_batch(model,batch,noise,scaler,'C0',refs)
    with pytest.raises(ValueError,match='K >= 2'):
        losses_for_batch(model,batch,noise[:,:1],scaler,'C0')


def test_source_pair_padding_and_length_separation():
    batch,_,noise=setup_batch();model=small_model(True)
    p=forward_a0(model,**paired_model_inputs(batch),noise=noise)
    changed={k:v.clone() for k,v in batch.items()}
    for i in range(4):
        changed['paired_target'][i,int(batch['pair_lengths'][i]):]=999
        for key in ('speaker_audio','speaker_emotion','speaker_3dmm'):
            changed[key][i,int(batch['source_lengths'][i]):]=-999
    q=forward_a0(model,**paired_model_inputs(changed),noise=noise)
    torch.testing.assert_close(p,q,rtol=0,atol=0)
    assert torch.equal(conditional_score(p,batch)['loss'],conditional_score(q,changed)['loss'])
    seen=[]
    hook=model.encoder.register_forward_pre_hook(lambda m,args:seen.append(args[1]))
    try:forward_a0(model,**paired_model_inputs(batch),noise=noise)
    finally:hook.remove()
    assert torch.equal(seen[0].sum(1),batch['source_lengths'])
    assert not torch.equal(seen[0].sum(1),batch['pair_lengths'])


@pytest.mark.parametrize('pre_norm',[False,True])
def test_new_head_sampling_contract(pre_norm):
    batch,_,_=setup_batch();model=small_model(pre_norm)
    noise=torch.randn(4,10,32)
    p=forward_a0(model,**paired_model_inputs(batch),sample_count=10,noise=noise)
    q=forward_a0(model,**paired_model_inputs(batch),sample_count=4,noise=noise[:,:4])
    torch.testing.assert_close(p[:,:4],q,atol=2e-6,rtol=2e-5)
    perm=torch.randperm(10)
    r=forward_a0(model,**paired_model_inputs(batch),sample_count=10,noise=noise[:,perm])
    torch.testing.assert_close(r,p[:,perm],atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(p,forward_a0(model,**paired_model_inputs(batch),sample_count=10,noise=noise),rtol=0,atol=0)
    mask=torch.arange(12)[None]<batch['source_lengths'][:,None]
    valid=mask[:,None].expand(4,10,12)
    assert p.shape==(4,10,12,25) and (p[~valid]==0).all()
    assert (p[valid][:,:15]>=0).all() and (p[valid][:,:15]<=1).all()
    assert (p[valid][:,15:17].abs()<=1).all()
    torch.testing.assert_close(p[...,17:].sum(-1)[valid],torch.ones_like(p[...,0][valid]))


def test_frozen_prior_nonzero_noise_path_and_diagnostic_gradients():
    batch,_,noise=setup_batch();model=small_model(True)
    before={k:v.clone() for k,v in model.prior.state_dict().items()}
    diag=output_path_diagnostic(model,batch,noise)
    assert all(v['finite'] and v['vjp_norm']>0 for v in diag['noise_jacobian'].values())
    assert all(p.grad is None for p in model.parameters())
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    epsilon=noise.clone().requires_grad_()
    p=forward_a0(model,**paired_model_inputs(batch),noise=epsilon)
    conditional_score(p,batch)['loss'].backward()
    assert torch.isfinite(epsilon.grad).all() and epsilon.grad.norm()>0
    assert all(p.grad is None for p in model.prior.parameters())
    optimizer.step()
    assert all(torch.equal(before[k],v) for k,v in model.prior.state_dict().items())


def test_zero_film_blocks_initial_prior_gradient_regression():
    batch,_,noise=setup_batch()
    norms=[]
    for zero in (False,True):
        model=small_model();model.prior.requires_grad_(True)
        if zero:
            with torch.no_grad():
                for module in model.modules():
                    if isinstance(module,FiLM):module.projection.weight.zero_();module.projection.bias.zero_()
        pred=model(**paired_model_inputs(batch),noise=noise)
        conditional_score(pred,batch)['loss'].backward()
        grad=model.prior.log_sigma.weight.grad
        assert torch.isfinite(grad).all()
        norms.append(grad.norm().item())
    assert norms[0]>0 and norms[1]==0


def test_gradient_cosine_finite():
    batch,scaler,noise=setup_batch();model=small_model(True)
    refs=(torch.rand(3,75),torch.ones(3,75,dtype=torch.bool))
    result=losses_for_batch(model,batch,noise,scaler,'C2',refs)
    audit=component_gradients(model,result['conditional']['loss'],result['group']['loss'],.1)
    assert audit['finite'] and -1.00001<=audit['cosine']<=1.00001
    assert audit['conditional']['prior']==audit['group']['prior']==0


@pytest.mark.parametrize('device',['cpu','cuda:2'])
@pytest.mark.parametrize('mode',['train','eval'])
def test_amp_explicit_noise_prior_dtype_float32_distance(device,mode):
    if device.startswith('cuda') and (not torch.cuda.is_available() or torch.cuda.device_count()<3):
        pytest.skip('CUDA 2 unavailable')
    batch,_,noise=setup_batch();model=small_model(True).to(device)
    model.train(mode=='train')
    batch={k:v.to(device) for k,v in batch.items()};noise=noise.to(device).requires_grad_()
    with torch.autocast(device_type=device.split(':')[0],dtype=torch.bfloat16):
        aux=forward_a0(model,**paired_model_inputs(batch),noise=noise,return_aux=True)
        score=conditional_score(aux['predictions'],batch)
    assert aux['noise'].dtype==aux['latent_mu'].dtype==torch.bfloat16
    assert torch.equal(aux['noise'],noise.to(torch.bfloat16))
    assert score['loss'].dtype==torch.float32
    score['loss'].backward()
    assert torch.isfinite(noise.grad).all() and noise.grad.norm()>0


def test_toy_recovers_from_nonoptimal_initialization():
    result=run_toy(steps=300,sample_count=8192)
    for row in result['rows']:
        assert row['initial']!=[.1,.9]
        assert row['marginal_probability_error']<.02
        if row['arm'] in ('C0','C2'):assert row['conditional_probability_mae']<.02
        else:assert .02<row['conditional_probability_mae']<.06



def test_optional_head_shared_initialization_and_small_projection():
    from hirp import HiRPNet
    torch.manual_seed(123);original=HiRPNet()
    old=make_model();normalized=make_model(output_config=OutputConfig(head_pre_norm=True))
    small=make_model(output_config=OutputConfig(projection_init='small'))
    for name,value in original.state_dict().items():
        assert torch.equal(value,old.state_dict()[name])
        assert torch.equal(value,normalized.state_dict()[name])
        if not name.startswith('output_head.'):
            assert torch.equal(value,small.state_dict()[name])
    assert 0<small.output_head.stochastic.weight.std()<.02
    assert all(m.projection.weight.count_nonzero()>0 for m in small.modules() if isinstance(m,FiLM))


def test_runner_preflight_and_matched_training(tmp_path,monkeypatch):
    import json
    from hirp import train_phase21 as runner
    batch,scaler,noise=setup_batch()
    sources={i:{k:v[i] for k,v in batch.items()} for i in range(4)}
    records=[dict(step=i+1,session_id='s',source_indices=[0,1,2,3],reference_ids=['r0','r1']) for i in range(8)]
    refs={name:(torch.rand(75),torch.ones(75,dtype=torch.bool)) for name in ('r0','r1')}
    data=(records,noise[None].expand(8,-1,-1,-1).clone(),scaler,sources,refs)
    monkeypatch.setattr(runner,'load_data',lambda *args:data)
    monkeypatch.setattr(runner,'make_model',lambda *args:small_model(True))
    (tmp_path/'smoke_summary.json').write_text(json.dumps({'selected':'pre_norm_head','output_config':{'head_pre_norm':True}}))
    (tmp_path/'preparation_manifest.json').write_text('{}')
    (tmp_path/'descriptor_stats.json').write_text('{}')
    runner.preflight(tmp_path,'cpu')
    audit=json.loads((tmp_path/'gradient_preflight.json').read_text())
    assert len(audit['rows'])==16 and audit['selected_lambda']>0
    results={arm:runner.train(tmp_path,arm,arm,OutputConfig(head_pre_norm=True),audit['selected_lambda'],
                             2,data,'cpu')[0] for arm in ('C0','C1','C2')}
    for key in ('initialization_hash','noise_hash','source_occurrence_hash','session_schedule_hash','scaler_hash','channel_scale_hash'):
        assert len({r[key] for r in results.values()})==1
    assert results['C0']['population_reference_lookups']==0
    assert results['C1']['reference_selection_hash']==results['C2']['reference_selection_hash']


@pytest.mark.parametrize('device',['cpu','cuda:2'])
def test_amp_no_grad_evaluation(device):
    if device == 'cpu' and torch.__version__.startswith('2.1.'):
        pytest.xfail('PyTorch 2.1 CPU fused Transformer eval/no_grad ignores CPU autocast; not supported or used by this float32 pilot')
    if device.startswith('cuda') and (not torch.cuda.is_available() or torch.cuda.device_count()<3):
        pytest.skip('CUDA 2 unavailable')
    batch,_,noise=setup_batch();batch={k:v.to(device) for k,v in batch.items()}
    model=small_model(True).to(device).eval()
    with torch.no_grad(),torch.autocast(device_type=device.split(':')[0],dtype=torch.bfloat16):
        aux=forward_a0(model,**paired_model_inputs(batch),noise=noise.to(device),return_aux=True)
        score=conditional_score(aux['predictions'],batch)
    assert score['loss'].dtype==torch.float32 and torch.isfinite(score['loss'])
    assert aux['noise'].dtype==aux['latent_mu'].dtype==torch.bfloat16
