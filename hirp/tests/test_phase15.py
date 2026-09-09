from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from hirp import HiRPConfig,HiRPNet
from hirp.losses import paired_energy_score
from hirp.paired_data import PairedReactionDataset,pair_valid_mask,paired_model_inputs
from hirp.phase1_diagnostics import prior_mode,evaluate_diagnostics
from hirp.phase15_audit import (data_provenance,verify_provenance,random_crop_risk,
                               canonical_hash,stratified_indices,sha256_file)
from hirp.phase15_metrics import (channel_metrics,aggregate_channels,trajectory_summary,
                                 real_real_reference)
from hirp.phase15_interventions import (within_session_permutation,intervention_prior,
                                       override_prior)


@pytest.fixture
def batch(inputs):
    values={key:value for key,value in inputs.items() if key!='lengths'}
    target=torch.rand(2,32,25)
    target[...,15:17]=target[...,15:17]*2-1
    target[...,17:]=target[...,17:].softmax(-1)
    return dict(**values,source_lengths=inputs['lengths'],pair_lengths=torch.tensor([27,17]),
                paired_target=target,clip_id=['a','b'],session_id=['s','s'])


def test_global_prior_only_bias_updates(model,batch):
    before={name:p.detach().clone() for name,p in model.prior.named_parameters()}
    params=sum(p.numel() for p in model.parameters())
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.1)
    optimizer.zero_grad(set_to_none=True)
    with prior_mode(model,'A_global'):
        aux=model(**paired_model_inputs(batch),return_aux=True)
        assert torch.equal(aux['latent_mu'][0],aux['latent_mu'][1])
        assert torch.equal(aux['latent_log_sigma'][0],aux['latent_log_sigma'][1])
    loss=paired_energy_score(aux['predictions'],batch['paired_target'],pair_valid_mask(batch))
    loss.backward()
    assert model.prior.mu.weight.grad is None and model.prior.log_sigma.weight.grad is None
    assert model.prior.mu.bias.grad.norm()>0 and model.prior.log_sigma.bias.grad.norm()>0
    optimizer.step()
    for name,p in model.prior.named_parameters():
        assert p.requires_grad
        if name.endswith('weight'): assert torch.equal(before[name],p)
        else: assert not torch.equal(before[name],p)
    assert params==sum(p.numel() for p in model.parameters())


def test_global_hook_exception_cleanup(model):
    with pytest.raises(RuntimeError):
        with prior_mode(model,'A_global'):
            raise RuntimeError('deliberate')
    assert all(p.requires_grad for p in model.prior.parameters())
    assert not model.prior._forward_pre_hooks


@pytest.mark.parametrize('k',[0,1])
def test_diagnostics_reject_small_k(model,batch,k):
    with pytest.raises(ValueError,match='K >= 2'):
        evaluate_diagnostics(model,batch,torch.zeros(2,k,32))
    assert not model.output_head.stochastic._forward_hooks


def test_session_selection_shuffle():
    ds=SimpleNamespace(records=[Path(f'session{i}/clip{j}.npy') for i in range(3) for j in range(6)])
    chosen=stratified_indices(ds,4,1501)
    assert chosen==stratified_indices(ds,4,1501)
    assert len(set(chosen))==12
    sessions=[ds.records[i].parent.name for i in chosen]
    permutation=within_session_permutation(sessions)
    assert len(set(permutation.tolist()))==12
    assert all(sessions[i]==sessions[j] and i!=j for i,j in enumerate(permutation.tolist()))
    assert torch.equal(permutation,within_session_permutation(sessions))


def test_mean_prior_averages_sigma_not_log():
    mu=torch.tensor([[0.,2.],[2.,4.]])
    logs=torch.tensor([[1.,1.],[4.,4.]]).log()
    mean_mu,mean_logs=intervention_prior(mu,logs,'dataset_mean')
    torch.testing.assert_close(mean_mu,torch.tensor([[1.,3.],[1.,3.]]))
    torch.testing.assert_close(mean_logs.exp(),torch.full((2,2),2.5))


def test_eval_override_alpha_zero_and_no_mutation(model,batch):
    noise=torch.randn(2,4,32)
    reference=model.sample(**paired_model_inputs(batch),noise=noise,return_aux=True)
    state={name:value.clone() for name,value in model.state_dict().items()}
    with override_prior(model,reference['latent_mu'],reference['latent_log_sigma']):
        correct=model.sample(**paired_model_inputs(batch),noise=noise)
        zero=model.sample(**paired_model_inputs(batch),noise=noise*0)
    torch.testing.assert_close(correct,reference['predictions'],rtol=0,atol=0)
    torch.testing.assert_close(zero,zero[:,:1].expand_as(zero),rtol=0,atol=0)
    assert all(torch.equal(state[name],value) for name,value in model.state_dict().items())
    model.train()
    with pytest.raises(ValueError,match='eval-only'):
        with override_prior(model,reference['latent_mu'],reference['latent_log_sigma']):pass


def test_channel_overall_equals_paired_es(model,batch):
    p=model.sample(**paired_model_inputs(batch))
    details=paired_energy_score(p,batch['paired_target'],pair_valid_mask(batch),return_details=True)
    channels=aggregate_channels(channel_metrics(p,batch['paired_target'],batch['pair_lengths']))
    for key,dkey in [('ES','loss'),('cross_distance','cross_distance'),('self_distance','self_distance')]:
        assert channels['overall'][key]==pytest.approx(details[dkey].item(),abs=1e-7)
    altered=p.clone(); target=batch['paired_target'].clone()
    for i,n in enumerate(batch['pair_lengths']):
        altered[i,:,n:]=float('nan'); target[i,n:]=float('nan')
    assert channel_metrics(p,batch['paired_target'],batch['pair_lengths'])==channel_metrics(altered,target,batch['pair_lengths'])


def test_entropy_velocity_boundaries():
    onehot=torch.zeros(2,4,8);onehot[...,0]=1
    stats=trajectory_summary(onehot,'expression')
    assert stats['expression_entropy']==0 and stats['boundary_fraction']==1
    assert stats['velocity_abs_mean']==0
    uniform=torch.full((2,4,8),1/8)
    assert trajectory_summary(uniform,'expression')['expression_entropy']==pytest.approx(np.log(8),abs=1e-6)
    va=torch.tensor([[[-1.,1.],[1.,-1.]]])
    stats=trajectory_summary(va,'VA')
    assert stats['velocity_abs_mean']==2 and stats['velocity_rms']==2 and stats['boundary_fraction']==1


def test_real_reference_is_same_session_only(batch):
    reference=real_real_reference(batch)
    assert reference['same_session_pair_count']==1
    assert reference['pairs'][0]['common_length']==17
    q=batch['paired_target']
    expected=((q[0,:17]-q[1,:17]).square().mean()+1e-8).sqrt().item()
    assert reference['aggregate']['overall']['distance']['mean']==pytest.approx(expected)
    batch['session_id']=['s','other']
    assert real_real_reference(batch)['same_session_pair_count']==0


@pytest.mark.parametrize('source,target,clip',[(11,5,4),(10,9,4),(4,0,8),(3,2,8)])
def test_random_crop_audit_exact(source,target,clip):
    starts=list(range(max(0,source-clip)+1))
    expected=sum(start>=target for start in starts)
    risk=random_crop_risk(source,target,clip)
    assert risk['zero_overlap_starts']==expected
    assert risk['probability']==expected/len(starts)


def test_provenance_detects_changed_data(tmp_path):
    norm=tmp_path/'external/FaceVerse';norm.mkdir(parents=True)
    np.save(norm/'mean_face.npy',np.zeros(58,dtype=np.float32))
    np.save(norm/'std_face.npy',np.ones(58,dtype=np.float32))
    target=None
    for folder,role,width in [('audio-features','speaker',768),('facial-attributes','speaker',25),
                              ('coefficients','speaker',58),('facial-attributes','listener',25)]:
        path=tmp_path/'data/train'/folder/role/'s'/'x.npy';path.parent.mkdir(parents=True,exist_ok=True)
        np.save(path,np.ones((4,width),dtype=np.float32))
        if role=='listener':target=path
    ds=PairedReactionDataset(tmp_path/'data','train')
    manifest=data_provenance(tmp_path,[(ds,[0])])
    assert len(manifest['data_files'])==4 and len(manifest['normalization_files'])==2
    verify_provenance(tmp_path,manifest)
    np.save(target,np.zeros((4,25),dtype=np.float32))
    with pytest.raises(ValueError,match='data hash mismatch'):verify_provenance(tmp_path,manifest)


def test_audit_checkpoint_roundtrip(model,batch,tmp_path):
    path=tmp_path/'audit.pt'
    torch.save(dict(model=model.state_dict(),config=asdict(model.config),arm='A1'),path)
    digest=sha256_file(path)
    saved=torch.load(path,map_location='cpu',weights_only=True)
    restored=HiRPNet(HiRPConfig(**saved['config'])).eval()
    restored.load_state_dict(saved['model'],strict=True)
    noise=torch.randn(2,4,32)
    torch.testing.assert_close(model.sample(**paired_model_inputs(batch),noise=noise),
                               restored.sample(**paired_model_inputs(batch),noise=noise),rtol=0,atol=0)
    assert sha256_file(path)==digest



def test_session_bootstrap_constant_difference():
    from hirp.summarize_phase15 import session_bootstrap_difference
    def records(offset):
        return dict(per_example=[dict(clip_id=str(i),session_id=str(i//2),
            channels={'overall':{'ES':float(i)+offset}}) for i in range(8)])
    result=session_bootstrap_difference(records(0),records(.125))
    assert result['session_count']==4 and result['delta']==.125
    assert result['ci95']==[.125,.125]
