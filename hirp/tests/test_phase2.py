import inspect,json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import default_collate
from hirp import HiRPNet
from hirp.group_features import reaction_descriptor,DescriptorScaler,fit_descriptor_scaler
from hirp.group_scores import (descriptor_distances,descriptor_es,b2_score,b3_score,b4_score,
                               session_mixture_score,toy_hierarchy)
from hirp.session_data import SessionPopulation,make_schedule
from hirp.phase15_audit import sha256_file
from hirp.phase1_diagnostics import prior_mode
from hirp.train_phase1 import state_hash
from hirp.train_phase2 import losses_for_batch,seeded_model,prepare_step


@pytest.fixture
def population(tmp_path):
    norm=tmp_path/'external/FaceVerse';norm.mkdir(parents=True)
    np.save(norm/'mean_face.npy',np.zeros(58,dtype=np.float32));np.save(norm/'std_face.npy',np.ones(58,dtype=np.float32))
    rng=np.random.default_rng(2)
    for split in ('train','val'):
        for session,count in [('s0',3),('s1',1)]:
            for i in range(count):
                for folder,role,width in [('audio-features','speaker',768),('facial-attributes','speaker',25),
                    ('coefficients','speaker',58),('facial-attributes','listener',25)]:
                    path=tmp_path/'data'/split/folder/role/session/f'{split}_{i}.npy'
                    path.parent.mkdir(parents=True,exist_ok=True)
                    shape=(14+i,25) if role=='listener' else (8+i,width)
                    np.save(path,rng.normal(size=shape).astype(np.float32))
    return SessionPopulation(tmp_path/'data','train',4),SessionPopulation(tmp_path/'data','val',4)


def test_pool_split_isolation(population):
    train,val=population
    assert all('/train/' in str(p) for p in train.reference_paths.values())
    assert all('/val/' in str(p) for p in val.reference_paths.values())
    assert not set(train.reference_paths)&set(val.reference_paths)
    with pytest.raises(KeyError):train.load_reference(next(iter(val.reference_paths)))


def test_reference_unique_mass(population):
    train,_=population
    original=next(iter(train.reference_paths.values()))
    (original.parent/'alias.npy').symlink_to(original)
    rebuilt=SessionPopulation(train.dataset.root,'train',4)
    assert len(rebuilt.reference_paths)==len(train.reference_paths)==4
    records,_,_=make_schedule(rebuilt,32)
    for row in records:
        assert len(row['reference_ids'])==len(set(row['reference_ids']))==len(rebuilt.references[row['session_id']])


def test_source_reference_independent_crop(population):
    train,_=population
    source=train.dataset[train.sources['s0'][0]]
    ref=train.load_reference('listener/s0/train_0')
    assert source['crop_start'].item()==2 and ref['crop_start']==5
    raw=torch.from_numpy(np.load(train.reference_paths['listener/s0/train_0']))
    torch.testing.assert_close(source['paired_target'],raw[2:6])
    torch.testing.assert_close(ref['reaction'],raw[5:9])


def test_phi_sample_permutation_determinism():
    x=torch.randn(3,8,25);lengths=torch.tensor([8,6,2]);order=torch.tensor([2,0,1])
    a,am=reaction_descriptor(x,lengths);b,bm=reaction_descriptor(x[order],lengths[order])
    assert torch.equal(a[order],b) and torch.equal(am[order],bm)
    assert torch.equal(a,reaction_descriptor(x,lengths)[0])


def test_phi_padding_and_gradients():
    x=torch.randn(2,8,25,requires_grad=True);lengths=torch.tensor([8,3])
    feature,mask=reaction_descriptor(x,lengths)
    changed=x.detach().clone();changed[1,3:]=float('nan')
    torch.testing.assert_close(feature,reaction_descriptor(changed,lengths)[0],atol=0,rtol=0)
    feature.sum().backward();assert not x.grad[1,3:].any() and torch.isfinite(x.grad).all()


def test_phi_single_frame_velocity_validity():
    x=torch.ones(1,3,25,requires_grad=True)
    feature,mask=reaction_descriptor(x,torch.tensor([1]))
    assert mask[0,:50].all() and not mask[0,50:].any()
    other=feature.detach().clone();other[:,50:]=1000
    d=descriptor_distances(feature,mask,other,torch.ones_like(mask))
    assert d.item()==pytest.approx(1e-4)
    feature.sum().backward();assert torch.isfinite(x.grad).all()


def test_phi_manual_values():
    x=torch.tensor([1.,3.,5.])[None,:,None].expand(1,3,25)
    feature,_=reaction_descriptor(x,torch.tensor([3]))
    torch.testing.assert_close(feature[0,:25],torch.full((25,),3.))
    torch.testing.assert_close(feature[0,25:50],torch.full((25,),(8/3)**.5))
    torch.testing.assert_close(feature[0,50:],torch.full((25,),2.))


def test_scaler_train_only_and_frozen(population,tmp_path):
    train,val=population
    path=tmp_path/'stats.json';scaler=fit_descriptor_scaler(train,path)
    with pytest.raises(ValueError,match='TRAIN'):fit_descriptor_scaler(val,tmp_path/'bad.json')
    before={k:v.clone() for k,v in scaler.state_dict().items()}
    x=torch.randn(2,75,requires_grad=True);values,_=scaler(x,torch.ones_like(x,dtype=torch.bool));values.sum().backward()
    assert not list(scaler.parameters()) and all(not b.requires_grad and b.grad is None for b in scaler.buffers())
    assert all(torch.equal(before[k],v) for k,v in scaler.state_dict().items())
    for p in val.reference_paths.values():np.save(p,np.full((14,25),999.,dtype=np.float32))
    path2=tmp_path/'stats2.json';fit_descriptor_scaler(train,path2)
    assert sha256_file(path)==sha256_file(path2)


def test_b2_hand_calculation():
    p=torch.tensor([[[0.],[2.]]]);v=torch.ones_like(p,dtype=torch.bool)
    result=b2_score(p,v,torch.ones(1,1),torch.ones(1,1,dtype=torch.bool))
    assert result['cross'].item()==pytest.approx((1+1e-8)**.5)
    assert result['self'].item()==pytest.approx((4+1e-8)**.5)
    assert result['loss'].item()==pytest.approx((1+1e-8)**.5-.5*(4+1e-8)**.5,abs=1e-7)


def test_b3_reference_set_permutation():
    p=torch.randn(4,4,75);v=torch.ones_like(p,dtype=torch.bool);r=torch.randn(5,75);rv=torch.ones_like(r,dtype=torch.bool)
    a=b3_score(p,v,r,rv);b=b3_score(p[:,[2,0,3,1]],v,r[[4,1,2,0,3]],rv)
    for k in a:torch.testing.assert_close(a[k],b[k])


def test_b4_bank_exchange():
    p=torch.randn(4,4,75);v=torch.ones_like(p,dtype=torch.bool);r=torch.randn(3,75);rv=torch.ones_like(r,dtype=torch.bool)
    a=b4_score(p,v,r,rv);b=b4_score(p[[2,3,0,1]],v,r,rv)
    for k in a:torch.testing.assert_close(a[k],b[k])


def test_b4_not_flatten_iid():
    p=torch.tensor([0.,0.,2.,2.])[:,None,None].expand(4,2,1);v=torch.ones_like(p,dtype=torch.bool)
    r=torch.tensor([[0.],[2.]]);rv=torch.ones_like(r,dtype=torch.bool)
    correct=b4_score(p,v,r,rv)
    wrong=descriptor_es(p.flatten(0,1),v.flatten(0,1),r,rv)
    assert correct['self'].item()==pytest.approx(2.)
    assert correct['self']-wrong['self']>.8


def test_common_noise_mixture_estimator():
    one=torch.randn(1,4,75);p=one.expand(4,-1,-1);v=torch.ones_like(p,dtype=torch.bool)
    refs=torch.randn(3,75);rv=torch.ones_like(refs,dtype=torch.bool)
    expected=descriptor_es(one[0],v[0],refs,rv);actual=session_mixture_score(p,v,refs,rv)
    for k in actual:torch.testing.assert_close(expected[k],actual[k])


def test_schedule_independence_and_hashes(population):
    train,_=population
    a,noise,hashes=make_schedule(train,512)
    b,noise_b,hashes_b=make_schedule(train,512,reference_seed=999)
    assert [r['source_indices'] for r in a]==[r['source_indices'] for r in b]
    assert torch.equal(noise,noise_b)
    for key in ('session_schedule_hash','source_occurrence_hash','noise_hash'):assert hashes[key]==hashes_b[key]
    assert hashes['reference_selection_hash']!=hashes_b['reference_selection_hash']
    s0=[r for r in a if r['session_id']=='s0']
    assert abs(len(s0)/len(a)-.75)<.08
    same=sum(r['source_indices'][0]==r['source_indices'][2] for r in s0)/len(s0)
    assert abs(same-1/3)<.08
    assert not torch.equal(noise[:,:2],noise[:,2:])
    c,nc,hc=make_schedule(train,512)
    assert hashes==hc and torch.equal(noise,nc)


def test_matched_initialization_and_b2_no_refs(population):
    train,_=population;records,noises,hashes=make_schedule(train,1)
    cache={i:train.dataset[i] for i in records[0]['source_indices']}
    class FailOnRead(dict):
        def __getitem__(self,key):raise AssertionError('B2 read a population ref')
    batch,refs=prepare_step(records[0],cache,FailOnRead(),'B2','cpu')
    assert refs is None
    initial=[]
    for _ in ('B2','B3','B4'):initial.append(state_hash(seeded_model('cpu')))
    assert len(set(initial))==1


def test_no_refs_enter_generator(model,population,monkeypatch):
    train,_=population
    batch=default_collate([train.dataset[0]]*4)
    scaler=DescriptorScaler(torch.zeros(75),torch.ones(75),torch.ones(75))
    noise=torch.randn(4,4,32);refs=torch.randn(3,75);rv=torch.ones_like(refs,dtype=torch.bool)
    seen=[];original=model.forward
    def checked(*args,**kwargs):
        assert not set(kwargs)&{'target','paired_target','reference','references','pair_lengths'}
        seen.append(set(kwargs));return original(*args,**kwargs)
    monkeypatch.setattr(model,'forward',checked)
    for arm in ('B2','B3','B4'):
        pair,dist=losses_for_batch(model,batch,noise,scaler,arm,None if arm=='B2' else (refs,rv))
        assert pair['overall'].requires_grad and dist['loss'].requires_grad
    assert len(seen)==6
    assert not set(inspect.signature(HiRPNet.forward).parameters)&{'target','reference','references'}


def test_toy_hierarchy_sanity():
    result=toy_hierarchy()
    assert result['initial_B3_ES']>result['initial_B4_ES']+.3
    assert result['initial_B4_ES']==pytest.approx(result['homogeneous_ES'])
    assert abs(result['final']['B3']['conditional_gap'])<1e-5
    assert result['final']['B4']['conditional_gap']==pytest.approx(.8)
    assert result['final']['B4']['marginal_probability']==pytest.approx(.5)
    print('TOY_HIERARCHY',json.dumps(result))


def test_a0_k_prefix_still_holds(model,inputs):
    noise=torch.randn(2,10,32)
    with prior_mode(model,'A0'):
        a=model.sample(**inputs,sample_count=4,noise=noise[:,:4])
        b=model.sample(**inputs,sample_count=10,noise=noise)
    torch.testing.assert_close(a,b[:,:4],atol=1e-6,rtol=1e-5)



def test_split_escape_reference_rejected(population):
    train,val=population
    target=next(iter(train.reference_paths.values()))
    destination=next(iter(val.reference_paths.values())).parent/'outside.npy'
    destination.symlink_to(target)
    with pytest.raises(ValueError,match='escapes split'):
        SessionPopulation(val.dataset.root,'val',4)


def test_joint_descriptor_coordinate_mask():
    x=torch.tensor([[1.,1000.,4.]]);y=torch.tensor([[1.,1.,1000.]])
    xv=torch.tensor([[True,False,True]]);yv=torch.tensor([[True,True,False]])
    assert descriptor_distances(x,xv,y,yv).item()==pytest.approx(1e-4)
    with pytest.raises(ValueError,match='jointly valid'):
        descriptor_distances(x,torch.zeros_like(xv),y,yv)


def test_scaler_invalid_velocity_not_observed_zero(population,tmp_path):
    train,_=population
    first=next(iter(train.reference_paths.values()))
    np.save(first,np.ones((1,25),dtype=np.float32))
    scaler=fit_descriptor_scaler(train,tmp_path/'short_stats.json')
    assert (scaler.count[:50]==4).all() and (scaler.count[50:]==3).all()
    velocities=[]
    for ref_id in train.reference_paths:
        item=train.load_reference(ref_id)
        if item['length']>=2:velocities.append(reaction_descriptor(item['reaction'],item['length'])[0][50:])
    torch.testing.assert_close(scaler.mean[50:],torch.stack(velocities).mean(0))


def test_actual_pilot_matched_controls():
    root=Path(__file__).resolve().parents[2]
    path=root/'runs/phase2/training_summary.json'
    if not path.exists():pytest.skip('Phase 2 pilot artifacts not available')
    result=json.loads(path.read_text())
    if result.get('stopped_before_training'):pytest.skip('gradient gate intentionally stopped pilot')
    arms=result['arms']
    for key in ('initialization_hash','parameter_count','steps','session_schedule_hash','source_occurrence_hash','noise_hash','descriptor_scaler_hash'):
        assert len({value[key] for value in arms.values()})==1
    assert all(value['steps']==128 for value in arms.values())
    assert arms['B2']['population_reference_lookups']==0
    assert arms['B3']['reference_selection_hash']==arms['B4']['reference_selection_hash']
    for arm in arms:
        rows=[json.loads(line) for line in (path.parent/f'{arm}_training.jsonl').read_text().splitlines()]
        assert len(rows)==128 and all(r['prior_gradient_norm']==0 for r in rows)
