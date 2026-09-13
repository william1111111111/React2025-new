import copy,json
from pathlib import Path
import pytest
from semantic_supervision.models.source_cache import source_events,training_observations,load_source_cache

def record():
    r=json.loads((Path(__file__).parents[1]/'vendor/react_semantic_supervision/examples.synthetic.json').read_text())[0]
    # Unit fixture only; never emitted into any production annotation/cache file.
    r['is_synthetic_example']=False;r['clip_id']='UNIT_TEST_ONLY'
    p=dict(clip_id=r['clip_id'],actual_visible_inputs=[dict(role='speaker',sha256='unit-media')],review_status='accepted',training_eligible=True,speaker_channel_role_verified=True,sync_verified=True,evidence_refs=['synthetic_transcript_segment_1'])
    return r,p

def test_unreviewed_source_is_null():
    r,p=record();p['review_status']='pending'
    assert source_events(r,p,split='train',expected_media_hashes=['unit-media']) is None

def test_source_cache_target_independence(tmp_path):
    r,p=record();ann=tmp_path/'source.json';prov=tmp_path/'provenance.json';target=tmp_path/'listener_target.json'
    ann.write_text(json.dumps([r]));prov.write_text(json.dumps([p]));target.write_text('{"target": [1]}')
    a=load_source_cache(ann,prov,r['clip_id'],split='train',expected_media_hashes=['unit-media'])
    target.write_text('invalid target cache: deliberately cannot parse')
    b=load_source_cache(ann,prov,r['clip_id'],split='train',expected_media_hashes=['unit-media'])
    assert a==b==r['events']

@pytest.mark.parametrize('field,value',[('speaker_channel_role_verified',False),('sync_verified',False)])
def test_unverified_roles_or_times_fall_back(field,value):
    r,p=record();p[field]=value
    assert source_events(r,p,split='train',expected_media_hashes=['unit-media']) is None

def test_source_provenance_leak_rejected():
    r,p=record();p['actual_visible_inputs'][0]['role']='listener'
    with pytest.raises(ValueError):source_events(r,p,split='train',expected_media_hashes=['unit-media'])

def test_wrong_source_hash_rejected():
    r,p=record()
    with pytest.raises(ValueError):source_events(r,p,split='train',expected_media_hashes=['wrong'])

def test_unknown_evidence_rejected():
    r,p=record();p['evidence_refs']=[]
    with pytest.raises(ValueError):source_events(r,p,split='train',expected_media_hashes=['unit-media'])

def test_split_leak_rejected():
    r,p=record()
    with pytest.raises(ValueError):source_events(r,p,split='test',expected_media_hashes=['unit-media'])

def test_privileged_observation_not_accepted_as_source():
    r,p=record();r.update(record_type='listener_observations',visible_roles=['listener'],allowed_for_model_input=False)
    with pytest.raises(ValueError):source_events(r,p,split='train',expected_media_hashes=['unit-media'])
    with pytest.raises(ValueError):training_observations(r,p,split='test')

def test_synthetic_example_rejected_by_model_loader():
    r,p=record();r['is_synthetic_example']=True
    with pytest.raises(ValueError):source_events(r,p,split='train',expected_media_hashes=['unit-media'])
