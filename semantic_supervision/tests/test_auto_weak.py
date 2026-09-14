import copy,json
from dataclasses import replace
from pathlib import Path
import numpy as np,pytest,torch
from semantic_supervision.align.support import support
from semantic_supervision.models.auto_weak import WeakEvent,crop_events,read_weak,digest
from semantic_supervision.models.semantic_flow import SemanticBranch

def score(a,c=None,start=0,end=4,times=None,video_start=0,video_end=4,grid_origin=.02):
    return support(start,end,np.arange(.02,4,.04) if times is None else times,a,a if c is None else c,video_start=video_start,video_end=video_end,audio_start=0,audio_end=10,grid_origin=grid_origin,control_shift_s=8)

def test_one_of_100_positive_not_coverage():
    a=np.full(100,np.nan);a[0]=3;s=score(a,np.zeros(100))
    assert s['expected_grid_points']==100 and s['common_finite_frames']==1 and not s['passed']

def test_all_nan():assert not score(np.full(100,np.nan))['passed']
def test_partial_video_overlap():
    s=score(np.ones(50),np.zeros(50),times=np.arange(.02,2,.04),video_end=2)
    assert s['expected_grid_points']==100 and s['overlap_grid_points']==50 and not s['passed']
def test_short_sentence():assert not score(np.ones(100),np.zeros(100),start=.1,end=.15)['rules']['minimum_duration']
def test_out_of_bounds():assert not score(np.ones(100),np.zeros(100),start=-.1)['rules']['within_audio']
def test_nonzero_origin():
    s=score(np.ones(100),np.zeros(100),start=5,end=9,times=np.arange(5.02,9,.04),video_start=5,video_end=9,grid_origin=5.02)
    assert s['expected_grid_points']==100 and s['overlap_grid_points']==100

def event():return {'event_id':'event','text':'Hello','event_type':'utterance','weak_input_eligible':True,'role':'source_speaker_candidate','timing_evidence':{'start_s':.9,'end_s':1.8},'sync_status':{'status':'estimated','offset_s':10.,'effective_audio_range_s':[0,10]}}
def test_crop_intersection_and_tail():
    e=event();pts=np.array([10.,10.5,11.,11.5,12.])
    x=crop_events([e],pts,2,4,trajectory_frames=5)[0]
    assert x.interval==pytest.approx((0,.8)) and x.frame_interval==(0,2)
    e['timing_evidence']={'start_s':2.2,'end_s':2.8}
    assert crop_events([e],pts,4,5,trajectory_frames=5)[0].interval==pytest.approx((.2,.5))
    with pytest.raises(ValueError):crop_events([e],pts,0,5,trajectory_frames=4)
def test_unknown_and_missing_remain_null():
    e=event();e['role']='unknown'
    assert crop_events([e],np.arange(10.),0,5,trajectory_frames=10) is None
    assert crop_events(None,np.arange(10.),0,5,trajectory_frames=10) is None

def test_time_shift_changes_only_time_components():
    branch=SemanticBranch(16,4,0).eval();e=WeakEvent('e','English quote','question',(1,2),(2,4))
    a=branch.components(e,'event','cpu',torch.float32);b=branch.components(replace(e,interval=(2,3)),'event','cpu',torch.float32)
    assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1]) and not torch.equal(a[2],b[2])
    assert branch(torch.zeros(1,3,16),[None]).shape==(1,3,16)

def test_real_cache_42_trace_and_source_rejection(tmp_path):
    root=Path(__file__).resolve().parents[2];base=root/'runs/reaction_flow/semantic_auto_weak_v1';idx=json.loads((base/'cache/index.json').read_text());loaded=[]
    for cid,m in idx.items():
        kwargs=dict(split='train',expected_cache_sha256=m['cache_sha256'],expected_media_hashes=m['media_hashes'],expected_policy_sha256=m['policy_sha256'])
        loaded.extend(read_weak(base/'cache'/(cid+'.json'),**kwargs) or [])
    old=json.loads((base/'support_report.json').read_text());assert len(old['old_42_ids'])==42
    assert {e['event_id'] for e in loaded}==set(old['new_ids']) and len(loaded)>0
    original=json.loads((base/'cache'/(cid+'.json')).read_text())
    for key,value in [('visible_roles',['listener']),('session_id','session0'),('split','val')]:
        bad={**original,key:value};p=tmp_path/'bad.json';p.write_text(json.dumps(bad));kw={**kwargs,'expected_cache_sha256':digest(p.read_bytes())}
        with pytest.raises(ValueError):read_weak(p,**kw)

def test_listener_file_mutation_cannot_reach_semantic_interface(tmp_path,monkeypatch):
    branch=SemanticBranch(16,4,0).eval();h=torch.randn(1,5,16);events=[[WeakEvent('e','Source text','utterance',None,None)]]
    target=tmp_path/'listener_cache.json';target.write_text('first target')
    a=branch(h,events).detach();target.write_text('completely different target')
    def forbidden(*args,**kwargs):raise AssertionError('semantic branch attempted filesystem access')
    monkeypatch.setattr(Path,'read_text',forbidden);monkeypatch.setattr(Path,'read_bytes',forbidden)
    b=branch(h,events).detach();assert torch.equal(a,b)

def test_unresolved_sync_keeps_content_without_precise_time():
    e=event();e['sync_status']['status']='unresolved';e['sync_status']['offset_s']=None
    actual=crop_events([e],np.arange(10.),0,4,trajectory_frames=10)
    assert actual and actual[0].interval is None and actual[0].frame_interval is None and actual[0].text=='Hello'

def test_missing_cache_returns_null(tmp_path):
    assert read_weak(tmp_path/'missing.json',split='train',expected_cache_sha256='unused',expected_media_hashes=[],expected_policy_sha256='unused') is None
