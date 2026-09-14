"""Separate opt-in weak contract. Gold source_cache is deliberately untouched."""
from dataclasses import dataclass
import hashlib,json,re
from pathlib import Path
import numpy as np
TYPES=('utterance','question','request','evaluation','disclosure','topic_change','unknown')
@dataclass(frozen=True)
class WeakEvent:
    event_id:str
    text:str
    event_type:str
    interval:tuple|None  # crop-relative PTS seconds, never guessed from character index
    frame_interval:tuple|None
    regime:str='auto_weak'

def digest(data):return hashlib.sha256(data).hexdigest()
def read_weak(path,*,split,expected_cache_sha256,expected_media_hashes,expected_policy_sha256):
    if not Path(path).is_file():return None
    data=Path(path).read_bytes()
    if digest(data)!=expected_cache_sha256:raise ValueError('cache identity mismatch')
    obj=json.loads(data)
    required={'schema_version','annotation_regime','clip_id','split','human_reviewed','training_eligible','visible_roles','input_hashes','teacher_hashes','policy_sha256','code_hashes','events'}
    if set(obj)!=required:raise ValueError('unexpected weak cache fields')
    if obj['annotation_regime']!='auto_weak' or obj['schema_version']!='auto-weak-v1' or obj['human_reviewed'] is not False or obj['training_eligible'] is not False:raise ValueError('regime mismatch')
    if obj['split']!=split or obj['visible_roles']!=['speaker']:raise ValueError('source-only split violation')
    if not re.fullmatch('rec_[0-9a-f]{16}',obj['clip_id']):raise ValueError('nonopaque identity')
    if sorted(obj['input_hashes'])!=sorted(expected_media_hashes) or obj['policy_sha256']!=expected_policy_sha256:raise ValueError('provenance mismatch')
    if not obj['teacher_hashes'] or not obj['code_hashes']:raise ValueError('missing teacher or pipeline identity')
    selected=[]
    allowed={'event_id','text','event_type','role','role_evidence','timing_evidence','sync_status','time_domain','weak_input_eligible','rules','rejection_reasons','source_version_hashes'}
    for e in obj['events']:
        if set(e)!=allowed:raise ValueError('unexpected event fields')
        if e['event_type'] not in TYPES or not isinstance(e['text'],str):raise ValueError('bad content')
        if not e['weak_input_eligible']:continue
        if e['role']!='source_speaker_candidate' or not e['rules'] or not all(e['rules'].values()) or e['rejection_reasons']:raise ValueError('ineligible role or gates')
        if not e['source_version_hashes'] or e['time_domain']!='speaker_audio_sample_clock':raise ValueError('unsupported evidence')
        selected.append(e)
    return selected or None

def crop_events(events,frame_pts,crop_start,crop_end,*,trajectory_frames):
    pts=np.asarray(frame_pts,float)
    if len(pts)!=trajectory_frames or len(pts)<2 or not np.all(np.isfinite(pts)) or np.any(np.diff(pts)<=0):raise ValueError('unmapped trajectory time domain')
    if not 0<=crop_start<crop_end<=len(pts):raise ValueError('crop out of bounds')
    end_pts=pts[crop_end] if crop_end<len(pts) else pts[-1]+np.median(np.diff(pts));origin=pts[crop_start]
    result=[]
    for e in events or []:
        if not e.get('weak_input_eligible') or e.get('role')!='source_speaker_candidate':continue
        timing=e['timing_evidence'];sync=e['sync_status'];interval=None;frames=None
        if sync['status']=='estimated' and sync['offset_s'] is not None:
            # audio absolute -> video absolute -> real PTS/crop intersection.
            lo=timing['start_s']+sync['offset_s'];hi=timing['end_s']+sync['offset_s']
            valid=sync['effective_audio_range_s']
            if timing['start_s']>=valid[0] and timing['end_s']<=valid[1]:
                lo=max(lo,origin);hi=min(hi,end_pts)
                if lo>=hi:continue
                # Include the frame whose interval intersects a crossing event.
                left=max(crop_start,int(np.searchsorted(pts,lo,side='right'))-1)
                right=min(crop_end,int(np.searchsorted(pts,hi,side='left')))
                interval=(float(lo-origin),float(hi-origin));frames=(left-crop_start,right-crop_start)
        result.append(WeakEvent(e['event_id'],e['text'],e['event_type'],interval,frames))
    return result or None
