"""CPU extraction helpers; real media are available only inside role sandboxes."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np

def checked(path):
    p=Path(path).resolve()
    if not p.is_relative_to('/inputs'):raise ValueError('media path outside role mount')
    return p

def sha(path):return hashlib.sha256(checked(path).read_bytes()).hexdigest()
def output(name,value):Path('/outputs',name).write_text(json.dumps(value,indent=2,allow_nan=False))
def timeline(path,expected):
    path=checked(path)
    raw=json.loads(subprocess.check_output(['ffprobe','-v','error','-threads','1','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time,pkt_duration_time','-of','json',str(path)],text=True))
    ts=np.array([float(f['best_effort_timestamp_time']) for f in raw['frames']],dtype=np.float64)
    if len(ts)!=expected or len(ts)<2 or not np.isfinite(ts).all() or not (np.diff(ts)>0).all():raise ValueError('invalid/nonmonotonic frame PTS or frame count mismatch')
    # Last duration is packet duration where present; otherwise explicitly estimated.
    last=raw['frames'][-1].get('pkt_duration_time');duration=float(last) if last is not None else float(np.median(np.diff(ts)))
    return ts,dict(frame_count=len(ts),first_pts=float(ts[0]),end_pts=float(ts[-1]+duration),last_duration_estimated=last is None,max_frame_interval=float(np.diff(ts).max()),min_frame_interval=float(np.diff(ts).min()),frame_pts=ts.tolist())

def wrapper(kind,clip,duration,events):
    return dict(schema_version='react-label-v0.1',record_type=kind,split='train',clip_id=clip,duration_s=duration,is_synthetic_example=False,allowed_for_model_input=kind=='speaker_events',visible_roles=['speaker'] if kind=='speaker_events' else ['listener'] if kind=='listener_observations' else ['speaker','listener'],annotator_version='cpu-proposals-v1-no-teacher',events=events)
