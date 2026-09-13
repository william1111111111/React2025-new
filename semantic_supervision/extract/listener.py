"""TRAIN listener numeric change proposals; no semantic/psychological naming."""
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter1d
from .common import checked,sha,timeline,output,wrapper
GROUPS=[('AU',0,15),('VA',15,17),('expression',17,25)]

def changes(values,times):
    if values.shape!=(len(times),25) or not np.isfinite(values).all():raise ValueError('invalid listener attributes')
    # Time-aware fixed200ms local average, using measured PTS; not assumed FPS.
    begin=np.searchsorted(times,times-.1);end=np.searchsorted(times,times+.1,side='right')
    prefix=np.vstack([np.zeros((1,25)),np.cumsum(values,axis=0,dtype=np.float64)])
    smooth=(prefix[end]-prefix[begin])/(end-begin)[:,None]
    return np.abs(np.diff(smooth,axis=0))/np.diff(times)[:,None]

def main():
    job=json.loads(Path('/job/job.json').read_text());data=[];scales={g:[] for g,_,_ in GROUPS}
    for clip in job['records']:
        rel=clip['relative'];vp=f'/inputs/video/{rel}.mp4';ap=f'/inputs/attributes/{rel}.npy'
        a=np.load(checked(ap),allow_pickle=False).astype(np.float64);pts,tm=timeline(vp,clip['listener_frames']);speed=changes(a,pts)
        for g,lo,hi in GROUPS:scales[g].append(speed[:,lo:hi].mean(1))
        data.append((clip,a,pts,tm,speed,vp,ap))
    # Fitted only to the34 TRAIN pilot recordings, never DEV/MAM outputs.
    thresholds={g:float(np.quantile(np.concatenate(v),.90)) for g,v in scales.items()}
    records=[];provenance=[];stats=[]
    for clip,a,pts,tm,speed,vp,ap in data:
        key=clip['id'];events=[]
        for g,lo,hi in GROUPS:
            activity=speed[:,lo:hi].mean(1);mask=activity>max(thresholds[g],1e-8);indices=np.flatnonzero(mask)
            chunks=np.split(indices,np.where(np.diff(indices)>1)[0]+1) if len(indices) else []
            for idx in chunks:
                if not len(idx):continue
                start=int(idx[0]);end=int(idx[-1]+1);apex=int(idx[np.argmax(activity[idx])]+1)
                event_id=f'{key}_{g}_{start}_{end}'
                events.append(dict(event_id=event_id,start_s=float(pts[start]),end_s=float(pts[end]),description=f'{g}数值变化候选；尚未核验是否对应可见动作。',event_type='numeric_change_candidate',evidence_status='ambiguous',evidence_ref=f'{key}:frames:{start}:{end}:apex:{apex}'))
        records.append(wrapper('listener_observations',key,tm['end_pts'],events))
        provenance.append(dict(clip_id=key,review_status='pending',training_eligible=False,origin='observed_attributes_numeric_proposal_not_visual_truth',teacher_model=None,teacher_version=None,prompt_sha256=job['prompt_sha256'],actual_visible_inputs=[dict(path=p,sha256=sha(p),role='listener') for p in [vp,ap]],timeline=tm,visible_no_reaction=None,no_numeric_event=len(events)==0,occlusion='unknown',sync_verified=False,evidence_refs=[e['evidence_ref'] for e in events],global_summary=dict(mean=a.mean(0).tolist(),q10=np.quantile(a,.1,axis=0).tolist(),q90=np.quantile(a,.9,axis=0).tolist())))
        stats.append(dict(clip_id=key,numeric_event_count=len(events),visual_no_reaction='unknown',occlusion='unknown'))
        print('LISTENER',key,len(events),flush=True)
    output('annotations.json',records);output('provenance.json',provenance);output('detector.json',dict(thresholds=thresholds,fit_split='train',fit_recordings=len(data),smoothing='200ms PTS centered mean',quantile=.9,units='mean absolute attribute change per second',semantic_accuracy='not_assessed'));output('stats.json',stats)
if __name__=='__main__':main()
