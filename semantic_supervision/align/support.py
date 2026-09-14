"""Fixed engineering gates; no fitted probabilities or DEV metric inputs."""
import numpy as np
POLICY=dict(coverage_min=.9,overlap_min=.9,min_valid_frames=5,min_duration_s=.2,max_negative_run_s=.24,active_fraction_min=.8,control_margin_min=.5,control_seam_guard_s=.12)

def support(start,end,times,actual,control,*,video_start,video_end,audio_start,audio_end,grid_origin=.02,step=.04,offset=0.,control_shift_s=3.,policy=POLICY):
    t=np.asarray(times,float);a=np.asarray(actual,float);c=np.asarray(control,float)
    if len(t)!=len(a) or len(t)!=len(c) or np.any(np.diff(t)<=0):raise ValueError('invalid score grid')
    duration=end-start;vstart=start+offset;vend=end+offset
    lo=int(np.ceil((vstart-grid_origin)/step-1e-9));hi=int(np.ceil((vend-grid_origin)/step-1e-9))
    expected=max(0,hi-lo);grid=grid_origin+np.arange(lo,hi)*step
    overlap=int(np.sum((grid>=video_start)&(grid<video_end)))
    i,j=np.searchsorted(t,[vstart,vend]);tt=t[i:j];aa=a[i:j];cc=c[i:j]
    fa=np.isfinite(aa);fc=np.isfinite(cc);common=fa&fc
    length=audio_end-audio_start;effective=control_shift_s%length if length>0 else 0.
    # torch.roll splice is at effective shift, not automatically at 3 seconds.
    seam=(np.abs((tt-offset)-(audio_start+effective))<policy['control_seam_guard_s'])
    valid=common&~seam
    fraction=float(np.mean(aa[valid]>0)) if valid.any() else None
    margin=float(np.mean(aa[valid]-cc[valid])) if valid.any() else None
    negative=fa&(aa<=0);longest=run=0
    for x in negative:
        run=run+1 if x else 0;longest=max(longest,run)
    rules={'valid_interval':bool(np.isfinite(start) and np.isfinite(end) and duration>0),'within_audio':bool(start>=audio_start and end<=audio_end),'minimum_duration':duration>=policy['min_duration_s'],'time_domain_overlap':overlap/max(expected,1)>=policy['overlap_min'],'common_finite_coverage':int(common.sum())/max(expected,1)>=policy['coverage_min'],'usable_coverage':int(valid.sum())/max(expected,1)>=policy['coverage_min'],'minimum_valid_frames':int(valid.sum())>=policy['min_valid_frames'],'control_nontrivial':min(effective,max(0,length-effective))>=.5,'no_obvious_turn':longest*step<policy['max_negative_run_s'],'active_fraction':fraction is not None and fraction>=policy['active_fraction_min'],'control_margin':margin is not None and margin>=policy['control_margin_min']}
    return {'expected_grid_points':expected,'overlap_grid_points':overlap,'actual_finite_frames':int(fa.sum()),'control_finite_frames':int(fc.sum()),'common_finite_frames':int(common.sum()),'usable_frames':int(valid.sum()),'common_finite_coverage':int(common.sum())/max(expected,1),'usable_coverage':int(valid.sum())/max(expected,1),'time_domain_overlap':overlap/max(expected,1),'event_duration_s':duration,'crosses_obvious_turn':not rules['no_obvious_turn'],'negative_run_s':longest*step,'active_fraction':fraction,'control_margin':margin,'effective_circular_shift_s':effective,'control_seam_excluded_frames':int((common&seam).sum()),'rules':rules,'rejection_reasons':[k for k,v in rules.items() if not v],'passed':all(rules.values())}
