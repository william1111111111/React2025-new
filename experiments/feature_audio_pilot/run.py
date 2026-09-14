"""Independent, CPU-only TRAIN feature/audio synchronization diagnostic."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.ndimage import gaussian_filter1d
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/reaction_flow/feature_audio_pilot_pts_v1'
POLICY=dict(lag_seconds=[round(x*.04,2) for x in range(-25,26)], lag_grid_step_s=.04,
    smoothing_sigma_s=.08, minimum_frames=125, minimum_peak_correlation=.2,
    minimum_far_peak_margin=.03, far_peak_distance_s=.2, maximum_half_lag_difference_s=.12,
    minimum_null_margin=.05, null_shifts_s=[-7,-5,-3,3,5,7],
    mapping='corr(M(t), A(t+delta)); feature_time = audio_time - delta',
    feature_time_basis='actual video PTS by corresponding frame index; equal frame counts required, extractor correspondence still an assumption',
    audio_origin='first decoded WAV sample, 0 seconds', au_mapping='unresolved; all AU are anonymous',
    calibration='none; fixed engineering gates, not calibrated probabilities',
    scope='TRAIN speaker only; no listener, API, GPU, cache mutation or DEV tuning')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def corr(x,y):
    k=np.isfinite(x)&np.isfinite(y)
    if k.sum()<POLICY['minimum_frames'] or np.std(x[k])<1e-10 or np.std(y[k])<1e-10:return None
    return float(np.corrcoef(x[k],y[k])[0,1])
def scan(t,m,at,a,limits,lags):
    # Common noncircular support across ALL searched lags; no wrapped samples.
    k=(t>=max(limits[0],at[0]-min(lags)))&(t<=min(limits[1],at[-1]-max(lags)))
    return [corr(m[k],np.interp(t[k]+lag,at,a)) for lag in lags],int(k.sum())
def peak(scores,lags):
    if not any(x is not None for x in scores):return None
    i=max(range(len(scores)),key=lambda j: scores[j] if scores[j] is not None else -np.inf)
    return dict(delta_s=float(lags[i]),corr=scores[i],boundary=i in (0,len(scores)-1))
def diagnostic(t,m,at,a):
    lags=POLICY['lag_seconds'];end=min(t[-1],at[-1]);limits=(.12,end-.12)
    scores,n=scan(t,m,at,a,limits,lags);p=peak(scores,lags)
    if p is None:return dict(status='unresolved',reason=['insufficient_or_constant'],scores=scores)
    halves=[peak(scan(t,m,at,a,lim,lags)[0],lags) for lim in [(limits[0],end/2),(end/2,limits[1])]]
    far=[v for v,l in zip(scores,lags) if v is not None and abs(l-p['delta_s'])>=.2-1e-8]
    margin=p['corr']-max(far) if far else None
    # Re-optimize each shifted negative control; diagnostic, not a p-value.
    null=[]
    for shift in POLICY['null_shifts_s']:
        q=peak(scan(t,m,at,a,limits,[l+shift for l in lags])[0],[l+shift for l in lags]);null.append(q)
    nc=max([q['corr'] for q in null if q is not None],default=None)
    rules=dict(interior=not p['boundary'],correlation=p['corr']>=.2,
        sharpness=margin is not None and margin>=.03,
        half_stability=all(h is not None and not h['boundary'] and abs(h['delta_s']-p['delta_s'])<=.12+1e-8 for h in halves),
        negative_control=nc is not None and p['corr']-nc>=.05)
    return dict(status='candidate' if all(rules.values()) else 'unresolved',peak=p,
        feature_minus_audio_offset_s=-p['delta_s'],common_frames=n,far_peak_margin=margin,
        half_peaks=halves,null_peaks=null,rules=rules,reason=[k for k,v in rules.items() if not v],scores=scores)
def main():
    OUT.mkdir(parents=True,exist_ok=False);write(OUT/'POLICY.json',POLICY)
    job=ROOT/'runs/reaction_flow/semantic_alignment_pilot_v1/job/job.json'
    rows=[]
    for row in json.loads(job.read_text())['records']:
        rel=row['relative'];paths={k:ROOT/'data/train'/folder/'speaker'/(rel+ext) for k,folder,ext in [('au','facial-attributes','.npy'),('exp','coefficients','.npy'),('audio','audio','.wav'),('video','video-face-crop','.mp4')]}
        au=np.load(paths['au']);ex=np.load(paths['exp']);ex=ex[:,0,:] if ex.ndim==3 and ex.shape[1]==1 else ex;raw,sr=sf.read(paths['audio'],always_2d=True);wave=raw.mean(1)
        assert au.ndim==2 and au.shape[1]==25 and ex.ndim==2 and ex.shape[1]==58
        assert len(au)==len(ex)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=best_effort_timestamp_time','-of','json',str(paths['video'])]))
        pts=np.array([float(f['best_effort_timestamp_time']) for f in probe['frames']]);assert len(pts)==len(au) and np.all(np.diff(pts)>0);t=pts;frame_dt=float(np.median(np.diff(t)))
        hop=max(1,round(sr*.01));size=len(wave)//hop
        rms=np.sqrt(np.mean(wave[:size*hop].reshape(size,hop)**2,axis=1));at=(np.arange(size)+.5)*hop/sr
        envelope=gaussian_filter1d(np.log1p(rms/(np.median(rms)+1e-8)),8)
        audios={'log_rms':envelope,'envelope_change':np.abs(np.gradient(envelope))}
        signals={}
        def motion(x):return gaussian_filter1d(np.mean(np.abs(np.diff(x,axis=0,prepend=x[:1])),axis=1),.08/frame_dt)
        signals['anonymous_all15_AU_switching']=motion(au[:,:15])
        signals['global_52_expression_motion']=motion(ex[:,:52])
        # Individual anonymous channels are exploratory only, never select the best as mouth.
        channels=[diagnostic(t,motion(au[:,i:i+1]),at,envelope) for i in range(15)]
        results={f'{m}__{a}':diagnostic(t,v,at,av) for m,v in signals.items() for a,av in audios.items()}
        lagpath=ROOT/'runs/reaction_flow/semantic_auto_weak_v1/lag'/(row['id']+'.json')
        lag=json.loads(lagpath.read_text()) if lagpath.exists() else None
        for r in results.values():
            if 'peak' in r and lag and lag.get('offset_s') is not None:
                r['difference_from_TalkNet_offset_s']=r['feature_minus_audio_offset_s']-lag['offset_s']
        record=dict(id=row['id'],frames=len(au),audio_duration_s=len(wave)/sr,
            finite_AU_fraction=float(np.isfinite(au[:,:15]).mean()),binary_AU_fraction=float(np.isin(au[:,:15],[0,1]).mean()),
            video_frames=len(pts),video_origin_s=float(pts[0]),feature_video_grid_max_error_s=float(np.max(np.abs(pts-t))) if len(pts)==len(t) else None,
            input_hashes={k:sha(v) for k,v in paths.items()},results=results,anonymous_channel_diagnostics=channels,
            talknet=dict(status=lag['sync_status'],offset_s=lag['offset_s'],best_grid_lag_s=lag['best_grid_lag_s']) if lag else None)
        write(OUT/(row['id']+'.json'),record);rows.append(record)
        print(row['id'],sum(r['status']=='candidate' for r in results.values()),flush=True)
    summary={}
    for key in rows[0]['results']:
        rr=[r['results'][key] for r in rows];peaks=[r['peak']['corr'] for r in rr if 'peak'in r]
        summary[key]=dict(records=len(rr),candidates=sum(r['status']=='candidate' for r in rr),median_peak_corr=float(np.median(peaks)),rejections={k:sum(k in r.get('reason',[]) for r in rr) for k in ['correlation','sharpness','half_stability','negative_control','interior','insufficient_or_constant']},candidate_ids=[row['id'] for row,r in zip(rows,rr) if r['status']=='candidate'])
    write(OUT/'SUMMARY.json',dict(records=len(rows),results=summary,job_sha256=sha(job),script_sha256=sha(Path(__file__)),policy_sha256=sha(OUT/'POLICY.json'),all_binary=all(r['binary_AU_fraction']==1 for r in rows),video_grid_matches=sum(r['feature_video_grid_max_error_s'] is not None and r['feature_video_grid_max_error_s']<1e-5 for r in rows),talknet_present=sum(r['talknet'] is not None for r in rows)))
if __name__=='__main__':main()
