"""Bounded source-only ASD lag diagnostic. Never asserts verified synchronization."""
import json
from pathlib import Path
import numpy as np,torch,torchaudio,soundfile as sf
from .automatic import ActiveSpeaker,video_features,write
LAGS=[-.4,-.2,0.,.2,.4]
def main():
    torch.set_num_threads(4);out=Path('/outputs');job=json.loads(Path('/job/job.json').read_text());model=ActiveSpeaker()
    for row in job['records']:
        p=out/(row['id']+'.json')
        if p.exists():continue
        rel=row['relative'];raw,sr=sf.read('/inputs/audio/'+rel+'.wav',always_2d=True)
        wave=torchaudio.functional.resample(torch.from_numpy(raw.mean(1)).float(),sr,16000)
        vf,vt=video_features(Path('/inputs/video')/(rel+'.mp4'))
        assert abs(vt['first_pts'])<1e-6,'legacy video sampler cannot handle nonzero origin; unresolved'
        n=len(wave);scores=[];averages=[]
        for lag in LAGS:
            shift=round(lag*16000);shifted=torch.zeros_like(wave)
            if shift>=0:shifted[shift:]=wave[:n-shift]
            else:shifted[:shift]=wave[-shift:]
            values=model.score(shifted,vf);times=np.arange(len(values))*.04+.02
            keep=(times>.6)&(times<min(vt['duration_s'],n/16000)-.6)
            use=np.zeros(len(times),bool)
            for e in row['probe_events']:use|=(times>=e['start_s'])&(times<e['end_s'])
            keep&=use&np.isfinite(values)
            averages.append(float(np.mean(values[keep])) if keep.sum()>=5 else None)
            scores.append({'lag_s':lag,'times_s':times.tolist(),'logits':[float(x) if np.isfinite(x) else None for x in values],'score_frames':int(keep.sum())})
        nums=np.array([x if x is not None else -np.inf for x in averages]);best=int(np.argmax(nums));ordered=np.sort(nums)
        margin=float(ordered[-1]-ordered[-2]) if np.isfinite(ordered[-2]) else 0.
        resolved=best not in [0,len(LAGS)-1] and margin>=.1 and scores[best]['score_frames']>=25
        write(p,{'clip_id':row['id'],'audio_origin_s':0.,'audio_origin_source':'WAV first decoded sample','video_origin_s':vt['first_pts'],'video_origin_source':'ffprobe best_effort_timestamp_time','mapping':'video_pts = audio_sample_seconds + offset_s','offset_s':LAGS[best] if resolved else None,'sync_status':'estimated' if resolved else 'unresolved','sync_verified':False,'grid_s':LAGS,'mean_logits':averages,'best_grid_lag_s':LAGS[best],'best_vs_second_margin':margin,'effective_audio_range_s':[.6,n/16000-.6],'offset_interval_s':[LAGS[best]-.2,LAGS[best]+.2] if resolved else None,'method':'fixed bounded noncircular zero-padded ASD lag probe; thresholds engineering assumptions','probe_scores':scores})
        print(row['id'],'estimated' if resolved else 'unresolved',LAGS[best],round(margin,3),flush=True)
if __name__=='__main__':main()
