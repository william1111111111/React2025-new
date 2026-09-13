"""Speaker-only audio energy proposals; no ASR/semantic truth inferred."""
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from .common import checked,sha,timeline,output,wrapper

def main():
    job=json.loads(Path('/job/job.json').read_text());records=[];provenance=[];candidates=[]
    for clip in job['records']:
        key=clip['id'];rel=clip['relative'];audio=f'/inputs/audio/{rel}.wav';video=f'/inputs/video/{rel}.mp4';txt=f'/inputs/text/{rel}.txt'
        signal,rate=sf.read(checked(audio),always_2d=True);signal=signal.mean(1)
        if not np.isfinite(signal).all():raise ValueError('nonfinite audio')
        _,tm=timeline(video,clip['speaker_frames']);duration=min(len(signal)/rate,tm['end_pts'])
        # Fixed20ms amplitude bins: this is NOT a validated VAD or forced aligner.
        hop=max(1,round(.02*rate));energy=np.array([np.sqrt(np.mean(signal[s:s+hop]**2)) for s in range(0,len(signal),hop)])
        times=np.arange(len(energy))*hop/rate
        max_start=max(0.,duration-8.)
        starts=np.arange(0,max_start+.001,2.) if max_start else np.array([0.])
        means=np.array([np.mean(energy[(times>=s)&(times<min(duration,s+8))]) for s in starts])
        # All candidate windows are based ONLY on speaker channel energy and time.
        ranks=np.argsort(means,kind='stable');low=set(ranks[:max(1,len(ranks)//4)]);high=set(ranks[-max(1,len(ranks)//4):])
        for i,s in enumerate(starts):
            candidates.append(dict(window_id=f'{key}_w{i:04d}',clip_id=key,start_s=float(s),end_s=float(min(duration,s+8)),acoustic_stratum='low_energy' if i in low else 'high_energy' if i in high else 'typical_energy',RMS=float(means[i]),semantic_type='unknown',content_type_status='not_annotated',temporal_anchor='speaker_audio_sample_clock',sync_verified=False,review_status='pending'))
        records.append(wrapper('speaker_events',key,duration,[]))
        provenance.append(dict(clip_id=key,review_status='pending',training_eligible=False,origin='observed_audio_numeric_proposals',teacher_model=None,teacher_version=None,prompt_sha256=job['prompt_sha256'],actual_visible_inputs=[dict(path=p,sha256=sha(p),role='speaker') for p in [audio,video,txt]],timeline=tm,audio_sample_rate=rate,audio_samples=len(signal),audio_seconds=len(signal)/rate,transcript_present=bool(checked(txt).read_text().strip()),transcript_alignment='unavailable',word_times=None,speaker_channel_role_verified=False,sync_verified=False,empty_events_reason='semantic events not yet annotated; not a no-reaction label',audio_video_offset_s=None,offset_status='unverified',evidence_refs=[f'{key}:audio_sample_clock',f'{key}:frame_pts']))
        print('SOURCE',key,flush=True)
    output('annotations.json',records);output('provenance.json',provenance);output('window_candidates.json',candidates)
if __name__=='__main__':main()
