"""Source-only local CTC alignment and audiovisual active-speaker proposals."""
import json,hashlib,re,sys,subprocess,difflib,math
from pathlib import Path
import numpy as np
import torch,torchaudio,soundfile as sf,cv2

def write(p,x):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2));q.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def normalized_words(text):
    # Digits are excluded explicitly; candidates containing them are not accepted.
    return [(m.group().upper().replace('’',"'"),m.start(),m.end()) for m in re.finditer(r"[A-Za-z]+(?:['’][A-Za-z]+)*",text)]
def quote_words(text,quote,after=-1):
    pos=text.find(quote,after+1)
    if pos<0:raise ValueError('quote_not_ordered')
    words=normalized_words(text)
    ids=[i for i,(_,s,e) in enumerate(words) if s>=pos and e<=pos+len(quote)]
    return pos,ids

def emissions(model,wave):
    n=(len(wave)-400)//320+1
    if n<1:raise ValueError('audio_too_short')
    parts=[]
    for core in range(0,n,1000):
        end=min(n,core+1000);left=max(0,core-50);right=min(n,end+50)
        audio=wave[left*320:(right-1)*320+400].unsqueeze(0).cuda()
        with torch.inference_mode():part=model(audio)[0][0].log_softmax(-1).cpu()
        parts.append(part[core-left:end-left])
    return torch.cat(parts)

def align(emission,text,labels):
    words=normalized_words(text);seq='|'.join(w[0] for w in words);vocab={s:i for i,s in enumerate(labels)}
    ids=[vocab[c] for c in seq]
    path,scores=torchaudio.functional.forced_align(emission.unsqueeze(0),torch.tensor([ids]),blank=0)
    spans=torchaudio.functional.merge_tokens(path[0],scores[0].exp())
    if [s.token for s in spans]!=ids:raise ValueError('CTC_path_token_mismatch')
    result=[];cursor=0
    for word,s,e in words:
        span=spans[cursor:cursor+len(word)];cursor+=len(word)+1
        result.append({'word':word,'char_start':s,'char_end':e,'start_s':span[0].start*.02,'end_s':min(emission.shape[0]*.02+.025,span[-1].end*.02+.005),'ctc_mean_score':float(np.mean([x.score for x in span]))})
    return result

def decode(emission,labels):
    ids=torch.unique_consecutive(emission.argmax(-1)).tolist()
    return ''.join(labels[i] for i in ids if i).replace('|',' ').strip()

def video_features(path):
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','json',str(path)]))
    pts=np.array([float(r['best_effort_timestamp_time']) for r in probe['frames']]);assert len(pts)>1 and np.all(np.diff(pts)>0)
    duration=pts[-1]+np.median(np.diff(pts));times=np.arange(.02,duration,.04)
    idx=np.searchsorted(pts,times);idx=np.minimum(idx,len(pts)-1)
    previous=np.maximum(idx-1,0);idx=np.where(abs(pts[previous]-times)<abs(pts[idx]-times),previous,idx)
    wanted={int(i) for i in idx};frames={};cap=cv2.VideoCapture(str(path));i=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        if i in wanted:
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);gray=cv2.resize(gray,(224,224));frames[i]=gray[56:168,56:168]
        i+=1
    cap.release();assert i==len(pts)
    return np.stack([frames[int(i)] for i in idx]),{'first_pts':float(pts[0]),'last_pts':float(pts[-1]),'duration_s':float(duration),'source_frame_count':i,'resampled_hz':25,'max_resampling_error_s':float(np.max(abs(pts[idx]-times)))}

class ActiveSpeaker:
    def __init__(self):
        sys.path.insert(0,'/teachers/TalkNet-ASD');sys.path.insert(0,'/teachers/python')
        from model.talkNetModel import talkNetModel
        from python_speech_features import mfcc
        self.mfcc=mfcc;self.model=talkNetModel().cuda().eval();self.fc=torch.nn.Linear(256,2).cuda().eval()
        state=torch.load('/teachers/pretrain_TalkSet.model',map_location='cpu',weights_only=True)
        self.model.load_state_dict({k[len('model.'):]:v for k,v in state.items() if k.startswith('model.')},strict=True)
        self.fc.load_state_dict({k[len('lossAV.FC.'):]:v for k,v in state.items() if k.startswith('lossAV.FC.')},strict=True)
    def score(self,wave,video):
        audio=self.mfcc(wave.numpy()*32768,16000,numcep=13,winlen=.025,winstep=.010)
        n=min(len(video),len(audio)//4);all_scores=[]
        for length in (50,100,150):
            scores=[]
            for start in range(0,n,length):
                end=min(n,start+length)
                if end-start<4:
                    scores.extend([float('nan')]*(end-start));continue
                a=torch.tensor(audio[start*4:end*4],dtype=torch.float32).unsqueeze(0).cuda()
                v=torch.tensor(video[start:end],dtype=torch.float32).unsqueeze(0).cuda()
                with torch.inference_mode():
                    ae=self.model.forward_audio_frontend(a);ve=self.model.forward_visual_frontend(v)
                    ae,ve=self.model.forward_cross_attention(ae,ve)
                    av=self.model.forward_audio_visual_backend(ae,ve)
                    logits=self.fc(av).cpu().numpy()
                scores.extend(logits[:,1].tolist())
            all_scores.append(scores)
        return np.nanmean(all_scores,axis=0)

def main():
    torch.set_num_threads(4)
    out=Path('/outputs');job=json.loads(Path('/job/job.json').read_text())
    assert not Path('/home/zhengshiyi/react/data').exists() and not Path('/inputs/listener').exists()
    assert not Path('/home/zhengshiyi/.config/react2025').exists()
    print('SOURCE_ONLY_SANDBOX_OK',flush=True)
    bundle=torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model=bundle.get_model().cuda().eval();labels=bundle.get_labels();asd=ActiveSpeaker()
    for row in job['records']:
        dest=out/row['id'];dest.mkdir(exist_ok=True)
        if (dest/'result.json').exists():continue
        rel=row['relative'];audio=Path('/inputs/audio')/(rel+'.wav');video=Path('/inputs/video')/(rel+'.mp4');txt=Path('/inputs/text')/(rel+'.txt')
        text=txt.read_text();assert sha(txt)==row['transcript_sha256']
        raw,sr=sf.read(audio,always_2d=True);wave=torch.from_numpy(raw.mean(1)).float();wave=torchaudio.functional.resample(wave,sr,16000)
        emission=emissions(model,wave);words=align(emission,text,labels)
        write(dest/'words.json',words)
        vf,vt=video_features(video);scores=asd.score(wave,vf)
        # Deliberately shifted audio control, never changes actual paired timelines.
        shifted=torch.roll(wave,3*16000);control=asd.score(shifted,vf)
        write(dest/'active_speaker_scores.json',{'times_s':(np.arange(len(scores))*.04+.02).tolist(),'raw_speaking_logit':[float(x) if np.isfinite(x) else None for x in scores],'shifted_audio_3s_logit':[float(x) if np.isfinite(x) else None for x in control],'scores_are_probabilities':False,'control_is_diagnostic_only':True})
        events=[];previous=-1
        for j,c in enumerate(row['candidates']):
            pos,ids=quote_words(text,c['transcript_evidence'],previous);previous=pos
            if not ids:continue
            chosen=[words[i] for i in ids];start=chosen[0]['start_s'];end=chosen[-1]['end_s']
            estimate=decode(emission[max(0,int(start/.02)):min(len(emission),int(end/.02)+1)],labels)
            expected=' '.join(w['word'] for w in chosen)
            agreement=difflib.SequenceMatcher(None,expected,estimate,autojunk=False).ratio()
            ctc=float(np.mean([w['ctc_mean_score'] for w in chosen]));a=max(0,int(start*25));z=min(len(scores),int(math.ceil(end*25)))
            actual=scores[a:z];negative=control[a:z]
            finite=np.isfinite(actual)&np.isfinite(negative)
            fraction=float(np.mean(actual[finite]>0)) if finite.any() else None
            margin=float(np.mean(actual[finite]-negative[finite])) if finite.any() else None
            timing_ok=ctc>=.5 and agreement>=.6 and all(w['end_s']-w['start_s']<=2. for w in chosen) and not bool(re.search(r'\d',c['transcript_evidence']))
            # Conservative, preregistered heuristic. NOT human verification or calibrated accuracy.
            role_ok=timing_ok and fraction is not None and fraction>=.8 and margin>=.5
            events.append({**c,'event_id':row['id']+f'_e{j:03d}','start_s':start,'end_s':end,'time_reference':'speaker_audio_sample_clock','time_status':'automatic_supported' if timing_ok else 'automatic_uncertain','speaker_attribution':'source_speaker_candidate' if role_ok else 'unknown','role_status':'automatic_audiovisual_support' if role_ok else 'unknown','ctc_mean_score':ctc,'greedy_asr_char_agreement':agreement,'greedy_asr_text':estimate,'active_speech_fraction':fraction,'shift_control_logit_margin':margin,'automatic_weak_candidate':role_ok,'human_reviewed':False,'training_eligible':False,'sync_status':'container_origin_assumed_not_verified'})
        result={'clip_id':row['id'],'split':'train','events':events,'audio_duration_s':len(wave)/16000,'original_sample_rate':sr,'original_channels':raw.shape[1],'video_timeline':vt,'actual_visible_inputs':[{'role':'speaker','path':str(p),'sha256':sha(p)} for p in (audio,video,txt)],'teacher_provenance':job['teachers'],'human_reviewed':False,'human_review_policy':'omitted_by_user','training_eligible':False,'limitations':['CTC forced alignment can force incorrect transcript; thresholds are uncalibrated.','Active-speaker model domain and crop transfer are unvalidated.','Container audio/video origins assumed; true synchronization not independently verified.','Roles are model proposals, not confirmed identity; no listener data accessed.']}
        write(dest/'result.json',result)
        print(row['id'],len(events),sum(e['time_status']=='automatic_supported' for e in events),sum(e['automatic_weak_candidate'] for e in events),flush=True)
    print('FINISHED',flush=True)
if __name__=='__main__':main()
