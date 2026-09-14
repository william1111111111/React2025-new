import os,json,base64,io,time,hashlib,argparse
from pathlib import Path
import requests,numpy as np,soundfile as sf
from scipy.signal import resample_poly
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/reaction_flow/audio_llm_probe_v1'
PROMPT='''Listen to the attached audio itself. Estimate the number of distinct human speaking voices: 1, 2, or unknown. There are at most two speakers, but do not invent a second voice from pitch or volume changes. Background noise is not a speaker. Do not infer identity, gender, or source/listener role. Brief interjections can be a second speaker; if unclear return unknown. Return JSON only with: audio_accessible (boolean), speaker_count (1,2,null), short_transcript (the first clearly audible 5-15 words, or empty if unintelligible), evidence (brief audible observations), suspected_voice_changes (list of approximate start_s/end_s and brief evidence), limitations. Timestamps are rough observations, not verified alignment. Do not claim certainty when audio is weak.'''
def main():
 ap=argparse.ArgumentParser();ap.add_argument('id');ap.add_argument('--model',default='gemini-2.5-flash-lite');ap.add_argument('--silence',action='store_true');args=ap.parse_args()
 OUT.mkdir(exist_ok=True)
 r=json.loads((ROOT/'runs/reaction_flow/speaker_count_full_train_v1/records'/(args.id+'.json')).read_text());p=ROOT/'data/train/audio/speaker'/(r['relative']+'.wav')
 raw,sr=sf.read(p,always_2d=True);wave=raw.mean(1);wave=resample_poly(wave,16000//np.gcd(sr,16000),sr//np.gcd(sr,16000))
 peak=float(np.abs(wave).max());rms=float(np.sqrt(np.mean(wave**2)));gain=min(10**(30/20),.95/max(peak,1e-12),.05/max(rms,1e-12));gain=max(1.,gain) if peak<=.95 else .95/peak
 wave=wave*gain
 if args.silence:wave=np.zeros(16000*8)
 buf=io.BytesIO();sf.write(buf,wave,16000,format='WAV',subtype='PCM_16');data=buf.getvalue()
 suffix='silence' if args.silence else args.id;dest=OUT/(suffix+'_'+args.model.replace('/','_')+'.json');assert not dest.exists()
 body=dict(model=args.model,messages=[dict(role='user',content=[dict(type='text',text=PROMPT),dict(type='input_audio',input_audio=dict(data=base64.b64encode(data).decode(),format='wav'))])],max_tokens=700,temperature=0)
 key=(Path.home()/'.config/react2025/yunwu.key').read_text().strip();t=time.time()
 try:
  response=requests.post('https://yunwu.ai/v1/chat/completions',headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},json=body,timeout=120)
  try:result=response.json()
  except ValueError:result={'non_json_body':response.text[:1000]}
  record=dict(model=args.model,provider='yunwu',source_id=None if args.silence else args.id,source_sha256=None if args.silence else hashlib.sha256(p.read_bytes()).hexdigest(),payload_audio_sha256=hashlib.sha256(data).hexdigest(),synthetic_silence=args.silence,duration_s=len(wave)/16000,gain=gain if not args.silence else None,prompt=PROMPT,elapsed_s=time.time()-t,http_status=response.status_code,response=result)
 except requests.RequestException as e:record=dict(model=args.model,error_type=type(e).__name__,elapsed_s=time.time()-t)
 dest.write_text(json.dumps(record,indent=2,ensure_ascii=False));print(json.dumps(record,ensure_ascii=False))
if __name__=='__main__':main()
