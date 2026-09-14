"""Separate TRAIN-only low-level recovery and short-voice context diagnostics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import sys,json,time,argparse,hashlib,subprocess,multiprocessing,collections
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/speaker_count_pilot'))
import full_train as F
import numpy as np,soundfile as sf,torch,torchaudio
from scipy.ndimage import gaussian_filter1d
BASE=ROOT/'runs/reaction_flow/speaker_count_full_train_v1';OUT=ROOT/'runs/reaction_flow/voice_refinement_v1'
POLICY=dict(split='train',gain_target_rms=.05,max_gain_db=60,peak_ceiling=.95,low_rms_threshold=.001,
 gain_sensitivity='repeat VAD at half main gain; descriptive interval IoU, no threshold retuning',
 context_flank_s=2,context_min_embedding_s=.8,minimum_flank_vad_fraction=.5,
 roles='unknown; no new role or training eligibility',
 feature_time='actual video PTS associated by equal row count; zero audio offset for descriptive evidence only',
 allowed_features='first15 anonymous binary AU switching and first52 global expression motion, no guessed mouth indices',
 scope='95 low-RMS insufficient-evidence recordings and all 79 small-cluster cases; no API, GPU, listener or DEV')
def atomic(p,x):F.atomic(p,x)
def read(p):return json.loads(p.read_text())
def load_audio(r):
 p=ROOT/'data/train/audio/speaker'/(r['relative']+'.wav');assert F.sha(p)==r['audio_sha256'];x,sr=sf.read(p,always_2d=True);mono=x.mean(1)
 return torchaudio.functional.resample(torch.from_numpy(mono).float(),sr,16000),p
@torch.inference_mode()
def emb(x):
 e=F.MODEL(F.NORM(F.FBANK(x.unsqueeze(0)),torch.ones(1))).flatten().numpy();return e/np.linalg.norm(e)
def vad(x):
 F.VAD.reset_states();return F.get_speech_timestamps(x,F.VAD,sampling_rate=16000,threshold=.5,min_speech_duration_ms=250,min_silence_duration_ms=200,speech_pad_ms=0)
def duration(ss):return sum(s['end']-s['start'] for s in ss)/16000
# IoU on continuous sample intervals, no frame-origin approximation.
def interval_iou(a,b):
 intersection=sum(max(0,min(x['end'],y['end'])-max(x['start'],y['start'])) for x in a for y in b);union=sum(x['end']-x['start'] for x in a+b)-intersection
 return intersection/union if union else None
def gain_for(x):
 rms=float(torch.sqrt(torch.mean(x*x)));peak=float(x.abs().max());g=min(10**(POLICY['max_gain_db']/20),POLICY['gain_target_rms']/max(rms,1e-12),POLICY['peak_ceiling']/max(peak,1e-12));return max(1.,g) if peak<=.95 else g
@torch.inference_mode()
def low(row):
 old=read(BASE/'records'/(row['id']+'.json'));x,audio=load_audio(old);gain=gain_for(x);y=x*gain
 ss=vad(y);half=vad(x*(gain/2));ws,short=F.windows(ss);es=[emb(y[round(w['start_s']*16000):round(w['end_s']*16000)]) for w in ws];e=np.stack(es) if es else np.empty((0,192))
 np.save(OUT/'low_gain'/(row['id']+'_embeddings.npy'),e);new=F.classify(e,ws,short)
 result=dict(id=row['id'],relative=old['relative'],audio_sha256=old['audio_sha256'],source_record_sha256=F.sha(BASE/'records'/(row['id']+'.json')),policy_sha256=F.sha(OUT/'POLICY.json'),gain=gain,gain_db=float(20*np.log10(gain)),original_rms=float(torch.sqrt(torch.mean(x*x))),processed_rms=float(torch.sqrt(torch.mean(y*y))),processed_peak=float(y.abs().max()),exact_zero_fraction=float((x==0).float().mean()),nonzero_samples=int((x!=0).sum()),processed_float_sha256=hashlib.sha256(y.numpy().tobytes()).hexdigest(),audio_duration_s=len(x)/16000,
 original_bucket=old['screening_bucket'],original_vad_speech_s=sum(w['end_s']-w['start_s'] for w in old['vad_intervals']),original_windows=len(old['windows']),new_vad_speech_s=duration(ss),half_gain_vad_speech_s=duration(half),gain_vad_iou=interval_iou(ss,half),new_windows=len(ws),short_unassigned_segments=short,new_result=new,source_identity='unknown',human_reviewed=False,training_eligible=False)
 atomic(OUT/'low_gain'/(row['id']+'.json'),result);return dict(id=row['id'],bucket=new['screening_bucket'],new_vad_speech_s=duration(ss),new_windows=len(ws),gain_db=result['gain_db'])
def coverage(a,b,intervals):return sum(max(0,min(b,w['end_s'])-max(a,w['start_s'])) for w in intervals)/max(b-a,1e-9)
def intervals_summary(t,v,a,b):
 k=(t>=a)&(t<b)&np.isfinite(v);return dict(frames=int(k.sum()),mean=float(v[k].mean()) if k.any() else None,positive_fraction=float((v[k]>0).mean()) if k.any() else None)
@torch.inference_mode()
def context(row):
 old=read(BASE/'records'/(row['id']+'.json'));x,audio=load_audio(old);e=np.load(BASE/'embeddings'/(row['id']+'.npy'));runs=old['spectral_runs'];labels=np.asarray(runs[1]['labels']);ws=old['windows'];ds=np.array([w['end_s']-w['start_s'] for w in ws]);secs=[ds[labels==i].sum() for i in range(2)];minor=int(np.argmin(secs));major=1-minor
 # Minority is only a short-cluster proposal, never an identity.
 centers=np.stack([e[labels==i].mean(0) for i in range(2)]);centers/=np.linalg.norm(centers,axis=1,keepdims=True)
 rel=old['relative'];ap=ROOT/'data/train/facial-attributes/speaker'/(rel+'.npy');ep=ROOT/'data/train/coefficients/speaker'/(rel+'.npy');vp=ROOT/'data/train/video-face-crop/speaker'/(rel+'.mp4')
 j=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=best_effort_timestamp_time','-of','json',str(vp)]));pts=np.array([float(q['best_effort_timestamp_time']) for q in j['frames']]);au=np.load(ap)[:,:15];ex=np.load(ep).reshape(-1,58)[:,:52]
 assert len(pts)==len(au)==len(ex) and np.all(np.diff(pts)>0)
 dt=float(np.median(np.diff(pts)));signals={n:gaussian_filter1d(np.abs(np.diff(v,axis=0,prepend=v[:1])).mean(1),.08/dt) for n,v in [('anonymous_AU_switching',au),('global_expression_motion',ex)]}
 # Only previously computed TRAIN ASD evidence; no new visual teacher execution.
 asdpath=None
 for item in read(OUT/'ASD_INDEX.json'):
  if item['relative']==rel:asdpath=Path(item['path']);break
 asd=read(asdpath) if asdpath else None;result=[]
 for i in np.flatnonzero(labels==minor):
  a,b=ws[i]['start_s'],ws[i]['end_s'];segments=[('before',max(0,a-2),a),('core',a,b),('after',b,min(len(x)/16000,b+2))];items=[]
  for kind,s,t in segments:
   fraction=coverage(s,t,old['vad_intervals']);ev=None
   if t-s>=.8 and (kind=='core' or fraction>=.5):
    z=e[i] if kind=='core' else emb(x[round(s*16000):round(t*16000)])
    ev=dict(cosine_to_majority=float(z@centers[major]),cosine_to_minority=float(z@centers[minor]),note='core minority similarity includes itself; not independent evidence')
   item=dict(kind=kind,start_s=s,end_s=t,vad_fraction=fraction,voice_embedding=ev,feature_activity={n:intervals_summary(pts,v,s,t) for n,v in signals.items()})
   if asd:
    st=np.asarray(asd['times_s']);sv=np.array([float(z) if z is not None else np.nan for z in asd['raw_speaking_logit']]);item['legacy_ASD_uncorrected']=intervals_summary(st,sv,s,t)
   items.append(item)
  result.append(dict(core_window_index=int(i),segments=items))
 record=dict(id=row['id'],relative=rel,core_contexts=result,original_count=old['candidate_speaker_count'],minority_seconds=float(secs[minor]),minority_windows=int((labels==minor).sum()),source_identity='unknown',human_reviewed=False,training_eligible=False,
  audio_sha256=old['audio_sha256'],policy_sha256=F.sha(OUT/'POLICY.json'),input_hashes={n:F.sha(p) for n,p in [('au',ap),('expression',ep),('video',vp),('embedding',BASE/'embeddings'/(row['id']+'.npy')),('source_record',BASE/'records'/(row['id']+'.json'))]},ASD_sha256=F.sha(asdpath) if asdpath else None,
  mapping='zero-offset WAV seconds to feature-associated raw PTS, descriptive and unverified',decision='unknown; context evidence does not promote source identity or speaker-count eligibility')
 atomic(OUT/'context'/(row['id']+'.json'),record);return dict(id=row['id'],contexts=len(result),with_ASD=asd is not None)
def prepare():
 OUT.mkdir(exist_ok=False);(OUT/'low_gain').mkdir();(OUT/'context').mkdir()
 rows=[read(p) for p in sorted((BASE/'records').glob('*.json'))];lows=[];contexts=[]
 for r in rows:
  if r['screening_bucket']=='uncertain' and 'insufficient_evidence' in r['uncertainty_reasons']:
   x,sr=sf.read(ROOT/'data/train/audio/speaker'/(r['relative']+'.wav'),always_2d=True);rms=float(np.sqrt(np.mean(x.mean(1)**2)))
   if rms<.001:lows.append(dict(id=r['id'],relative=r['relative']))
  if 'small_cluster' in r['uncertainty_reasons']:contexts.append(dict(id=r['id'],relative=r['relative']))
 assert len(lows)==95 and len(contexts)==79
 index=[]
 for job,src in [(ROOT/'runs/reaction_flow/semantic_alignment_pilot_v1/job/job.json',ROOT/'runs/reaction_flow/semantic_alignment_pilot_v1/source'),(ROOT/'runs/reaction_flow/semantic_auto_weak_v1/expansion/job/job.json',ROOT/'runs/reaction_flow/semantic_auto_weak_v1/expansion/source')]:
  if not job.exists():continue
  for r in read(job)['records']:
   p=src/r['id']/'active_speaker_scores.json'
   if p.exists():index.append(dict(relative=r['relative'],path=str(p)))
 atomic(OUT/'ASD_INDEX.json',index);atomic(OUT/'MANIFEST.json',dict(low_gain=lows,context=contexts));atomic(OUT/'POLICY.json',dict(**POLICY,code_hashes={str(p.relative_to(ROOT)):F.sha(p) for p in [Path(__file__),ROOT/'experiments/speaker_count_pilot/full_train.py',ROOT/'experiments/speaker_count_pilot/run.py',ROOT/'experiments/speaker_count_pilot/spectral.py']},parent_policy_sha256=F.sha(BASE/'POLICY.json')))
 print('prepared',len(lows),len(contexts),len(index))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','low_gain','context']);ap.add_argument('--workers',type=int,default=12);args=ap.parse_args()
 if args.mode=='prepare':prepare();return
 import fcntl
 lock=open(OUT/(args.mode+'.lock'),'w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 rows=read(OUT/'MANIFEST.json')[args.mode];done=[];errors=[];t=time.time();fn=low if args.mode=='low_gain' else context
 with ProcessPoolExecutor(max_workers=args.workers,initializer=F.init,mp_context=multiprocessing.get_context('spawn')) as pool:
  futs={pool.submit(fn,row):row for row in rows}
  for fut in as_completed(futs):
   try:done.append(fut.result())
   except Exception as err:
    import traceback
    errors.append(dict(id=futs[fut]['id'],error=str(err),traceback=traceback.format_exc()))
   atomic(OUT/(args.mode+'_progress.json'),dict(completed=len(done),total=len(rows),errors=errors,elapsed_s=time.time()-t))
 atomic(OUT/(args.mode+'_SUMMARY.json'),dict(completed=len(done),total=len(rows),errors=errors,elapsed_s=time.time()-t,records=done));print(args.mode,len(done),len(errors),flush=True)
if __name__=='__main__':main()
