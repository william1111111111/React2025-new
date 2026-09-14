"""Resumable bounded CPU pool for frozen TRAIN-only voice-count screening."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import json,time,hashlib,csv,argparse,traceback,multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from run import (ROOT,CACHE,POLICY as VOICE_POLICY,torch,torchaudio,sf,np,Fbank,InputNormalization,ECAPA_TDNN,get_speech_timestamps,windows,decide,sha)
from spectral import spectral,POLICY as SPECTRAL_POLICY
from sklearn.metrics import adjusted_rand_score
OUT=ROOT/'runs/reaction_flow/speaker_count_full_train_v1'
def atomic(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False));q.replace(p)
def init():
 global MODEL,FBANK,NORM,VAD
 torch.set_num_threads(1);torch.manual_seed(123)
 MODEL=ECAPA_TDNN(input_size=80,channels=[1024,1024,1024,1024,3072],kernel_sizes=[5,3,3,3,1],dilations=[1,2,3,4,1],attention_channels=128,lin_neurons=192)
 MODEL.load_state_dict(torch.load(CACHE/'speaker_diagnostic_models/ecapa/embedding_model.ckpt',map_location='cpu'));MODEL.eval()
 FBANK=Fbank(n_mels=80);NORM=InputNormalization(norm_type='sentence',std_norm=False)
 VAD=torch.jit.load(str(CACHE/'speaker_diagnostic_models/silero/silero_vad.jit'),map_location='cpu').eval()
def classify(e,ws,short):
 runs=[spectral(e,p) for p in SPECTRAL_POLICY['neighbor_fractions']];main=runs[1];k=main['count'];labels=main['labels'];duration=np.array([w['end_s']-w['start_s'] for w in ws]);reasons=[]
 if k is None or sum(duration)<6:reasons.append('insufficient_evidence')
 if len(set(r['count'] for r in runs))!=1:reasons.append('neighbor_sensitivity')
 aris=[float(adjusted_rand_score(labels,r['labels'])) for r in runs] if k else []
 if aris and min(aris)<.8:reasons.append('partition_sensitivity')
 if k==2:
  for c in range(2):
   mask=np.asarray(labels)==c
   if mask.sum()<2 or duration[mask].sum()<2.4:reasons.append('small_cluster')
 if short:reasons.append('brief_voice_not_resolved')
 stable=not any(r!='brief_voice_not_resolved' for r in reasons)
 return dict(candidate_speaker_count=k,graph_stable=stable,screening_bucket=f'{k}_sustained_voice_candidate' if stable else 'uncertain',uncertainty_reasons=sorted(set(reasons)),spectral_runs=runs,partition_ARI=aris,windows=[dict(**w,candidate_voice=None if l is None else 'voice_'+str(l)) for w,l in zip(ws,labels)])
@torch.inference_mode()
def process(row):
 cid=row['id'];p=OUT/'records'/(cid+'.json');audio=ROOT/'data/train/audio/speaker'/(row['relative']+'.wav');h=sha(audio)
 if p.exists():
  old=json.loads(p.read_text());assert old['audio_sha256']==h and old['policy_sha256']==row['policy_sha256'];return summary(old)
 t=time.time();raw,sr=sf.read(audio,always_2d=True);wave=torchaudio.functional.resample(torch.from_numpy(raw.mean(1)).float(),sr,16000)
 VAD.reset_states();segments=get_speech_timestamps(wave,VAD,sampling_rate=16000,threshold=.5,min_speech_duration_ms=250,min_silence_duration_ms=200,speech_pad_ms=0)
 ws,short=windows(segments);emb=[]
 for w in ws:
  chunk=wave[round(w['start_s']*16000):round(w['end_s']*16000)].unsqueeze(0)
  e=MODEL(NORM(FBANK(chunk),torch.ones(1))).flatten().numpy();emb.append(e/np.linalg.norm(e))
 e=np.stack(emb) if emb else np.empty((0,192));assert np.isfinite(e).all()
 np.save(OUT/'embeddings'/(cid+'.npy'),e)
 result=dict(id=cid,relative=row['relative'],split='train',audio_sha256=h,policy_sha256=row['policy_sha256'],audio_duration_s=len(wave)/16000,audio_channels=raw.shape[1],source_identity='unknown',human_reviewed=False,training_eligible=False,annotation_regime='audio_count_diagnostic',**classify(e,ws,short),short_unassigned_segments=short,vad_intervals=[dict(start_s=s['start']/16000,end_s=s['end']/16000) for s in segments],conservative_diagnostic=decide(e,[w['end_s']-w['start_s'] for w in ws]),runtime_s=time.time()-t)
 atomic(p,result);return summary(result)
def summary(r):return {k:r[k] for k in ['id','relative','candidate_speaker_count','graph_stable','screening_bucket','uncertainty_reasons']}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=24);a=ap.parse_args();assert 1<=a.workers<=32
 OUT.mkdir(exist_ok=True);(OUT/'records').mkdir(exist_ok=True);(OUT/'embeddings').mkdir(exist_ok=True)
 import fcntl
 lock=open(OUT/'queue.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 policy=dict(voice=VOICE_POLICY,spectral=SPECTRAL_POLICY,workers=a.workers,torch_threads_per_worker=1,device='cpu',code_hashes={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},model_hashes=json.loads((ROOT/'runs/reaction_flow/speaker_count_pilot_v1/POLICY.json').read_text())['model_hashes'],notes='Frozen pilot methods; counts not calibrated; no media upload or paid API; no ASR rerun, no automatic source identity or semantic-cache changes.')
 pp=OUT/'POLICY.json'
 if pp.exists():assert json.loads(pp.read_text())==policy,'policy/code changed; use a new output directory'
 else:atomic(pp,policy)
 ph=sha(pp);audio_root=ROOT/'data/train/audio/speaker';rows=[]
 for p in sorted(audio_root.rglob('*.wav')):
  rel=str(p.relative_to(audio_root).with_suffix(''));rows.append(dict(id='rec_'+hashlib.sha256(rel.encode()).hexdigest()[:16],relative=rel,policy_sha256=ph))
 mp=OUT/'MANIFEST.json'
 if mp.exists():assert json.loads(mp.read_text())==rows
 else:atomic(mp,rows)
 start=time.time();done=[];errors=[]
 atomic(OUT/'progress.json',dict(status='running',total=len(rows),completed=0,errors=0,workers=a.workers,pid=os.getpid(),started_unix=start))
 with ProcessPoolExecutor(max_workers=a.workers,initializer=init,mp_context=multiprocessing.get_context('spawn')) as pool:
  futures={pool.submit(process,row):row for row in rows}
  for fut in as_completed(futures):
   row=futures[fut]
   try:done.append(fut.result())
   except Exception as ex:
    errors.append(dict(id=row['id'],error=str(ex),traceback=traceback.format_exc()));atomic(OUT/'errors.json',errors)
   status=dict(status='running',total=len(rows),completed=len(done),errors=len(errors),workers=a.workers,pid=os.getpid(),elapsed_s=time.time()-start,updated_unix=time.time(),buckets={k:sum(r['screening_bucket']==k for r in done) for k in ['1_sustained_voice_candidate','2_sustained_voice_candidate','uncertain']})
   atomic(OUT/'progress.json',status)
   if (len(done)+len(errors))%25==0:print(json.dumps(status),flush=True)
 done.sort(key=lambda r:r['relative'])
 with open(OUT/'counts.csv','w') as f:
  w=csv.DictWriter(f,fieldnames=list(done[0]) if done else ['id']);w.writeheader();w.writerows(done)
 status['status']='complete' if len(done)==len(rows) and not errors else 'incomplete';atomic(OUT/'progress.json',status);atomic(OUT/'SUMMARY.json',dict(**status,records=done,failed=errors));print(json.dumps(status),flush=True)
if __name__=='__main__':main()
