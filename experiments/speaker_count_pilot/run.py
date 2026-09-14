"""CPU-only, at-most-two-speaker diagnostic. No inferred cluster is an identity."""
import sys,json,hashlib,csv
from pathlib import Path
CACHE=Path.home()/'.cache/react2025'
sys.path.insert(0,str(CACHE/'speaker_diagnostic_deps'))
sys.path.insert(0,str(CACHE/'speaker_diagnostic_models/silero'))
import numpy as np,torch,torchaudio,soundfile as sf
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from speechbrain.lobes.features import Fbank
from speechbrain.processing.features import InputNormalization
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
from utils_vad import get_speech_timestamps
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/reaction_flow/speaker_count_pilot_v1'
POLICY=dict(max_speakers=2,vad_threshold=.5,min_speech_ms=250,min_silence_ms=200,
 window_max_s=2.,window_min_s=.8,minimum_total_evidence_s=6,
 minimum_cluster_windows=2,minimum_cluster_seconds=2.4,
 one_cluster_pairwise_p90_max=[.20,.25,.30],two_cluster_centroid_distance_min=[.30,.35,.40],
 silhouette_min=.35,within_cluster_cosine_radius_p90_max=.30,
 decision='all three preset policies must agree; otherwise uncertain; engineering gates, no calibration',
 audio_origin='first WAV sample',window_overlap=False,overlapping_speech_detector=False,
 scope='33 TRAIN speaker WAV files only; no listener access, paid API or GPU',
 interpretation='estimated number of sustained distinguishable voices, not proof of exact speaker count; brief second voice may be missed')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def decide(emb,durations):
 n=len(emb)
 if n<4 or sum(durations)<6:return dict(count=None,reason='insufficient_voice_evidence',labels=[None]*n)
 dist=np.clip(1-emb@emb.T,0,2);np.fill_diagonal(dist,0)
 p90=float(np.quantile(dist[np.triu_indices(n,1)],.9))
 try:lab=AgglomerativeClustering(n_clusters=2,metric='precomputed',linkage='average').fit_predict(dist)
 except TypeError:lab=AgglomerativeClustering(n_clusters=2,affinity='precomputed',linkage='average').fit_predict(dist)
 # Stable labels by first chronological occurrence, not gender/identity.
 if lab[0]==1:lab=1-lab
 centers=np.stack([emb[lab==i].mean(0) for i in range(2)]);centers/=np.linalg.norm(centers,axis=1,keepdims=True)
 separation=float(1-centers[0]@centers[1]);sil=float(silhouette_score(dist,lab,metric='precomputed'))
 sizes=[int((lab==i).sum()) for i in range(2)];secs=[float(np.asarray(durations)[lab==i].sum()) for i in range(2)]
 radii=[float(np.quantile(1-emb[lab==i]@centers[i],.9)) for i in range(2)]
 support=min(sizes)>=2 and min(secs)>=2.4 and max(radii)<=.30 and sil>=.35
 choices=[]
 for one,two in zip(POLICY['one_cluster_pairwise_p90_max'],POLICY['two_cluster_centroid_distance_min']):
  a=p90<=one;b=support and separation>=two
  choices.append(1 if a and not b else 2 if b and not a else None)
 count=choices[0] if choices[0] is not None and len(set(choices))==1 else None
 return dict(count=count,reason='stable_heuristic_estimate' if count else 'threshold_sensitive_or_ambiguous',
  labels=([0]*n if count==1 else lab.tolist() if count==2 else [None]*n),
  diagnostic_two_cluster_labels=lab.tolist(),pairwise_cosine_distance_p90=p90,
  centroid_cosine_distance=separation,silhouette=sil,cluster_windows=sizes,cluster_seconds=secs,
  cluster_radius_p90=radii,policy_counts=choices)
def windows(segments):
 result=[];short=[]
 for seg in segments:
  start,end=seg['start']/16000,seg['end']/16000;duration=end-start
  if duration<.8:short.append(dict(start_s=start,end_s=end,reason='too_short_for_embedding'));continue
  n=max(1,int(np.ceil(duration/2)))
  for a,b in zip(np.linspace(start,end,n+1)[:-1],np.linspace(start,end,n+1)[1:]):result.append(dict(start_s=float(a),end_s=float(b)))
 return result,short
@torch.inference_mode()
def main():
 torch.set_num_threads(4);torch.manual_seed(123)
 OUT.mkdir(parents=True,exist_ok=False)
 modeldir=CACHE/'speaker_diagnostic_models/ecapa';model=ECAPA_TDNN(input_size=80,channels=[1024,1024,1024,1024,3072],kernel_sizes=[5,3,3,3,1],dilations=[1,2,3,4,1],attention_channels=128,lin_neurons=192)
 model.load_state_dict(torch.load(modeldir/'embedding_model.ckpt',map_location='cpu'));model.eval()
 fbank=Fbank(n_mels=80);norm=InputNormalization(norm_type='sentence',std_norm=False)
 vad=torch.jit.load(str(CACHE/'speaker_diagnostic_models/silero/silero_vad.jit'),map_location='cpu').eval()
 job=ROOT/'runs/reaction_flow/semantic_alignment_pilot_v1/job/job.json'
 write(OUT/'POLICY.json',dict(**POLICY,job_sha256=sha(job),script_sha256=sha(__file__),
  model_hashes={str(p.relative_to(CACHE/'speaker_diagnostic_models')):sha(p) for p in [modeldir/'embedding_model.ckpt',modeldir/'hyperparams.yaml',CACHE/'speaker_diagnostic_models/silero/silero_vad.jit',CACHE/'speaker_diagnostic_models/silero/utils_vad.py']},
  embedding_recipe='official Fbank80 + sentence mean normalization + ECAPA; raw encoder embedding L2 normalization, no global embedding normalization'))
 summary=[]
 for row in json.loads(job.read_text())['records']:
  cid=row['id'];path=ROOT/'data/train/audio/speaker'/(row['relative']+'.wav');raw,sr=sf.read(path,always_2d=True)
  wave=torchaudio.functional.resample(torch.from_numpy(raw.mean(1)).float(),sr,16000)
  vad.reset_states();segments=get_speech_timestamps(wave,vad,sampling_rate=16000,threshold=.5,min_speech_duration_ms=250,min_silence_duration_ms=200,speech_pad_ms=0)
  ws,short=windows(segments);es=[]
  for w in ws:
   chunk=wave[round(w['start_s']*16000):round(w['end_s']*16000)].unsqueeze(0)
   e=model(norm(fbank(chunk),torch.ones(1))).flatten().numpy();es.append(e/np.linalg.norm(e))
  emb=np.stack(es) if es else np.empty((0,192));np.save(OUT/(cid+'_embeddings.npy'),emb)
  result=decide(emb,[w['end_s']-w['start_s'] for w in ws])
  for w,l in zip(ws,result['labels']):w['voice_cluster']=None if l is None else 'voice_'+str(l)
  source=ROOT/'runs/reaction_flow/semantic_alignment_pilot_v1/source'/cid
  words=json.loads((source/'words.json').read_text());assigned=[]
  for w in words:
   a,b=w.get('start_s'),w.get('end_s');label=None
   if a is not None and b is not None and b>a:
    ov={}
    for q in ws:
     if q['voice_cluster'] is not None:ov[q['voice_cluster']]=ov.get(q['voice_cluster'],0)+max(0,min(b,q['end_s'])-max(a,q['start_s']))
    if ov:
     best=max(ov,key=ov.get)
     if ov[best]/(b-a)>=.8:label=best
   assigned.append(dict(word=w['word'],start_s=a,end_s=b,voice_cluster=label))
  # Existing ASD is supporting evidence only. No automatic face-identity assignment.
  scores=json.loads((source/'active_speaker_scores.json').read_text());st=np.asarray(scores['times_s']);sv=np.array([float(x) if x is not None else np.nan for x in scores['raw_speaking_logit']]);asd=[]
  for c in range(result['count'] or 0):
   mask=np.zeros(len(st),bool)
   for w in ws:
    if w['voice_cluster']=='voice_'+str(c):mask|=(st>=w['start_s'])&(st<w['end_s'])
   keep=mask&np.isfinite(sv)
   asd.append(dict(voice_cluster='voice_'+str(c),finite_frames=int(keep.sum()),requested_score_frames=int(mask.sum()),mean_logit=float(sv[keep].mean()) if keep.any() else None,positive_fraction=float((sv[keep]>0).mean()) if keep.any() else None,source_identity='unknown',timeline='legacy ASD time grid, offset not corrected; diagnostic only'))
  record=dict(id=cid,relative=row['relative'],split='train',audio_sha256=sha(path),audio_channels=raw.shape[1],audio_duration_s=len(wave)/16000,
   estimated_speaker_count=result['count'],human_reviewed=False,annotation_regime='audio_count_diagnostic',
   decision=result,vad_intervals=[dict(start_s=s['start']/16000,end_s=s['end']/16000) for s in segments],windows=ws,
   short_unassigned_segments=short,words=assigned,existing_ASD_by_voice=asd,
   original_words_sha256=sha(source/'words.json'),ASD_sha256=sha(source/'active_speaker_scores.json'),
   warnings=['No overlap detector; windows can contain a speaker change.','One sustained voice does not rule out a brief second voice.','Voice clusters are local to this recording, not source/listener identities.'])
  write(OUT/(cid+'.json'),record)
  summary.append(dict(id=cid,relative=row['relative'],estimated_speaker_count=result['count'],windows=len(ws),short_segments=len(short),reason=result['reason']))
  write(OUT/'progress.json',dict(completed=len(summary),total=33,last=summary[-1]));print(cid,result['count'],len(ws),flush=True)
 write(OUT/'SUMMARY.json',dict(total=len(summary),counts={str(k):sum(r['estimated_speaker_count']==k for r in summary) for k in [1,2,None]},records=summary))
 with open(OUT/'counts.csv','w') as f:
  w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
if __name__=='__main__':main()
