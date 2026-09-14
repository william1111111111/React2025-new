import hashlib,json,random,subprocess
from pathlib import Path
import numpy as np
import torch
from semantic_supervision.models.auto_weak import read_weak,crop_events
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/reaction_flow/bert_semantic_v1';OLD=ROOT/'runs/reaction_flow/semantic_controlled_v1'
PARENT=ROOT/'runs/reaction_flow/task_dynamics_v1/T0-task/attempt_000/checkpoints/step_014000.pt'
PARENT_SHA='e6199183cdc3526e1fc1390c89057e40514af8d2f2101d69a5e1dabb7f43f4d6'
ARMS=('P0-null','P1-byte','P2-bert','P3-bert-time');EVAL=ROOT/'runs/phase24/evaluation_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def text_key(s):return hashlib.sha256(s.encode('utf-8')).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');q.replace(p)
def content_allowed(e):return bool(e.text.strip())
def weak(path,m,split):return read_weak(path,split=split,expected_cache_sha256=m['cache_sha256'],expected_media_hashes=m['media_hashes'],expected_policy_sha256=m['policy_sha256'])
def time_grid(pts,start,n,T=750):
 pts=np.asarray(pts,float);assert n>0 and 0<=start<start+n<=len(pts) and np.all(np.diff(pts)>0)
 end=pts[start+n] if start+n<len(pts) else pts[-1]+np.median(np.diff(pts));q=np.zeros(T);q[:n]=pts[start:start+n]-pts[start];return q,float(end-pts[start])
def state_hash(state):
 h=hashlib.sha256()
 for name,value in sorted(state.items()):h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def verify():
 p=read(OUT/'PROTOCOL.json');assert sha(PARENT)==PARENT_SHA
 if (OUT/'RUN_IDENTITY.json').exists():
  for path,h in read(OUT/'RUN_IDENTITY.json').items():assert sha(path)==h,'changed run artifact '+path
 for path,h in p['frozen_inputs'].items():assert sha(path)==h,'changed '+path
 assert sha(OUT/'schedule.json')==p['schedule_sha256'];assert sha(OUT/'dropout.json')==p['dropout_sha256']
 if (OUT/'INPUT_IDENTITY.json').exists():
  for path,h in read(OUT/'INPUT_IDENTITY.json').items():assert sha(path)==h,'changed source input '+path
 return p
class ContentCache:
 def __init__(self):
  meta=read(OUT/'content/index.json');assert sha(OUT/'content/vectors.npy')==meta['vectors_sha256'];self.vectors=np.load(OUT/'content/vectors.npy');self.index=meta['text_to_row'];self.meta=meta
 def get(self,text):
  k=text_key(text)
  if not text.strip():return None
  if k not in self.index:raise ValueError('missing frozen legal content vector')
  row=self.index[k];return None if row is None else torch.from_numpy(self.vectors[row].copy())
class Sources:
 def __init__(self,split):
  self.split=split;self.rows=read(OUT/('cohort.json' if split=='train' else 'dev_sources.json'));self.events={}
  for name,r in self.rows.items():self.events[name]=weak(Path(r['cache_path']),r['cache'],split)
 def crop(self,name,start,n):
  r=self.rows[name];pts=r['frame_pts'];es=crop_events(self.events[name],pts,start,start+n,trajectory_frames=len(pts));return es,*time_grid(pts,start,n)
def prepare():
 assert sha(PARENT)==PARENT_SHA;OUT.mkdir(parents=True,exist_ok=False)
 cohort=read(OLD/'cohort.json');assert len(cohort)==48;frozen={str(PARENT):PARENT_SHA};train_texts=set();dev_texts=set()
 for name,r in cohort.items():
  r['cache_path']=str(OLD/'train_cache'/(r['clip_id']+'.json'));es=weak(Path(r['cache_path']),r['cache'],'train');assert es
  train_texts.update(e['text'] for e in es if e['text'].strip());frozen[r['cache_path']]=sha(r['cache_path'])
 write(OUT/'cohort.json',cohort)
 by_session={}
 for key,r in cohort.items():by_session.setdefault(key.split('/')[1],[]).append(r['dataset_index'])
 rng=random.Random(123);schedule=[];sessions=sorted(by_session)
 for step in range(14001,15001):
  s=rng.choice(sessions);schedule.append(dict(step=step,session_id=s,source_indices=rng.choices(by_session[s],k=4),crop_occurrences=[rng.randrange(1000000) for _ in range(4)],target_slots=[rng.randrange(4) for _ in range(4)]))
 assert schedule[:500]==read(OLD/'schedule.json');write(OUT/'schedule.json',schedule)
 rng=random.Random(9300123);dropout=[[rng.random()<.1 for _ in range(4)] for _ in range(1000)];write(OUT/'dropout.json',dropout)
 manifest=read(EVAL/'multitarget_development_manifest.json');index=read(OLD/'dev_weak/cache/index.json');jobs={r['id']:r for r in read(OLD/'dev_job/job.json')['records']};dev={};strata={'all':list(range(80)),'non_null':[],'null':[],'timed':[]};crop_manifest={}
 for i,s in enumerate(manifest['sources']):
  cid='rec_'+text_key(s['clip_id'])[:16];row=jobs[cid];path=OLD/'dev_weak/cache'/(cid+'.json');m=index[cid];es=weak(path,m,'val');frozen[str(path)]=sha(path);dev_texts.update(e['text'] for e in es or [] if e['text'].strip())
  vp=ROOT/'data/val/video-face-crop/speaker'/(row['relative']+'.mp4');pts=[float(x['best_effort_timestamp_time']) for x in json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','json',str(vp)]))['frames']];assert len(pts)==s['length']
  for folder,ext in [('audio','.wav'),('video-face-crop','.mp4'),('text','.txt')]:
   q=ROOT/'data/val'/folder/'speaker'/(row['relative']+ext);assert sha(q) in m['media_hashes']
  dev[s['clip_id']]=dict(clip_id=cid,index=i,frame_pts=pts,cache_path=str(path),cache=m);strata['non_null' if es else 'null'].append(i);crops=[];timed=False
  for start in range(0,len(pts),750):
   n=min(750,len(pts)-start);events=crop_events(es,pts,start,start+n,trajectory_frames=len(pts)) or [];timed|=any(e.interval is not None for e in events)
   crops.append(dict(start=start,length=n,events=[dict(event_id=e.event_id,text=e.text,interval=e.interval,frame_interval=e.frame_interval) for e in events]))
  if timed:strata['timed'].append(i)
  crop_manifest[s['clip_id']]=crops
 assert len(dev)==80 and len(strata['non_null'])==48 and len(strata['null'])==32
 write(OUT/'dev_sources.json',dev);write(OUT/'strata.json',strata);write(OUT/'dev_crops.json',crop_manifest)
 # Pre-generate content and time interventions without reading targets or predictions.
 text_map={};time_map={};donor_records=[]
 for receiver,crops in crop_manifest.items():
  session=receiver.split('/')[1];donors=[(name,c) for name,cs in crop_manifest.items() if name!=receiver and name.split('/')[1]==session for c in cs]
  for c in crops:
   key=f"{receiver}|{c['start']}";text_map[key]={}
   for e in c['events']:
    options=sorted({(name,d['event_id'],d['text']) for name,dc in donors for d in dc['events'] if d['text'].strip() and d['text']!=e['text']})
    if options:
     donor=options[int(text_key(key+'|'+e['event_id']),16)%len(options)];text_map[key][e['event_id']]=donor[2];donor_records.append(dict(receiver=receiver,crop=c['start'],event_id=e['event_id'],donor_source=donor[0],donor_event=donor[1]))
   timed=[i for i,e in enumerate(c['events']) if e['interval'] is not None];time_map[key]={}
   if len({tuple(c['events'][i]['interval']) for i in timed})>=2:
    # Fixed nonzero cyclic permutation, never moves text/order.
    for j,i in enumerate(timed):time_map[key][c['events'][i]['event_id']]=c['events'][timed[(j+1)%len(timed)]]['interval']
 write(OUT/'interventions.json',dict(text=text_map,time=time_map,donors=donor_records))
 write(OUT/'legal_texts.json',dict(train=sorted(train_texts),val=sorted(dev_texts)))
 for q in [OLD/'PROTOCOL.json',OLD/'cohort.json',OLD/'event_typer.json',OLD/'DEV_READY.json',OLD/'dev_weak/cache/index.json',EVAL/'multitarget_development_manifest.json',EVAL/'processed_targets/completed.json',OUT/'cohort.json',OUT/'dev_sources.json',OUT/'strata.json',OUT/'dev_crops.json',OUT/'interventions.json',OUT/'legal_texts.json']:frozen[str(q)]=sha(q)
 write(OUT/'PROTOCOL.json',dict(parent_sha256=PARENT_SHA,arms=ARMS,seed=123,steps=1000,checkpoints=[0,100,500,1000],B=4,T=750,K_train=4,K_eval=10,velocity_lr=2e-5,semantic_lr=1e-4,warmup_steps=100,weight_decay=.01,clip_norm=1.,dropout=.1,schedule_sha256=sha(OUT/'schedule.json'),dropout_sha256=sha(OUT/'dropout.json'),frozen_inputs=frozen,event_types=False,scope='48 TRAIN sources; full DEV80 primary; frozen auto-weak gates, no listener-derived semantic input',precision='FP32 train; FP64 Euler16 export -> FP32 official metrics',time_prior=dict(post_slack_s=2.,sigma_s=1.,minimum_bias=-8.),screening_tolerances=dict(FRC_min_ratio=1.,FRD_max_ratio=1.02,SMSE_min_ratio=.95),MAM_reference=dict(FRC=.810962319,FRD20=172.576423473,SMSE=.157203704),no_automatic_extension=True))
 print('prepared 48 TRAIN / 80 DEV / 1000 matched records',len(train_texts),len(dev_texts),flush=True)
if __name__=='__main__':prepare()
