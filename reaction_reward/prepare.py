"""TRAIN inventory, conservative recording-day grouping, duplicate closure and PTS audit.
No learned weights, validation data or generated preference labels are read here.
"""
import subprocess,concurrent.futures,collections,time,random
import numpy as np
from .common import *

def inspect(r):
 name=r['clip_id'];base=name.split('/')[-1];session=name.split('/')[1];files={};timelines={};quality={}
 for role in ['speaker','listener']:
  for folder in ['audio-features','facial-attributes','coefficients','audio','video-face-crop']:
   ext={'audio':'.wav','video-face-crop':'.mp4'}.get(folder,'.npy');p=DATA/folder/role/session/(base+ext)
   if not p.exists():raise FileNotFoundError(p)
   files[folder+'/'+role]=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
  p=files['video-face-crop/'+role]['path']
  result=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','json',p],capture_output=True,text=True,check=True)
  pts=np.array([float(x['best_effort_timestamp_time']) for x in json.loads(result.stdout)['frames']]);assert len(pts)>1 and np.all(np.diff(pts)>0)
  sizes=[]
  for folder,width in [('audio-features',768),('facial-attributes',25),('coefficients',58)]:
   x=np.load(files[folder+'/'+role]['path'],mmap_mode='r');x=x[:,0] if x.ndim==3 else x
   assert x.ndim==2 and x.shape[1]==width
   sizes.append(len(x));assert np.isfinite(x).all()
  y=np.load(files['facial-attributes/'+role]['path'],mmap_mode='r');n=min(len(y),len(pts));dt=np.diff(pts[:n]);v=np.diff(y[:n],axis=0)/dt[:,None]
  quality[role]=dict(mean_activity=float(np.abs(v).mean()),mean_intensity=float(np.abs(y[:n]).mean()),feature_lengths=sizes,pts_frame_count=len(pts),exact_frame_count=all(k==len(pts) for k in sizes))
  timelines[role]=pts.tolist()
 return dict(id=name.removeprefix('speaker/'),session=session,date='-'.join(base.split('-')[1:4]),files=files,pts=timelines,quality=quality,paired_observation='official opposite-role identical-basename pairing; not independent sensor synchronization verification')

def main():
 OUT.mkdir(parents=True,exist_ok=True);cache=OUT/'inventory';cache.mkdir(exist_ok=True)
 rows=read(ROOT/'runs/reaction_flow/mode_supervision_v1/cohort.json')
 todo=[r for r in rows if not (cache/(str(r['index'])+'.json')).exists()]
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
  for r,x in zip(todo,ex.map(inspect,todo)):
   write(cache/(str(r['index'])+'.json'),x);count=len(list(cache.glob('*.json')));write(OUT/'prepare_status.json',dict(stage='hash_and_PTS_inventory',completed=count,total=len(rows),time=time.time()));print('inventory',count,'/',len(rows),flush=True)
 records=[read(cache/(str(r['index'])+'.json')) for r in rows]
 # Conservative superset of filename recording groups: all files from one date
 # stay together across every session and role. Never claim participant disjointness.
 parents=list(range(len(records)))
 def find(i):
  while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
  return i
 def union(i,j):parents[find(i)]=find(j)
 by_date={};by_hash={}
 for i,r in enumerate(records):
  date=r['date'];assert len(date)==10
  if date in by_date:union(i,by_date[date])
  else:by_date[date]=i
  for f in r['files'].values():
   h=f['sha256']
   if h in by_hash:union(i,by_hash[h])
   else:by_hash[h]=i
 groups=collections.defaultdict(list)
 for i in range(len(records)):groups[find(i)].append(i)
 order=list(groups.values());random.Random(123).shuffle(order)
 # Fixed seeded greedy assignment to nearest 70/15/15 occupancy, no label selection.
 counts=[0,0,0];targets=np.array([.7,.15,.15])*len(records);splits=['RM_fit','RM_cal','RM_audit'];assign={}
 for ids in order:
  k=int(np.argmax((targets-np.array(counts))/targets));counts[k]+=len(ids)
  gid=hashlib.sha256('|'.join(sorted(records[i]['id'] for i in ids)).encode()).hexdigest()[:16]
  for i in ids:records[i].update(split=splits[k],group=gid);assign[i]=splits[k]
 assert min(counts)>0
 seen={}
 for r in records:
  for f in r['files'].values():
   h=f['sha256'];assert h not in seen or seen[h]==r['split'];seen[h]=r['split']
 summary=dict(unit='conservative recording-date components plus exact raw-media/feature duplicate closure',scope='recording-date-disjoint; NOT person/dyad-disjoint',limitation='No independent participant or original-session ID map found. Date grouping is a conservative provenance assumption, not verified identity. All same-date recordings, both roles and all derived crops are bound.',seed=123,counts=dict(zip(splits,counts)),groups=len(groups),dates=len(by_date),ratios=[c/len(records) for c in counts],raw_hash_overlap_across_folds=0,records=records)
 write(OUT/'SPLIT_MANIFEST.json',summary)
 # Only RM_fit determines scales and activity thresholds.
 fit=[r for r in records if r['split']=='RM_fit'];stats={};activities=[]
 for field,role in [('audio-features','speaker'),('facial-attributes','speaker'),('coefficients','speaker'),('facial-attributes','listener')]:
  total=None;sq=None;count=0
  for r in fit:
   x=np.load(r['files'][field+'/'+role]['path']);x=x[:,0] if x.ndim==3 else x;x=x.astype(np.float64)
   v=x.sum(0);v2=(x*x).sum(0);total=v if total is None else total+v;sq=v2 if sq is None else sq+v2;count+=len(x)
  mean=total/count;std=np.sqrt(np.maximum(sq/count-mean*mean,1e-8));stats[field+'/'+role]=dict(mean=mean.tolist(),std=std.tolist(),frames=count)
 activities=[r['quality'][role]['mean_activity'] for r in fit for role in ['speaker','listener']]
 write(OUT/'NORMALIZATION.json',dict(fit_split='RM_fit',stats=stats,activity_threshold=float(np.quantile(activities,.25)),provenance_sha256=sha(OUT/'SPLIT_MANIFEST.json')))
 write(OUT/'prepare_complete.json',dict(complete=True,counts=summary['counts'],groups=len(groups),time=time.time(),limitations=summary['limitation']))
 print('PREPARE COMPLETE',summary['counts'],'groups',len(groups),flush=True)
if __name__=='__main__':main()
