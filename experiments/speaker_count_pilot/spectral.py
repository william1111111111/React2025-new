"""Separate exploratory eigengap analysis; never overwrites conservative v1."""
import json,csv,hashlib
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
ROOT=Path(__file__).resolve().parents[2];SRC=ROOT/'runs/reaction_flow/speaker_count_pilot_v1';OUT=ROOT/'runs/reaction_flow/speaker_count_spectral_v1'
POLICY=dict(method='cosine affinity, row top-neighbor pruning, symmetric unnormalized Laplacian, largest eigengap among k=1 and k=2 INCLUDING first eigenvalue',neighbor_fractions=[.2,.3,.4],primary_fraction=.3,seed=123,max_speakers=2,stable_rule='all three counts equal; if two, partition ARI >= .8; minimum two windows and 2.4s per cluster',status='exploratory after distance-gate v1 failed on 33/33; not calibrated or accepted weak labels',role='unknown; anonymous AU and 52-D motion are auxiliary, not identity or speaker-count truth')
def spectral(e,p):
 n=len(e)
 if n<4:return dict(count=None,labels=[None]*n,eigenvalues=[],gaps=[])
 a=np.maximum(e@e.T,0);np.fill_diagonal(a,0);keep=max(2,int(np.ceil(p*(n-1))));b=np.zeros_like(a)
 for i in range(n):
  ix=np.argsort(a[i])[-keep:];b[i,ix]=a[i,ix]
 a=(b+b.T)/2;lap=np.diag(a.sum(1))-a;vals,vecs=eigh(lap);gaps=np.diff(vals[:3]);k=int(np.argmax(gaps)+1)
 labels=np.zeros(n,int) if k==1 else KMeans(n_clusters=2,n_init=20,random_state=123).fit_predict(vecs[:,:2])
 if labels[0]==1:labels=1-labels
 return dict(count=k,labels=labels.tolist(),eigenvalues=vals[:4].tolist(),gaps=gaps.tolist())
def main():
 OUT.mkdir(exist_ok=False);(OUT/'POLICY.json').write_text(json.dumps(POLICY,indent=2));rows=[]
 for rr in json.loads((SRC/'SUMMARY.json').read_text())['records']:
  cid=rr['id'];raw=json.loads((SRC/(cid+'.json')).read_text());e=np.load(SRC/(cid+'_embeddings.npy'));runs=[spectral(e,p) for p in POLICY['neighbor_fractions']];main=runs[1];k=main['count'];labels=main['labels'];duration=np.array([w['end_s']-w['start_s'] for w in raw['windows']]);reasons=[]
  if k is None or sum(duration)<6:reasons.append('insufficient_evidence')
  if len(set(r['count'] for r in runs))!=1:reasons.append('neighbor_sensitivity')
  aris=[float(adjusted_rand_score(labels,r['labels'])) for r in runs] if k else []
  if aris and min(aris)<.8:reasons.append('partition_sensitivity')
  if k==2:
   for c in range(2):
    mask=np.asarray(labels)==c
    if mask.sum()<2 or duration[mask].sum()<2.4:reasons.append('small_cluster')
  if raw['short_unassigned_segments']:reasons.append('brief_voice_not_resolved')
  # Brief ignored segments are explicitly uncertainty, not silently assigned.
  stable=not any(r!='brief_voice_not_resolved' for r in reasons)
  wins=[dict(start_s=w['start_s'],end_s=w['end_s'],candidate_voice=None if l is None else 'voice_'+str(l)) for w,l in zip(raw['windows'],labels)]
  assigned=[]
  for word in raw['words']:
   a,b=word['start_s'],word['end_s'];ov={};label=None
   if a is not None and b is not None and b>a:
    for w in wins:
     if w['candidate_voice'] is not None:ov[w['candidate_voice']]=ov.get(w['candidate_voice'],0)+max(0,min(b,w['end_s'])-max(a,w['start_s']))
    if ov:
     z=max(ov,key=ov.get)
     if ov[z]/(b-a)>=.8:label=z
   assigned.append(dict(word=word['word'],start_s=a,end_s=b,candidate_voice=label))
  result=dict(id=cid,relative=raw['relative'],candidate_speaker_count=k,graph_stable=stable,uncertainty_reasons=sorted(set(reasons)),source_identity='unknown',human_reviewed=False,training_eligible=False,spectral_runs=runs,partition_ARI=aris,windows=wins,words=assigned,short_unassigned_segments=raw['short_unassigned_segments'],source_result_sha256=hashlib.sha256((SRC/(cid+'.json')).read_bytes()).hexdigest())
  (OUT/(cid+'.json')).write_text(json.dumps(result,indent=2));rows.append({key:result[key] for key in ['id','relative','candidate_speaker_count','graph_stable','uncertainty_reasons']})
 summary=dict(total=len(rows),candidate_counts={str(k):sum(r['candidate_speaker_count']==k for r in rows) for k in [1,2,None]},stable_counts={str(k):sum(r['candidate_speaker_count']==k and r['graph_stable'] for r in rows) for k in [1,2]},records=rows,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
 with open(OUT/'counts.csv','w') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 print({k:v for k,v in summary.items() if k!='records'})
if __name__=='__main__':main()
