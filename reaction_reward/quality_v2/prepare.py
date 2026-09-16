"""Fit-only expanded real evidence and shared A/B quality targets; no model scores used for source selection."""
import collections,random,math
import numpy as np
from reaction_reward.common import ROOT,OUT as OLD,read,write,sha,seed
from reaction_reward.review_v2.temporal import nearest
from reaction_reward.review_v2.sampling import EffectiveSampler
OUT=ROOT/'runs/reaction_reward/quality_v2';REVIEW=ROOT/'runs/reaction_reward/review_v2'
def main():
 if (OUT/'PREPARE_COMPLETE.json').exists():return
 manifest=read(OLD/'SPLIT_MANIFEST.json');allrows={r['id']:r for r in manifest['records']};rows={k:r for k,r in allrows.items() if r['split']=='RM_fit' and all(r['quality'][role]['exact_frame_count'] for role in ['speaker','listener'])};policy=read(REVIEW/'PTS_POLICY.json');threshold=policy['activity_threshold'];rng=random.Random(123);windows=[]
 for rid,r in sorted(rows.items()):
  N=min(len(r['pts']['speaker']),len(r['pts']['listener']));starts=[s for s in range(0,N,750) if N-s>=60];starts.sort(key=lambda s:seed(f'expand|123|{rid}|{s}'))
  for s in starts[:2]:windows.append(dict(id='expanded_'+str(len(windows)),source=rid,group=r['group'],session=r['session'],start=s,n=min(750,N-s),split='RM_fit'))
 pairs=[];unused=list(range(len(windows)))
 while len(unused)>1:
  i=unused.pop(0);a=windows[i];ra=rows[a['source']];cand=[j for j in unused if windows[j]['session']!=a['session'] and windows[j]['group']!=a['group']]
  if not cand:continue
  def distance(j):
   b=windows[j];qa,qb=ra['quality']['listener'],rows[b['source']]['quality']['listener'];return abs(np.log((qa['mean_activity']+1e-5)/(qb['mean_activity']+1e-5)))+abs(qa['mean_intensity']-qb['mean_intensity'])+abs(a['n']-b['n'])/750
  j=min(cand,key=distance);unused.remove(j);b=windows[j];active=min(ra['quality']['listener']['mean_activity'],rows[b['source']]['quality']['listener']['mean_activity'])>threshold and distance(j)<1
  for x,y in [(a,b),(b,a)]:pairs.append(dict(family='base_context',source=x['source'],start=x['start'],n=min(x['n'],y['n']),A=dict(id=x['source'],start=x['start']),B=dict(id=y['source'],start=y['start']),split='RM_fit',group=x['group'],weight=.5 if active else 0.,state='PREFERRED' if active else 'UNKNOWN',window=x['id'],reason='reciprocal activity/intensity matched cross-session weak proxy'))
 base={r['window']:r for r in pairs};cache={}
 for w in windows:
  r=rows[w['source']];rid=r['id'];s=w['start'];n=w['n'];donors=[q for q in rows.values() if q['session']==r['session'] and q['group']!=r['group'] and len(q['pts']['listener'])>=n];rng.shuffle(donors)
  for d in donors[:3]:
   b=base.get(w['id']);ok=bool(b and b['weight']>0 and d['quality']['listener']['mean_activity']>threshold);pairs.append(dict(family='weak_context',source=rid,start=s,n=min(n,b['n']) if b else n,A=dict(id=d['id'],start=0),B=b['B'] if b else dict(id=rid,start=s),split='RM_fit',group=w['group'],weight=.25 if ok else 0.,state='PREFERRED' if ok else 'UNKNOWN',window=w['id'],reason='same-session global weak support; temporal head masked'))
  pts=np.asarray(r['pts']['listener']);yp=np.load(r['files']['facial-attributes/listener']['path']);assert sha(r['files']['facial-attributes/listener']['path'])==r['files']['facial-attributes/listener']['sha256']
  for shift in [-4.,-2.,2.,4.]:
   ii,jj,d=nearest(pts,np.arange(s,s+n),shift,policy['nominal_interval'],policy['precision_s']);checks=d['checks'];geom=all(checks.values());start=int(ii[0]) if geom else s;v=int(jj[0]) if geom else s;N=len(ii) if geom else n;active=False
   if geom:active=float(np.abs(np.diff(yp[ii],axis=0)/np.diff(pts[ii])[:,None]).mean())>threshold and float(np.mean((yp[ii]-yp[jj])**2))>1e-4
   checks.update(listener_active=bool(active),speaker_active=r['quality']['speaker']['mean_activity']>threshold,delay_policy=abs(shift)>2,sync=bool(geom and abs(r['pts']['speaker'][start]-pts[start])<.04));ok=all(checks.values());pairs.append(dict(family='temporal',source=rid,start=start,n=N,A=dict(id=rid,start=start),B=dict(id=rid,start=v),split='RM_fit',group=w['group'],weight=.5 if ok else 0.,state='PREFERRED' if ok else 'UNKNOWN',window=w['id'],shift_seconds=shift,rules=checks,max_abs_residual_s=d['max_abs_residual_s'],supported_frames=d['supported_frames']))
 # Existing cal/development evidence is fixed; never used to choose fit rows.
 pairs += [r for r in read(REVIEW/'REAL_EVIDENCE.json') if r['split']!='RM_fit'];pairs += read(OLD/'GENERATED_EVIDENCE.json')
 for r in pairs:
  assert all(allrows[z]['split']==r['split'] for z in [r['source'],r['A']['id'],r['B']['id']]);assert r['state']!='UNKNOWN' or r['weight']==0
 write(OUT/'RECORDS.json',pairs);write(OUT/'EXPANDED_WINDOWS.json',windows)
 sampler=EffectiveSampler(pairs);families={'base_context':16,'temporal':8,'weak_context':4,'generated':4};schedule=[{f:sampler.draw(f,n) for f,n in families.items()} for _ in range(4000)];write(OUT/'SCHEDULE.json',schedule)
 genwindows=[w for w in read(OLD/'WINDOWS.json') if w['split']=='RM_fit'];quality=[]
 for w in genwindows:
  for gen in ['P2','N0']:
   m=read(OLD/'bank'/w['id']/(gen+'_scores.json'));C=np.asarray(m['C']).max(1);D=np.asarray(m['D']).min(1)
   for k,(c,d) in enumerate(zip(C,D)):quality.append(dict(source=w['source'],window=w['id'],group=w['group'],split='RM_fit',start=w['start'],n=w['n'],A=dict(id=w['source'],start=w['start'],generated=str(OLD/'bank'/w['id']/(gen+'.npy')),candidate=k),generator=gen,C=float(c),D=float(d),uD=float(np.log1p(d/np.sqrt(w['n'])))))
 groups=collections.defaultdict(set)
 for r in quality:groups[r['group']].add(r['source'])
 freq=collections.Counter(r['source'] for r in quality);weights=np.array([1/len(groups)/len(groups[r['group']])/freq[r['source']] for r in quality]);values=np.array([[r['C'],r['uD']] for r in quality]);mean=(values*weights[:,None]).sum(0);std=np.sqrt(((values-mean)**2*weights[:,None]).sum(0)).clip(1e-6)
 for r,v in zip(quality,values):r['target']=((v-mean)/std).tolist()
 write(OUT/'QUALITY_RECORDS.json',quality);write(OUT/'QUALITY_SCALE.json',dict(fit_split='RM_fit',mean=mean.tolist(),std=std.tolist(),transform='C; log1p(native_DTW/sqrt(valid_frames))',weighting='equal date, equal source within date, equal candidate within source'))
 # Group/source-balanced four windows, cyclic generator/quality-quantile candidate exposure.
 bysource=collections.defaultdict(list)
 for i,r in enumerate(quality):bysource[r['source']].append(i)
 for src,ids in bysource.items():
  ordered=[]
  for gen in ['P2','N0']:
   idsgen=sorted([i for i in ids if quality[i]['generator']==gen],key=lambda i:quality[i]['target'][0]-quality[i]['target'][1]);quartiles=[idsgen[j::4] for j in range(4)]
   for z in zip(*quartiles):ordered.extend(z)
  bysource[src]=ordered
 gs=collections.Counter();ss=collections.Counter();quality_schedule=[]
 for step in range(4000):
  chosen=[];used=set()
  for j in range(4):
   sources=[s for s in bysource if s not in used];rng.shuffle(sources);src=min(sources,key=lambda s:(gs[allrows[s]['group']],ss[s]));ids=bysource[src];pos=ss[src]*2;chosen.extend([ids[pos%len(ids)],ids[(pos+1)%len(ids)]]);used.add(src);ss[src]+=1;gs[allrows[src]['group']]+=1
  quality_schedule.append(chosen)
 write(OUT/'QUALITY_SCHEDULE.json',quality_schedule)
 counts={f:dict(valid=sum(r['family']==f and r['split']=='RM_fit' and r['weight']>0 for r in pairs),sources=len({r['source'] for r in pairs if r['family']==f and r['split']=='RM_fit' and r['weight']>0})) for f in families}
 write(OUT/'PREPARE_COMPLETE.json',dict(complete=True,fit_recordings=len(rows),expanded_windows=len(windows),counts=counts,generated_quality_candidates=len(quality),normalization_sha256=sha(OLD/'NORMALIZATION.json'),split_sha256=sha(OLD/'SPLIT_MANIFEST.json'),PTS_policy_sha256=sha(REVIEW/'PTS_POLICY.json')));print(counts,flush=True)
if __name__=='__main__':main()
