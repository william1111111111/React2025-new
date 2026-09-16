"""Same-fold weak evidence, non-circular shifts, explicit UNKNOWN."""
import collections,random
import numpy as np
from .common import *
from .data import load

def main():
 manifest=read(OUT/'SPLIT_MANIFEST.json');rows=manifest['records'];normal=read(OUT/'NORMALIZATION.json');threshold=normal['activity_threshold'];rng=random.Random(123);all_windows=[];all_pairs=[]
 for split,budget in [('RM_fit',256),('RM_cal',64),('RM_audit',96)]:
  pool=[r for r in rows if r['split']==split and all(r['quality'][role]['exact_frame_count'] for role in ['speaker','listener'])];bins=collections.defaultdict(list)
  for r in pool:bins[r['session']].append(r)
  for rs in bins.values():rs.sort(key=lambda r:(r['quality']['listener']['mean_activity'],r['id']));rng.shuffle(rs)
  chosen=[]
  while len(chosen)<budget and any(bins.values()):
   for s in sorted(bins):
    if bins[s] and len(chosen)<budget:chosen.append(bins[s].pop())
  windows=[]
  for j,r in enumerate(chosen):
   N=min(len(r['pts']['speaker']),len(r['pts']['listener']));starts=[s for s in range(0,N,750) if N-s>=60]
   if not starts:continue
   start=rng.choice(starts);n=min(750,N-start);windows.append(dict(id=f'{split}_{j:04d}',source=r['id'],group=r['group'],session=r['session'],split=split,start=start,n=n,targets=[dict(id=r['id'],start=start)],window_policy='seeded native block; paired common extent'))
  byid={r['id']:r for r in pool};paired=[]
  # Reciprocal cross pairs balance each Y in positive/negative positions exactly.
  unused=list(range(len(windows)))
  while len(unused)>1:
   i=unused.pop(0);a=windows[i];ra=byid[a['source']];eligible=[j for j in unused if windows[j]['session']!=a['session'] and windows[j]['group']!=a['group']]
   if not eligible:continue
   def distance(j):
    b=windows[j];rb=byid[b['source']];qa,qb=ra['quality']['listener'],rb['quality']['listener'];return abs(np.log((qa['mean_activity']+1e-5)/(qb['mean_activity']+1e-5)))+abs(qa['mean_intensity']-qb['mean_intensity'])+abs(a['n']-b['n'])/750
   j=min(eligible,key=distance);unused.remove(j);b=windows[j];rb=byid[b['source']];n=min(a['n'],b['n']);active=min(ra['quality']['listener']['mean_activity'],rb['quality']['listener']['mean_activity'])>threshold;matched=distance(j)<1.0
   for x,y in [(a,b),(b,a)]:paired.append(dict(family='base_context',source=x['source'],start=x['start'],n=n,A=dict(id=x['source'],start=x['start']),B=dict(id=y['source'],start=y['start']),split=split,group=x['group'],donor_group=y['group'],weight=.5 if active and matched else 0.,state='PREFERRED' if active and matched else 'UNKNOWN',reason='activity/intensity matched cross-session statistical proxy' if active and matched else 'low activity or insufficient match',window=x['id']))
  all_pairs.extend(paired)
  for w in windows:
   r=byid[w['source']];start=w['start'];n=w['n'];same=[q for q in pool if q['session']==r['session'] and q['group']!=r['group'] and len(q['pts']['listener'])>=n];rng.shuffle(same)
   for donor in same[:3]:w['targets'].append(dict(id=donor['id'],start=0))
   base=next((z for z in paired if z['window']==w['id']),None)
   for target in w['targets'][1:]:
    active=byid[target['id']]['quality']['listener']['mean_activity']>threshold;ok=bool(base and base['weight']>0 and active)
    all_pairs.append(dict(family='weak_context',source=w['source'],start=start,n=min(n,base['n']) if base else n,A=target,B=base['B'] if base else dict(id=w['source'],start=start),split=split,group=w['group'],weight=.25 if ok else 0.,state='PREFERRED' if ok else 'UNKNOWN',reason='same-session weak global support versus matched cross-scene; no temporal label',window=w['id']))
   pts=np.array(r['pts']['listener']);yp=load(r['files']['facial-attributes/listener']['path'])
   for shift in [-4.,-2.,2.,4.]:
    # Joint valid support on actual native PTS; no circular wrap or synthetic seams.
    srcidx=np.arange(start,start+n);mapped=np.searchsorted(pts,pts[srcidx]+shift);valid=(mapped<len(pts))&(pts[srcidx]+shift>=pts[0]);ii=srcidx[valid];jj=mapped[valid]
    consecutive=len(ii)>=60 and np.all(np.diff(ii)==1) and np.all(np.diff(jj)==1)
    if consecutive:
     s=int(ii[0]);v=int(jj[0]);N=len(ii);a=yp[s:s+N];b=yp[v:v+N];dt=np.diff(pts[s:s+N]);activity=float(np.abs(np.diff(a,axis=0)/dt[:,None]).mean());neq=float(np.mean((a-b)**2));active=activity>threshold and neq>1e-4 and r['quality']['speaker']['mean_activity']>threshold and abs(shift)>2 and abs(r['pts']['speaker'][s]-pts[s])<.04
    else:s=start;v=start;N=n;active=False
    all_pairs.append(dict(family='temporal',source=w['source'],start=s,n=N,A=dict(id=w['source'],start=s),B=dict(id=w['source'],start=v),split=split,group=w['group'],weight=.5 if active else 0.,state='PREFERRED' if active else 'UNKNOWN',reason='weak active non-equivalent temporal proxy; delay ambiguity not independently resolved' if active else 'plausible delay/static/insufficient/nonconsecutive or unresolved PTS support',shift_seconds=shift,window=w['id']))
  all_windows.extend(windows)
 # Every donor, including UNKNOWN, must stay inside the originating fold.
 lookup={r['id']:r for r in rows}
 for p in all_pairs:
  assert all(lookup[z]['split']==p['split'] for z in [p['source'],p['A']['id'],p['B']['id']]);assert p['state']!='UNKNOWN' or p['weight']==0
 write(OUT/'WINDOWS.json',all_windows);write(OUT/'REAL_EVIDENCE.json',all_pairs)
 write(OUT/'LABEL_PROTOCOL.json',dict(seed=123,threshold_fit_only=threshold,scope='automatic weak statistical proxies, not human preference',same_session_temporal=False,circular_shift=False,unknown_gradient_weight=0,temporal_delay_limitation='No independent event labels. Shift labels remain weak proxies; report each shift and UNKNOWN separately.',counts={s:dict(collections.Counter(p['family']+'_'+p['state'] for p in all_pairs if p['split']==s)) for s in ['RM_fit','RM_cal','RM_audit']}))
 print('EVIDENCE',len(all_windows),'windows',len(all_pairs),'comparisons',flush=True)
if __name__=='__main__':main()
