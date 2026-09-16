import collections
import numpy as np
from reaction_reward.common import OUT as OLD,read,write,sha,ROOT
from reaction_reward.review_v2.temporal import nearest
OUT=ROOT/'runs/reaction_reward/review_v2'
def main():
 assert not (OUT/'REAL_EVIDENCE.json').exists(),'version already exists; do not overwrite'
 rows={r['id']:r for r in read(OLD/'SPLIT_MANIFEST.json')['records']};windows=read(OLD/'WINDOWS.json');old=read(OLD/'REAL_EVIDENCE.json');oldtime={(r['window'],r['shift_seconds']):r for r in old if r['family']=='temporal'};threshold=read(OLD/'NORMALIZATION.json')['activity_threshold']
 medians=[float(np.median(np.diff(r['pts']['listener']))) for r in rows.values() if r['split']=='RM_fit'];nominal=float(np.median(medians));policy=dict(fit_split='RM_fit',interval_statistic='median of recording median listener PTS intervals',nominal_interval=nominal,precision_s=1e-6,residual_limit='0.5*min(local adjacent interval, fit nominal interval)+1e-6',large_gap_limit='1.5*fit nominal interval + 1e-6',activity_threshold=threshold,delay_policy='abs(shift)<=2 remains UNKNOWN; +/-4 weak proxy only',frozen_before_application=True)
 write(OUT/'PTS_POLICY.json',policy);new=[r for r in old if r['family']!='temporal'];details=[]
 for w in windows:
  r=rows[w['source']];p=np.asarray(r['pts']['listener']);yp=np.load(r['files']['facial-attributes/listener']['path']);assert sha(r['files']['facial-attributes/listener']['path'])==r['files']['facial-attributes/listener']['sha256'];indices=np.arange(w['start'],w['start']+w['n'])
  for shift in [-4.,-2.,2.,4.]:
   ii,jj,d=nearest(p,indices,shift,nominal);checks=d['checks'].copy();geom=all(checks.values());s=int(ii[0]) if geom else w['start'];v=int(jj[0]) if geom else w['start'];n=len(ii) if geom else w['n'];activity=None;neq=None
   if geom:
    a,b=yp[ii],yp[jj];activity=float(np.abs(np.diff(a,axis=0)/np.diff(p[ii])[:,None]).mean());neq=float(np.mean((a-b)**2))
   checks.update(listener_active=bool(geom and activity>threshold),non_equivalent=bool(geom and neq>1e-4),speaker_active=r['quality']['speaker']['mean_activity']>threshold,delay_unambiguous_by_policy=abs(shift)>2,speaker_listener_origin_supported=bool(geom and abs(r['pts']['speaker'][s]-p[s])<.04))
   good=all(checks.values());prior=oldtime[w['id'],shift];x=dict(prior,start=s,n=n,A=dict(id=w['source'],start=s),B=dict(id=w['source'],start=v),state='PREFERRED' if good else 'UNKNOWN',weight=.5 if good else 0.,reason='weak active nearest-PTS temporal proxy' if good else ' / '.join(k for k,val in checks.items() if not val));new.append(x);details.append(dict(window=w['id'],source=w['source'],split=w['split'],group=w['group'],shift_seconds=shift,old_state=prior['state'],new_state=x['state'],activity=activity,mean_squared_change=neq,rules=checks,rejections=[k for k,val in checks.items() if not val],mapping=d))
 for r in new:
  assert all(rows[z]['split']==r['split'] for z in [r['source'],r['A']['id'],r['B']['id']]);assert r['state']!='UNKNOWN' or r['weight']==0
 write(OUT/'REAL_EVIDENCE.json',new);write(OUT/'TEMPORAL_MAPPING.json',details);stats={}
 for split in ['RM_fit','RM_cal','RM_audit']:
  ds=[r for r in details if r['split']==split];stats[split]=dict(total=len(ds),old_valid=sum(r['old_state']=='PREFERRED' for r in ds),new_valid=sum(r['new_state']=='PREFERRED' for r in ds),recovered=sum(r['old_state']=='UNKNOWN' and r['new_state']=='PREFERRED' for r in ds),lost=sum(r['old_state']=='PREFERRED' and r['new_state']=='UNKNOWN' for r in ds),valid_sources=len({r['source'] for r in ds if r['new_state']=='PREFERRED'}),valid_date_groups=len({r['group'] for r in ds if r['new_state']=='PREFERRED'}),rejection_counts=dict(collections.Counter(k for r in ds for k in r['rejections'])),by_shift={str(s):dict(old=sum(r['old_state']=='PREFERRED' for r in ds if r['shift_seconds']==s),new=sum(r['new_state']=='PREFERRED' for r in ds if r['shift_seconds']==s)) for s in [-4.,-2.,2.,4.]})
 write(OUT/'TEMPORAL_SUMMARY.json',stats);write(OUT/'MANIFEST.json',dict(old_real_evidence_sha256=sha(OLD/'REAL_EVIDENCE.json'),new_real_evidence_sha256=sha(OUT/'REAL_EVIDENCE.json'),split_sha256=sha(OLD/'SPLIT_MANIFEST.json'),windows_sha256=sha(OLD/'WINDOWS.json'),policy_sha256=sha(OUT/'PTS_POLICY.json'),audit_reuse='development diagnostic; no longer a fresh independent confirmation',synthetic_count_included=False));print(stats)
if __name__=='__main__':main()
