import csv,json
from pathlib import Path
import torch
from hirp.session_data import SessionPopulation
from hirp.data_phase25 import reference_crop
from hirp.phase15_audit import sha256_file
from mam_refine.audit import ROOT,csvwrite,write
p=SessionPopulation('data','train',750);N=sum(map(len,p.sources.values()));rows=[];files=[]
for session in sorted(p.sources):
 for ref in p.references[session]:
  y,n,start=reference_crop(p,ref,25000,0);y=y[:int(n)];weight=len(p.sources[session])/N/len(p.references[session]);files.append(dict(path=str(p.reference_paths[ref]),sha256=sha256_file(p.reference_paths[ref])))
  for group,a,b in [('AU',0,15),('VA',15,17),('expression',17,25)]:
   x=y[:,a:b].double();rows.append(dict(reference=ref,session=session,group=group,crop_start=start,length=int(n),weight=weight,temporal_variance=float(x.var(0).mean()) if len(x)>1 else 0.,speed_abs=float(x.diff(dim=0).abs().mean()) if len(x)>1 else 0.,minimum=float(x.min()),maximum=float(x.max())))
csvwrite(ROOT/'train_activity_per_reference.csv',rows)
summary={g:{k:sum(r['weight']*r[k] for r in rows if r['group']==g) for k in ('temporal_variance','speed_abs','minimum','maximum')} for g in ('AU','VA','expression')}
write(ROOT/'train_activity.json',dict(summary=summary,files=files,crop_seed=25000,crop_draw=0,weighting='p(session)=n_source/N; uniform unique TRAIN listener per session',note='TRAIN temporal activity only. Different-source trajectories do not estimate within-input conditional variance. Existing evaluation targets use Processor, unlike these raw TRAIN crops.'))
print(summary)
