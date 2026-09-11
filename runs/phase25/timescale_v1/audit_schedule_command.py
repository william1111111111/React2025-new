import json,hashlib,random
from pathlib import Path
import numpy as np
from hirp.session_data import SessionPopulation
from hirp.train_phase22 import write
root=Path('runs/phase25/timescale_v1');pool=SessionPopulation('data','train',750);lengths={}
for i,path in enumerate(pool.dataset.records):
 rel=path.relative_to(pool.dataset.directory/'facial-attributes')
 lens=[len(np.load(p,mmap_mode='r')) for p in (path,pool.dataset.directory/'audio-features'/rel,pool.dataset.directory/'coefficients'/rel)]
 target=pool.dataset.directory/'facial-attributes'/'listener'/rel.parts[1]/path.name
 lengths[i]=(min(lens),len(np.load(target,mmap_mode='r')),rel.as_posix())
results=[]
for seed in (123,42,2026):
 records=json.load(open(root/f'seed_{seed}/schedule.json'))['records'];counts=dict(source_frame_exposures=0,paired_frame_exposures=0,zero_pair_count=0,source_occurrences=0,valid_length_gt128=0)
 for r in records:
  for i,occ in zip(r['source_indices'],r['crop_occurrences']):
   n,y,rel=lengths[i];payload=f'{seed}|{occ}|{rel}'.encode();g=random.Random(int.from_bytes(hashlib.sha256(payload).digest()[:8],'big'));start=g.randint(0,max(0,n-750));l=min(750,n-start);p=min(l,max(0,y-start))
   counts['source_frame_exposures']+=l;counts['paired_frame_exposures']+=p;counts['zero_pair_count']+=p==0;counts['source_occurrences']+=1;counts['valid_length_gt128']+=l>128
 results.append(dict(seed=seed,**counts))
write(root/'schedule_coverage_audit.json',dict(source_population=len(lengths),seeds=results,policy='all6000steps audited without loading target values; zero pairs fail without resampling'))
print(results)
