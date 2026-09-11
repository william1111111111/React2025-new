import json,csv,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/zhengshiyi/react2025_new')
from hirp.session_data import SessionPopulation
r=Path('runs/phase25/timescale_v1');old=Path('runs/phase22/replicated_v1/seed_123');new=r/'seed_123'
pool=SessionPopulation('data','train',128);sizes={}
for i,p in enumerate(pool.dataset.records):
 rel=p.relative_to(pool.dataset.directory/'facial-attributes');sizes[i]=min(len(np.load(x,mmap_mode='r')) for x in (p,pool.dataset.directory/'audio-features'/rel,pool.dataset.directory/'coefficients'/rel))
a=json.load(open(old/'schedule.json'))['records'];b=json.load(open(new/'schedule.json'))['records'][:2000]
assert all(x['source_indices']==y['source_indices'] and x['reference_ids']==y['reference_ids'] for x,y in zip(a,b))
old_frames=sum(min(128,sizes[i]) for rec in a for i in rec['source_indices'])
rows=[]
for arm in ('C0','C1','C2'):
 oldtask=Path('runs/phase24/evaluation_v1/task_multitarget')/f'seed123_{arm}_lambda{0 if arm=="C0" else .1:g}.json'
 for label,base,task in [('T128_old',old,oldtask),('T750_new',new,r/'task_multitarget'/f'seed123_{arm}_step2000.json')]:
  s=json.load(open(base/arm/'attempt_000/summary.json'));v=json.load(open(task))['result']
  rows.append(dict(protocol=label,arm=arm,step=2000,source_occurrences=s['source_occurrences'],valid_source_frames=old_frames if label=='T128_old' else s['valid_source_frame_exposures'],tensor_frames=2000*4*(128 if label=='T128_old' else 750),training_wall_seconds=s['attempt_wall_seconds'],peak_bytes=s['peak_allocated_bytes'],full_K10_multitarget_FRC=v['multi_target']['FRC'],full_K10_single_FRC=v['single_paired']['FRC'] if 'single_paired' in v else v['single_paired_FRC']))
with (r/'final_seed123_v1/analysis/T128_T750_training_cost_bridge.csv').open('x') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(rows)
