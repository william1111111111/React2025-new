import json,sys
from pathlib import Path
sys.path.insert(0,'/home/zhengshiyi/react2025_new')
import torch,numpy as np
from hirp.phase22 import load_checkpoint
from hirp.phase23_long import export_full
from hirp.train_phase21 import configure
configure();r=Path('runs/phase25/timescale_v1');m=json.load(open('runs/phase24/evaluation_v1/multitarget_development_manifest.json'));s=m['sources'][0]
cp=sorted((r/'seed_123/C0').glob('attempt_*/checkpoints/step_004000.pt'))[0];model,_=load_checkpoint(cp,'cuda:0')
mean=torch.from_numpy(np.load('external/FaceVerse/mean_face.npy')).float();std=torch.from_numpy(np.load('external/FaceVerse/std_face.npy')).float().clamp_min(1e-8)
streams={k:torch.from_numpy(np.load(v['path'])).float() for k,v in s['files'].items()};face=streams['speaker_3dmm'];streams['speaker_3dmm']=((face[:,0] if face.ndim==3 else face)-mean)/std
streams={k:v.to('cuda:0') for k,v in streams.items()};noise=torch.from_numpy(np.load(m['noise']['path']))[0,:10].to('cuda:0')
base=export_full(model,**streams,source_length=s['length'],noise=noise)
rows=[]
for chunk in (10,3,1):
 pred=export_full(model,**streams,source_length=s['length'],noise=noise,candidate_chunk=chunk);delta=(pred-base).abs()
 rows.append(dict(chunk=chunk,max_error=float(delta.max()),mean_error=float(delta.mean()),count_gt_1e5=int((delta>1e-5).sum()),total_elements=delta.numel(),finite=bool(torch.isfinite(pred).all())))
with (r/'chunk_failure_probe.json').open('x') as f:json.dump(dict(checkpoint=str(cp),source=s['clip_id'],rows=rows),f,indent=2)
print(rows)
