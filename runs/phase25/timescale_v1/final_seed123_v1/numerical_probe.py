import json,sys,time
from pathlib import Path
sys.path.insert(0,'/home/zhengshiyi/react2025_new')
import torch,numpy as np
from hirp.phase22 import load_checkpoint
from hirp.phase23_long import export_full
from hirp.train_phase21 import configure
from hirp.phase15_audit import sha256_file
configure();r=Path('runs/phase25/timescale_v1');out=r/'final_seed123_v1';manifest=json.load(open('runs/phase24/evaluation_v1/multitarget_development_manifest.json'))
mean=torch.from_numpy(np.load('external/FaceVerse/mean_face.npy')).float();std=torch.from_numpy(np.load('external/FaceVerse/std_face.npy')).float().clamp_min(1e-8)
noise=torch.from_numpy(np.load(manifest['noise']['path']))[0,:10].to('cuda:0');rows=[]
for arm in ('C0','C1','C2'):
 for step in (4000,6000):
  cp=sorted((r/'seed_123'/arm).glob(f'attempt_*/checkpoints/step_{step:06d}.pt'))[0]
  model,_=load_checkpoint(cp,'cuda:0')
  for idx in (0,20,40,60):
   s=manifest['sources'][idx];streams={k:torch.from_numpy(np.load(v['path'])).float() for k,v in s['files'].items()};face=streams['speaker_3dmm'];streams['speaker_3dmm']=((face[:,0] if face.ndim==3 else face)-mean)/std
   streams={k:v.to('cuda:0') for k,v in streams.items()}
   model.float();a=export_full(model,**streams,source_length=s['length'],noise=noise);b=export_full(model,**streams,source_length=s['length'],noise=noise,candidate_chunk=3)
   model.double();streams={k:v.double() for k,v in streams.items()};c=export_full(model,**streams,source_length=s['length'],noise=noise.double());d=export_full(model,**streams,source_length=s['length'],noise=noise.double(),candidate_chunk=3)
   row=dict(arm=arm,step=step,source_index=idx,checkpoint_sha256=sha256_file(cp),fp32_chunk_max=float((a-b).abs().max()),fp32_chunk_mean=float((a-b).abs().mean()),fp64_chunk_max=float((c-d).abs().max()),fp32_to_fp64_max=float((a.double()-c).abs().max()),fp32_other_chunk_to_fp64_max=float((b.double()-c).abs().max()))
   rows.append(row);print(row,flush=True)
  del model;torch.cuda.empty_cache()
with (out/'numerical_probe.json').open('x') as f:json.dump(dict(cases=rows,torch=torch.__version__,TF32=False,notes='FP64 diagnostic only; model weights and production FP32 predictions unchanged; same frozen sources and noise'),f,indent=2)
