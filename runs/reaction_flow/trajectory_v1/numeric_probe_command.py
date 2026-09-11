import json,time
import numpy as np
import torch
from reaction_flow.train import configure_flow
from reaction_flow.sampler import load_checkpoint
from reaction_flow.export import recording_noise,export_full
from reaction_flow.config import ROOT
from hirp.phase24 import normalization
configure_flow();manifest=json.load(open('runs/phase24/evaluation_v1/multitarget_development_manifest.json'));normalization(manifest)
model,_=load_checkpoint(ROOT/'A/attempt_000/checkpoints/step_002000.pt','cuda:0');mean=torch.from_numpy(np.load('external/FaceVerse/mean_face.npy')).float();std=torch.from_numpy(np.load('external/FaceVerse/std_face.npy')).float().clamp_min(1e-8)
s=manifest['sources'][1];inputs={k:torch.from_numpy(np.load(v['path'])).float() for k,v in s['files'].items()};face=inputs['speaker_3dmm'];inputs['speaker_3dmm']=((face[:,0] if face.ndim==3 else face)-mean)/std;inputs={k:v.cuda() for k,v in inputs.items()};noise=recording_noise(model.config,s['clip_id'],s['length']).cuda()
a=export_full(model,**inputs,source_length=s['length'],noise=noise);b=export_full(model,**inputs,source_length=s['length'],noise=noise,candidate_chunk=3)
model.double();double_inputs={k:v.double() for k,v in inputs.items()};d=export_full(model,**double_inputs,source_length=s['length'],noise=noise.double());e=export_full(model,**double_inputs,source_length=s['length'],noise=noise.double(),candidate_chunk=3)
x=dict(source=1,length=s['length'],fp32_chunk_max=float((a-b).abs().max()),fp64_chunk_max=float((d-e).abs().max()),fp32_fp64_max=float((a.double()-d).abs().max()),fp32_fp64_mean=float((a.double()-d).abs().mean()),scope='numerical diagnostic only, no target metrics or solver selection')
print(x);(ROOT/'numeric_probe_trained.json').write_text(json.dumps(x,indent=2))
