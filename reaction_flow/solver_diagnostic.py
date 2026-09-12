"""Prespecified Euler16/32/64 on four sources; diagnostic only, no solver selection."""
import json,time
from pathlib import Path
import numpy as np
import torch
from .sampler import load_checkpoint
from .export import recording_noise
from .dynamics import GROUPS
from .train import configure_flow
from hirp.phase24 import normalization,verify_files
from hirp.phase15_audit import sha256_file
from mam_target.losses import ccc25
OUT=Path('runs/reaction_flow/task_dynamics_v1')
CP=Path('runs/reaction_flow/trajectory_v1/A/attempt_000/checkpoints/step_012000.pt')
@torch.no_grad()
def export(model,streams,length,z,steps):
    outputs=[]
    for start in range(0,length,750):
        n=min(750,length-start);inputs=[]
        for key in ('speaker_audio','speaker_emotion','speaker_3dmm'):
            v=streams[key];x=v.new_zeros(1,750,v.shape[-1]);x[0,:n]=v[start:start+n];inputs.append(x)
        noise=z.new_zeros(1,10,750,24);noise[0,:,:n]=z[:,start:start+n]
        p=model.sample(*inputs,torch.tensor([n],device=z.device),10,noise,integration_steps=steps,position_offset=torch.tensor([start],device=z.device))
        outputs.append(p[0,:,:n].cpu())
    return torch.cat(outputs,1)
def main():
    configure_flow();model,meta=load_checkpoint(CP,'cuda:0');model.double()
    manifest=json.loads(Path('runs/phase24/evaluation_v1/multitarget_development_manifest.json').read_text());normalization(manifest)
    targets=json.loads(Path('runs/phase24/evaluation_v1/processed_targets/completed.json').read_text())['result']['files']
    mean=torch.from_numpy(np.load('external/FaceVerse/mean_face.npy')).float();std=torch.from_numpy(np.load('external/FaceVerse/std_face.npy')).float().clamp_min(1e-8)
    rows=[]
    for i in (0,20,40,60):
        s=manifest['sources'][i];verify_files(list(s['files'].values())+[targets[i]])
        streams={k:torch.from_numpy(np.load(v['path'])).float() for k,v in s['files'].items()};face=streams['speaker_3dmm'];streams['speaker_3dmm']=((face[:,0] if face.ndim==3 else face)-mean)/std
        streams={k:v.cuda().double() for k,v in streams.items()};z=recording_noise(model.config,s['clip_id'],s['length']).cuda().double();target=torch.from_numpy(np.load(targets[i]['path'])).float();previous=None
        for steps in (16,32,64):
            start=time.time();p=export(model,streams,s['length'],z,steps);seconds=time.time()-start
            row=dict(source=i,NFE=steps,seconds=seconds,FRC=float(ccc25(p.float()[:,None],target[None]).max(1).values.sum()),previous_NFE=None if previous is None else steps//2,max_delta=None if previous is None else float((p-previous).abs().max()),mean_delta=None if previous is None else float((p-previous).abs().mean()),groups={g:dict(speed=float(p[...,a:b].diff(dim=1).abs().mean()),acceleration=float(p[...,a:b].diff(dim=1).diff(dim=1).abs().mean())) for g,a,b in GROUPS})
            rows.append(row);previous=p
            with (OUT/f'solver_source{i:03d}_NFE{steps}.json').open('x') as f:json.dump(row,f,indent=2)
            print(i,steps,seconds,row['FRC'],flush=True)
    with (OUT/'solver_diagnostic.json').open('x') as f:json.dump(dict(checkpoint_sha256=sha256_file(CP),precision='FP64 for all solver comparisons',rows=rows,production='unchanged Euler16; no DEV-based solver selection'),f,indent=2)
if __name__=='__main__':main()
