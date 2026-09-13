"""Final fixed noise-factor probes; no optimizer and no policy selection."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .prepare_shared import ROOT
from .shared_model import load_checkpoint,export_full
from .shared_noise import recording_bases
from .train import configure_flow,write
from .dynamics import GROUPS
from hirp.phase24 import normalization,verify_files

def main(arm):
    configure_flow();model,meta=load_checkpoint(ROOT/arm/'attempt_000/checkpoints/step_016000.pt','cuda:0');model.double()
    task=json.loads((ROOT/'task_multitarget'/f'seed123_{arm}_step16000.json').read_text())['result'];manifest=json.loads(Path('runs/phase24/evaluation_v1/multitarget_development_manifest.json').read_text());normalization(manifest)
    mean=torch.from_numpy(np.load('external/FaceVerse/mean_face.npy')).float();std=torch.from_numpy(np.load('external/FaceVerse/std_face.npy')).float().clamp_min(1e-8);rows=[]
    for i in (0,20,40,60):
        s=manifest['sources'][i];verify_files(list(s['files'].values())+[task['exports'][i]])
        base=torch.from_numpy(np.load(task['exports'][i]['path']));streams={k:torch.from_numpy(np.load(v['path'])).float() for k,v in s['files'].items()};face=streams['speaker_3dmm'];streams['speaker_3dmm']=((face[:,0] if face.ndim==3 else face)-mean)/std;streams={k:v.cuda().double() for k,v in streams.items()}
        local,global_noise=recording_bases(model.config,s['clip_id'],s['length']);other_local,_=recording_bases(model.config,s['clip_id']+'|fixed_local_probe',s['length']);_,other_global=recording_bases(model.config,s['clip_id']+'|fixed_global_probe',s['length'])
        for name,l,g in [('change_global',local,other_global),('change_local',other_local,global_noise)]:
            y=export_full(model,**streams,source_length=s['length'],noise=l.cuda().double(),global_noise=g.cuda().double()).float()
            diag={key:dict(mean_abs_change=float((y[...,a:b]-base[...,a:b]).abs().mean()),temporal_mean_abs_change=float((y[...,a:b].mean(1)-base[...,a:b].mean(1)).abs().mean())) for key,a,b in GROUPS}
            if model.rho==0 and name=='change_global':assert torch.equal(y,base)
            rows.append(dict(source=i,probe=name,groups=diag));print(arm,i,name,diag,flush=True)
    write(ROOT/f'{arm}_noise_factor_probe.json',dict(arm=arm,rho=model.rho,recipients=[0,20,40,60],rows=rows,scope='fixed source-only factor sensitivity; not semantic disentanglement'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);a=p.parse_args();main(a.arm)
