"""Fixed TRAIN precision comparison, raw TRAIN activity and same-session response."""
import copy,json
from pathlib import Path
import numpy as np
import torch
from .config import FlowConfig
from .data import resources,draw
from .train import configure_flow
from .sampler import load_checkpoint
from .finetune import task_loss
from .solver_diagnostic import export,CP,OUT
from .export import recording_noise
from .dynamics import GROUPS
from hirp.paired_data import paired_model_inputs
from mam_target.losses import ccc25

def main():
    configure_flow();model,_=load_checkpoint(CP,'cuda:0');cfg=FlowConfig();_,records,data=resources(cfg,'cuda:0');b,t,l,ids,y,n,slots=data.batch(records[12000]);h,sm=model.condition(**paired_model_inputs(b));z=draw((4,4,750,24),cfg.rollout_seed,12001,'task_rollout').cuda()
    with torch.no_grad():
        a=model.rollout(h,sm,z,16,b['crop_start']);qa=task_loss(a,b,t,l)
        double=copy.deepcopy(model).double();hh,ss=double.condition(**{k:(v.double() if v.is_floating_point() else v) for k,v in paired_model_inputs(b).items()});p=double.rollout(hh,ss,z.double(),16,b['crop_start']);qp=task_loss(p.float(),b,t,l)
    precision=dict(TRAIN_step=12001,NFE=16,output_max_delta=float((a-p).abs().max()),output_mean_delta=float((a-p).abs().mean()),FP32_task=float(qa),FP64_task=float(qp),task_abs_delta=float(abs(qa-qp)))
    with (OUT/'TRAIN_precision_probe.json').open('x') as f:json.dump(precision,f,indent=2)
    raw=[]
    for rec in records[12000:12016]:
        batch,*_=data.batch(rec)
        for i,n in enumerate(batch['pair_lengths'].tolist()):
            yy=batch['paired_target'][i,:n]
            for g,a,b in GROUPS:
                q=yy[:,a:b];raw.append(dict(step=rec['step'],source=rec['source_indices'][i],group=g,length=n,speed=float(q.diff(dim=0).abs().mean()),acceleration=float(q.diff(dim=0).diff(dim=0).abs().mean())))
    with (OUT/'raw_TRAIN_activity.json').open('x') as f:json.dump(dict(protocol='TRAIN16 continuation paired raw absolute crops, no Processor; not same-input conditional variance',rows=raw),f,indent=2)
    del model,h,a,p
    manifest=json.loads(Path('runs/phase24/evaluation_v1/multitarget_development_manifest.json').read_text());ti=json.loads(Path('runs/phase24/evaluation_v1/processed_targets/completed.json').read_text())['result']['files']
    mean=torch.from_numpy(np.load('external/FaceVerse/mean_face.npy')).float();std=torch.from_numpy(np.load('external/FaceVerse/std_face.npy')).float().clamp_min(1e-8);rows=[]
    for i in (0,20,40,60):
        recipient=manifest['sources'][i];session=recipient['clip_id'].split('/')[1];members=[j for j,s in enumerate(manifest['sources']) if s['clip_id'].split('/')[1]==session];j=members[(members.index(i)+1)%len(members)];donor=manifest['sources'][j];n=min(recipient['length'],donor['length'])
        # Same recipient noise for both source contexts, prefix stable for any length.
        z=recording_noise(cfg,recipient['clip_id'],max(recipient['length'],donor['length'])).cuda().double();pred=[]
        for s in (recipient,donor):
            streams={k:torch.from_numpy(np.load(v['path'])).float() for k,v in s['files'].items()};face=streams['speaker_3dmm'];streams['speaker_3dmm']=((face[:,0] if face.ndim==3 else face)-mean)/std
            pred.append(export(double,{k:v.cuda().double() for k,v in streams.items()},s['length'],z[:,:s['length']],16)[:,:n].float())
        targets=torch.from_numpy(np.load(ti[i]['path']))[:,:n];correct,wrong=pred
        rows.append(dict(recipient=i,donor=j,common_length=n,output_mean_delta=float((correct-wrong).abs().mean()),paired_CCC_correct=float(ccc25(correct,targets[0]).mean()),paired_CCC_shuffled=float(ccc25(wrong,targets[0]).mean()),FRC_correct=float(ccc25(correct[:,None],targets[None]).max(1).values.sum()),FRC_shuffled=float(ccc25(wrong[:,None],targets[None]).max(1).values.sum())))
    with (OUT/'condition_probe.json').open('x') as f:json.dump(dict(rule='fixed cyclic same-session donor; identical recipient absolute-frame noise; common valid prefix',rows=rows),f,indent=2)
if __name__=='__main__':main()
