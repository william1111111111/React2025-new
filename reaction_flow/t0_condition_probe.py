"""Only four final T0 recipients, cached correct outputs and fixed donor inference."""
import json
from pathlib import Path
import numpy as np
import torch
from .train import configure_flow
from .sampler import load_checkpoint
from .export import recording_noise,export_full
from .t0_next_analysis import OUT,PATHS,csvout,load
from hirp.phase24 import normalization,verify_files
from hirp.phase15_audit import sha256_file,canonical_hash
from mam_target.losses import ccc25
CP=Path('runs/reaction_flow/task_dynamics_v1/T0-task/attempt_000/checkpoints/step_014000.pt')
def main():
 configure_flow();assert sha256_file(CP)=='e6199183cdc3526e1fc1390c89057e40514af8d2f2101d69a5e1dabb7f43f4d6'
 model,meta=load_checkpoint(CP,'cuda:0');model.double()
 old=json.loads(Path(PATHS['T0']).read_text());identity=old['eval_identity'];assert identity['checkpoint_sha256']==sha256_file(CP) and identity['solver']=='Euler16';assert 'FP64' in identity['sampling_precision']
 mp=Path('runs/phase24/evaluation_v1/multitarget_development_manifest.json');assert sha256_file(mp)==identity['target_manifest_sha256'];manifest=json.loads(mp.read_text());normalization(manifest)
 tp=Path('runs/phase24/evaluation_v1/processed_targets/completed.json');assert sha256_file(tp)==identity['processed_index_sha256'];ti=json.loads(tp.read_text())['result']['files']
 import hashlib
 fingerprints=[dict(clip_id=s['clip_id'],length=s['length'],sha256=hashlib.sha256(recording_noise(model.config,s['clip_id'],s['length']).numpy().tobytes()).hexdigest()) for s in manifest['sources']]
 assert canonical_hash(fingerprints)==identity['noise_content_sha256']
 mean=torch.from_numpy(np.load('external/FaceVerse/mean_face.npy')).float();std=torch.from_numpy(np.load('external/FaceVerse/std_face.npy')).float().clamp_min(1e-8);rows=[]
 for i in (0,20,40,60):
  recipient=manifest['sources'][i];session=recipient['clip_id'].split('/')[1];members=[j for j,s in enumerate(manifest['sources']) if s['clip_id'].split('/')[1]==session];j=members[(members.index(i)+1)%len(members)];donor=manifest['sources'][j];n=min(recipient['length'],donor['length']);verify_files(list(donor['files'].values()))
  correct=load(old['result']['exports'][i])[:,:n];target=load(ti[i])[:,:n]
  z=recording_noise(model.config,recipient['clip_id'],donor['length']).cuda().double()
  streams={k:torch.from_numpy(np.load(v['path'])).float() for k,v in donor['files'].items()};face=streams['speaker_3dmm'];streams['speaker_3dmm']=((face[:,0] if face.ndim==3 else face)-mean)/std
  wrong=export_full(model,**{k:v.cuda().double() for k,v in streams.items()},source_length=donor['length'],noise=z).float()[:,:n]
  rows.append(dict(recipient=i,recipient_id=recipient['clip_id'],donor=j,donor_id=donor['clip_id'],common_length=n,output_mean_delta=float((correct-wrong).abs().mean()),paired_CCC_correct=float(ccc25(correct,target[0]).mean()),paired_CCC_shuffled=float(ccc25(wrong,target[0]).mean()),FRC_correct=float(ccc25(correct[:,None],target[None]).max(1).values.sum()),FRC_shuffled=float(ccc25(wrong[:,None],target[None]).max(1).values.sum())))
  print(rows[-1],flush=True)
 csvout('T0_condition_probe.csv',rows)
 with (OUT/'T0_condition_probe.json').open('x') as f:json.dump(dict(checkpoint=str(CP),checkpoint_sha256=sha256_file(CP),production='FP64 Euler16 K10 native continuous AU, unchanged 750 blocks',correct_exports='reused with checkpoint, noise content, target manifest, precision and solver checked',rule='same four cyclic within-session donors as A; identical recipient absolute-frame noise; common valid lengths',rows=rows),f,indent=2)
if __name__=='__main__':main()
