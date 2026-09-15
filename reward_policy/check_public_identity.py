"""Speaker-only step0 DEV export equivalence; no DEV targets or reward computation."""
import numpy as np
import torch
from reaction_flow.train import configure_flow
from reaction_flow.export import recording_noise
from semantic_supervision.bert_experiments.common import Sources
from .generator import Generator
from .policy import Policy
from .common import OUT,BERT,read,write

def main():
    configure_flow();torch.set_num_threads(4);g=Generator();p=Policy().cuda().double();source=read('runs/phase24/evaluation_v1/multitarget_development_manifest.json')['sources'][0];name=source['clip_id'];sources=Sources('val');pts=sources.rows[name]['frame_pts'];streams={k:torch.from_numpy(np.load(v['path'])).float() for k,v in source['files'].items()};mean=torch.from_numpy(np.load('external/FaceVerse/mean_face.npy')).float();std=torch.from_numpy(np.load('external/FaceVerse/std_face.npy')).float().clamp_min(1e-8);face=streams['speaker_3dmm'];streams['speaker_3dmm']=((face[:,0] if face.ndim==3 else face)-mean)/std
    events=[sources.crop(name,start,min(750,len(pts)-start))[0] for start in range(0,len(pts),750)];noise=recording_noise(g.model.base.config,name,len(pts)).cuda().double()
    y=g.sample([streams[k] for k in ('speaker_audio','speaker_emotion','speaker_3dmm')],pts,events,noise,p).float().cpu()
    prior=read(BERT/'task_multitarget/seed123_P2-bert_step15000.json')['result']['exports'][0];old=torch.from_numpy(np.load(prior['path']));error=float((y-old).abs().max());assert error<1e-6
    prefix=g.sample([streams[k] for k in ('speaker_audio','speaker_emotion','speaker_3dmm')],pts,events,noise[:2],p).float().cpu();pe=float((prefix-y[:2]).abs().max());assert pe<1e-6
    write(OUT/'PUBLIC_IDENTITY.json',dict(source_only=True,DEV_targets_read=False,source=name,frames=len(pts),K=10,step0_parent_max_error=error,K_prefix_error=pe,scope='one full recording export equivalence, not a new metric evaluation'))
    print('public identity passed',error,pe)
if __name__=='__main__':main()
