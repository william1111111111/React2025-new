import torch,numpy as np
from .common import *
from .batched import BatchedGenerator
from reaction_flow.train import configure_flow
from reaction_flow.export import recording_noise

def main():
 configure_flow();torch.set_num_threads(4);m=read(ROOT/'manifest.json');s=m['sources'][0];gen=BatchedGenerator();n=s['length'];streams={k:torch.from_numpy(np.load(v['path'])).float() for k,v in s['files'].items()};f=streams['speaker_3dmm'];mean=torch.from_numpy(np.load('external/FaceVerse/mean_face.npy')).float();std=torch.from_numpy(np.load('external/FaceVerse/std_face.npy')).float().clamp_min(1e-8);streams['speaker_3dmm']=((f[:,0] if f.ndim==3 else f)-mean)/std;noise=recording_noise(gen.model.base.config,s['clip_id'],n).cuda().double();result={}
 for a in ['N0-quality','N1-quality-coverage']:
  saved=torch.load(m['checkpoints'][a]['path'],map_location='cpu',weights_only=True);gen.model.base.velocity.load_state_dict(saved['student']);new=gen.sample([streams[k] for k in ('speaker_audio','speaker_emotion','speaker_3dmm')],s['frame_pts'],[None for _ in range(0,n,750)],noise).float().cpu();row=read(ROOT/'exports'/a/'0000.json');old=torch.from_numpy(np.load(row['export']['path']));error=float((old-new).abs().max());torch.testing.assert_close(old,new,atol=1e-6,rtol=1e-6);result[a]=dict(max_abs_error=error,exact_equal=torch.equal(old,new));print(a,result[a],flush=True)
 write(ROOT/'BATCHING_ARM_CHECKS.json',result)
if __name__=='__main__':main()
