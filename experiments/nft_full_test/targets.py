import sys
import numpy as np
import torch
from .common import *
from hirp.phase24 import LEGACY,verify_files,normalization
from reaction_flow.train import configure_flow

def main():
 configure_flow();torch.set_num_threads(4);manifest=read(ROOT/'manifest.json');normalization(manifest);sys.path.insert(0,str(LEGACY))
 from regnn.eval_conditional_regnn_official_test import initialize_official_hydra_runtime
 initialize_official_hydra_runtime(LEGACY)
 from framework.modules.post_processor import Processor
 processor=Processor(cfg_dir=str(LEGACY),ckpt_dir=str(LEGACY/'pretrained_models/post_processor'),device=torch.device('cuda:0'),clip_len_test=1000,num_preds=10)
 old=read(DEV/'processed_targets/completed.json')['result']['files'];dest=ROOT/'targets';dest.mkdir(exist_ok=True);files=[]
 for i,s in enumerate(manifest['sources']):
  if s['dev80_index'] is not None:item=old[s['dev80_index']];verify_files([item])
  elif (dest/f'{i:04d}.json').exists():item=read(dest/f'{i:04d}.json');verify_files([item])
  else:
   verify_files(s['targets']);raw=[torch.from_numpy(np.load(t['path'])).float() for t in s['targets']];torch.manual_seed(36200+s['official_dataset_index'])
   with torch.no_grad():y=processor.forward([torch.zeros(10,s['length'],25)],[raw])[0].cpu()
   assert y.shape==(10,s['length'],25) and torch.isfinite(y).all()
   if len(raw[0])==s['length']:assert torch.equal(y[0],raw[0])
   path=dest/f'{i:04d}.npy';tmp=path.with_suffix('.tmp')
   with tmp.open('wb') as f:np.save(f,y.numpy())
   tmp.replace(path);item=dict(path=str(path),sha256=sha(path));write(dest/f'{i:04d}.json',item)
  files.append(item);write(ROOT/'target_status.json',dict(completed=i+1,total=manifest['population']));print('TARGET',i+1,'/',manifest['population'],flush=True)
 write(ROOT/'targets.json',dict(files=files,manifest_sha256=sha(ROOT/'manifest.json'),processor_source_sha256=sha(LEGACY/'framework/modules/post_processor.py')))
if __name__=='__main__':main()
