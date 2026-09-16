import sys,random,json,subprocess,concurrent.futures
import numpy as np
from .common import *
from hirp.phase24 import LEGACY,official_selection

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 if (ROOT/'manifest.json').exists():return
 sys.path.insert(0,str(LEGACY))
 from regnn.eval_conditional_regnn_official_test import initialize_official_hydra_runtime
 initialize_official_hydra_runtime(LEGACY)
 from dataset.react_2025 import ReactionDataset
 ds=ReactionDataset(root_dir=str(Path('data').resolve()),split='test',clip_length=750,load_video_s=False,load_video_l=False)
 old=read(DEV/'multitarget_development_manifest.json');existing={}
 scope=read(ROOT/'SCOPE.json')['directions'];population=[(i,str(p)) for i,p in enumerate(ds.speaker_path_list) if scope=='both' or p.parts[0]=='speaker'];assert len(population)==(1142 if scope=='both' else 571)
 def row(item):
  idx,name=item;oldidx=None
  if name in existing:oldidx,s=existing[name];s=dict(s)
  else:
   files={};lengths={}
   for key,folder in [('speaker_audio','audio-features'),('speaker_emotion','facial-attributes'),('speaker_3dmm','coefficients')]:
    p=Path('data/test')/folder/(name+'.npy');files[key]=dict(path=str(p),sha256=sha(p));lengths[key]=len(np.load(p,mmap_mode='r'))
   assert len(set(lengths.values()))==1,(name,lengths)
   seed=36100+idx;refs=official_selection(ds.listener_path_list[idx],ds.gt_path_list[idx],random.Random(seed));targets=[]
   for j,ref in enumerate(refs):
    p=Path('data/test/facial-attributes')/ref.with_suffix('.npy');assert ref.parts[0]!=Path(name).parts[0];targets.append(dict(path=str(p),sha256=sha(p),length=len(np.load(p,mmap_mode='r')),rank=j))
   s=dict(clip_id=name,length=lengths['speaker_audio'],files=files,targets=targets,selection_seed=seed,pool_order=list(map(str,ds.gt_path_list[idx])),official_dataset_index=idx)
  video=Path('data/test/video-face-crop')/(name+'.mp4')
  pts=[float(x['best_effort_timestamp_time']) for x in json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','json',str(video)]))['frames']]
  assert len(pts)==s['length'] and np.all(np.diff(pts)>0),name
  return dict(**s,dev80_index=oldidx,frame_pts=pts,video_sha256=sha(video))
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:rows=list(ex.map(row,population))
 write(ROOT/'manifest.json',dict(split='test',sources=rows,population=len(rows),official_bidirectional_entries=len(ds.speaker_path_list),scope='local TEST snapshot; direction defined by official loader; input is source only for that direction; content audit shows local TEST duplicates VAL',normalization=old['normalization'],targets='official paired+9 rule with opposite-role same-session pool; fixed per-source seed, identical across models',new_processor_seed='36200+official_dataset_index',semantic='NULL for every TEST source: no split-legal TEST semantic cache; VAL annotations never read',K=10,checkpoints={a:dict(path=str(PARENT if a=='P2' else NFT/'training'/a/'checkpoints/step_001000.pt'),sha256=sha(PARENT if a=='P2' else NFT/'training'/a/'checkpoints/step_001000.pt')) for a in ARMS}))
 print('manifest complete',len(rows),flush=True)
if __name__=='__main__':main()
