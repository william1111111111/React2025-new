"""Unchanged official DEV80 targets/metrics, FP64 Euler16 source-only weak inference."""
import argparse,json,time,sys,hashlib,subprocess
from pathlib import Path
import numpy as np,torch
from reaction_flow.sampler import ReactionFlow
from reaction_flow.config import FlowConfig
from hirp.config import HiRPConfig
from reaction_flow.export import recording_noise
from reaction_flow.train import configure_flow
from hirp.phase24 import normalization,verify_files,LEGACY
from hirp.phase15_audit import canonical_hash,sha256_file
from hirp.phase23_cache import cached,save_cached
from hirp.task_phase24 import raw_metrics,check_prediction
from semantic_supervision.models.semantic_flow import SemanticFlow
from semantic_supervision.models.auto_weak import read_weak,crop_events
from .controlled import OUT,MODES
OLD=Path('runs/phase24/evaluation_v1')
def main(arm):
 configure_flow();path=OUT/'training'/arm/'checkpoints/step_000500.pt';saved=torch.load(path,map_location='cpu',weights_only=True);assert saved['format_version']=='auto-weak-flow-v1' and saved['arm']==arm and saved['step']==500 and not saved['diagnostic']
 metadata=saved['base_metadata'];cfg=dict(metadata['flow_config']);cfg['checkpoints']=tuple(cfg['checkpoints']);base=ReactionFlow(FlowConfig(**cfg),HiRPConfig(**metadata['condition_config']));base.load_state_dict(saved['base_state']);model=SemanticFlow(base,.1);model.semantic.load_state_dict(saved['semantic_state']);model.cuda().double().eval()
 manifest=json.loads((OLD/'multitarget_development_manifest.json').read_text());normalization(manifest);targets_payload=json.loads((OLD/'processed_targets/completed.json').read_text());assert canonical_hash(targets_payload['result'])==targets_payload['result_sha256'];verify_files(targets_payload['result']['files'])
 dep=json.loads((OLD/'dependency_snapshot.json').read_text());verify_files(dep['legacy'])
 index=json.loads((OUT/'dev_weak/cache/index.json').read_text());devjob={r['id']:r for r in json.loads((OUT/'dev_job/job.json').read_text())['records']}
 label=f'seed123_{arm}_step14500';dest=OUT/'exports'/label;dest.mkdir(parents=True,exist_ok=True);out=OUT/'task_multitarget';out.mkdir(exist_ok=True)
 identity={'version':'auto-weak-source-only-evaluation-v1','checkpoint_sha256':sha256_file(path),'protocol_sha256':sha256_file(OUT/'PROTOCOL.json'),'dev_cache_index_sha256':sha256_file(OUT/'dev_weak/cache/index.json'),'event_typer_sha256':sha256_file(OUT/'checkpoints/event_typer.pkl'),'target_manifest_sha256':sha256_file(OLD/'multitarget_development_manifest.json'),'processed_index_sha256':sha256_file(OLD/'processed_targets/completed.json'),'legacy':dep['legacy'],'precision':'FP64 Euler16 -> FP32 official metrics','K':10,'arm':arm,'source_only':True,'transcript_assisted':True,'code_hashes':{str(p):sha256_file(p) for folder in ('semantic_supervision','reaction_flow') for p in sorted(Path(folder).rglob('*.py'))}}
 cached_result=cached(out/(label+'.json'),identity)
 if cached_result is not None:return
 mean=torch.from_numpy(np.load('external/FaceVerse/mean_face.npy')).float();std=torch.from_numpy(np.load('external/FaceVerse/std_face.npy')).float().clamp_min(1e-8)
 preds=[];targets=[];speakers=[];exports=[];nonnull=0;started=time.time()
 for i,source in enumerate(manifest['sources']):
  verify_files(list(source['files'].values()));cid='rec_'+hashlib.sha256(source['clip_id'].encode()).hexdigest()[:16];m=index[cid];record=devjob[cid];es=read_weak(OUT/'dev_weak/cache'/(cid+'.json'),split='val',expected_cache_sha256=m['cache_sha256'],expected_media_hashes=m['media_hashes'],expected_policy_sha256=m['policy_sha256']);nonnull+=bool(es)
  vid=Path('data/val/video-face-crop/speaker')/(record['relative']+'.mp4');pts=[float(f['best_effort_timestamp_time']) for f in json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','json',str(vid)]))['frames']];assert len(pts)==source['length']
  actual=[sha256_file(Path('data/val')/d/'speaker'/(record['relative']+s)) for d,s in [('audio','.wav'),('video-face-crop','.mp4'),('text','.txt')]];assert sorted(actual)==sorted(m['media_hashes'])
  streams={k:torch.from_numpy(np.load(v['path'])).float() for k,v in source['files'].items()};face=streams['speaker_3dmm'];streams['speaker_3dmm']=((face[:,0] if face.ndim==3 else face)-mean)/std
  sidecar=dest/f'{i:03d}.json';ident={'evaluation_sha256':canonical_hash(identity),'source':i};item=cached(sidecar,ident)
  if item is None:
   noise=recording_noise(base.config,source['clip_id'],source['length']).cuda().double();outputs=[]
   with torch.no_grad():
    for start in range(0,source['length'],750):
     n=min(750,source['length']-start);inputs=[]
     for key in ['speaker_audio','speaker_emotion','speaker_3dmm']:
      x=streams[key].cuda().double();pad=x.new_zeros(1,750,x.shape[-1]);pad[0,:n]=x[start:start+n];inputs.append(pad)
     z=noise.new_zeros(1,10,750,24);z[0,:,:n]=noise[:,start:start+n];events=[crop_events(es,pts,start,start+n,trajectory_frames=source['length'])];lengths=torch.tensor([n],device='cuda');offset=torch.tensor([start],device='cuda')
     full=model.sample(*inputs,lengths,events,noise=z,mode=MODES[arm],integration_steps=16,position_offset=offset)
     repeat=torch.cat([model.sample(*inputs,lengths,events,noise=z[:,k:k+3],mode=MODES[arm],integration_steps=16,position_offset=offset) for k in range(0,10,3)],dim=1)
     error=float((full-repeat).abs().max());assert error<1e-10
     outputs.append(full[0,:,:n].float().cpu())
   prediction=torch.cat(outputs,dim=1);checks=check_prediction(prediction,source['length']);file=dest/f'{i:03d}.npy'
   with file.open('xb') as f:np.save(f,prediction.numpy())
   item={'path':str(file),'sha256':sha256_file(file),'checks':checks};save_cached(sidecar,ident,item)
  else:verify_files([item]);prediction=torch.from_numpy(np.load(item['path']))
  preds.append(prediction);targets.append(torch.from_numpy(np.load(targets_payload['result']['files'][i]['path'])));speakers.append(streams['speaker_emotion']);exports.append(item);print(arm,i+1,'/80',flush=True)
 sys.path.insert(0,str(LEGACY));from framework.utils.compute_metrics import compute_metrics
 metrics=compute_metrics(speakers,preds,targets,threads=4,metric_names=manifest['metrics']);raw=raw_metrics(preds,targets,speakers);assert abs(metrics['FRC']-np.mean([r['FRC'] for r in raw]))<1e-6
 result={'model':label,'metrics':metrics,'raw':raw,'exports':exports,'non_null_sources':nonnull,'seconds':time.time()-started};save_cached(out/(label+'.json'),identity,result);print(metrics,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=list(MODES),required=True);a=p.parse_args();main(a.arm)
