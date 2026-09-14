"""Unchanged official full DEV80 evaluation plus fixed, separate interventions."""
import argparse,time,sys
from dataclasses import replace
import numpy as np,torch
from hirp.config import HiRPConfig
from reaction_flow.config import FlowConfig
from reaction_flow.sampler import ReactionFlow
from reaction_flow.export import recording_noise
from reaction_flow.train import configure_flow
from reaction_flow.dynamics import decomposition
from hirp.phase24 import normalization,verify_files,LEGACY
from hirp.phase15_audit import canonical_hash
from hirp.phase23_cache import cached,save_cached
from hirp.task_phase24 import raw_metrics,check_prediction
from .common import *
from .model import BertSemanticFlow

def label_for(arm,step,perturb='correct'):return f'seed123_{arm}_step{14000+step}'+('' if perturb=='correct' else '_'+perturb)
def intervene(events,source,start,kind,table,frames):
 if kind=='correct':return events,False
 mapping=table[kind][f'{source}|{start}'];changed=False;out=[]
 for e in events or []:
  v=mapping.get(e.event_id)
  if v is None:out.append(e);continue
  if kind=='text':out.append(replace(e,text=v));changed|=v!=e.text
  else:
   interval=tuple(v);left=max(0,int(np.searchsorted(frames,interval[0],side='right'))-1);right=int(np.searchsorted(frames,interval[1],side='left'));out.append(replace(e,interval=interval,frame_interval=(left,right)));changed|=interval!=e.interval
 return out or None,changed

def compact(raw,preds,sources):
 rows=[]
 for i,(r,p,s) in enumerate(zip(raw,preds,sources)):
  matrix=np.array(r['CCC_candidate_target']);dist=np.array(r['S_MSE_candidate_pair']);td=np.array(r['temporal_S_MSE_candidate_pair']);k=len(dist);total,parts=decomposition(p.double())
  rows.append(dict(index=i,clip_id=s['clip_id'],session=s['clip_id'].split('/')[1],FRC=r['FRC'],paired_CCC_mean=r['single_paired_FRC']/k,smse=float(dist.sum()/(k*(k-1))),temporal_smse=float(td.sum()/(k*(k-1))),FRVar=float(np.mean(r['candidate_channel_temporal_variance'])),target_best_CCC_mean=float(matrix.max(0).mean()),DC=float(parts['DC']),slow=float(parts['slow']),fast=float(parts['fast']),spread_total=float(total)))
 return rows

def summaries(rows,strata):
 fields=['FRC','paired_CCC_mean','smse','FRVar','target_best_CCC_mean','DC','slow','fast'];result={}
 for key,indices in strata.items():
  selected=[rows[i] for i in indices];sessions=sorted({r['session'] for r in selected})
  result[key]=dict(sources=len(selected),source_mean={k:float(np.mean([r[k] for r in selected])) if selected else None for k in fields},session_equal_mean={k:float(np.mean([np.mean([r[k] for r in selected if r['session']==s]) for s in sessions])) if sessions else None for k in fields})
 return result

def main(arm,step,perturb='correct'):
 assert step in (500,1000) and (perturb=='correct' or step==1000 and arm in ('P2-bert','P3-bert-time'))
 protocol=verify();configure_flow();torch.set_num_threads(4)
 for p,h in read(OUT/'CODE_IDENTITY.json').items():assert sha(p)==h
 path=OUT/'training'/arm/'checkpoints'/f'step_{step:06d}.pt';saved=torch.load(path,map_location='cpu',weights_only=True);assert saved['format_version']=='bert-semantic-controls-v1' and saved['arm']==arm and saved['step']==step and saved['diagnostic'] is None
 assert saved['run_identity_sha256']==sha(OUT/'RUN_IDENTITY.json')
 assert saved['code_identity_sha256']==sha(OUT/'CODE_IDENTITY.json') and saved['protocol_sha256']==sha(OUT/'PROTOCOL.json') and saved['content_index_sha256']==sha(OUT/'content/index.json')
 cfg=dict(saved['base_metadata']['flow_config']);cfg['checkpoints']=tuple(cfg['checkpoints']);base=ReactionFlow(FlowConfig(**cfg),HiRPConfig(**saved['base_metadata']['condition_config']));model=BertSemanticFlow(base,arm,ContentCache());model.load_state_dict(saved['model']);model.cuda().double().eval()
 manifest=read(EVAL/'multitarget_development_manifest.json');normalization(manifest);targets_payload=read(EVAL/'processed_targets/completed.json');assert canonical_hash(targets_payload['result'])==targets_payload['result_sha256'];verify_files(targets_payload['result']['files']);dep=read(EVAL/'dependency_snapshot.json');verify_files(dep['legacy'])
 sources=Sources('val');table=read(OUT/'interventions.json');strata=read(OUT/'strata.json');label=label_for(arm,step,perturb);dest=OUT/'exports'/label;dest.mkdir(parents=True,exist_ok=True);task=OUT/'task_multitarget'/(label+'.json');task.parent.mkdir(exist_ok=True)
 identity=dict(version='frozen-bert-controls-eval-v1',checkpoint_sha256=sha(path),protocol_sha256=sha(OUT/'PROTOCOL.json'),code_identity_sha256=sha(OUT/'CODE_IDENTITY.json'),content_index_sha256=sha(OUT/'content/index.json'),strata_sha256=sha(OUT/'strata.json'),interventions_sha256=sha(OUT/'interventions.json'),target_manifest_sha256=sha(EVAL/'multitarget_development_manifest.json'),processed_index_sha256=sha(EVAL/'processed_targets/completed.json'),legacy=dep['legacy'],arm=arm,step=step,perturbation=perturb,source_only=True,transcript_assisted=True,precision='FP64 Euler16 -> FP32 official metrics',K=10)
 if cached(task,identity) is not None:return
 correct=None
 if perturb!='correct':correct=read(OUT/'task_multitarget'/(label_for(arm,step)+'.json'))['result']
 mean=torch.from_numpy(np.load(ROOT/'external/FaceVerse/mean_face.npy')).float();std=torch.from_numpy(np.load(ROOT/'external/FaceVerse/std_face.npy')).float().clamp_min(1e-8)
 preds=[];targets=[];speakers=[];exports=[];eligible=[];started=time.time();generation_seconds=0.;repeat_max=0.;perturb_changes=[]
 for i,source in enumerate(manifest['sources']):
  verify_files(list(source['files'].values()));name=source['clip_id'];sidecar=dest/f'{i:03d}.json';ident=dict(evaluation_sha256=canonical_hash(identity),source=i);item=cached(sidecar,ident)
  measurable=perturb=='correct' or any(table[perturb][f'{name}|{start}'] for start in range(0,source['length'],750))
  if perturb!='correct' and measurable:eligible.append(i)
  streams={k:torch.from_numpy(np.load(v['path'])).float() for k,v in source['files'].items()};face=streams['speaker_3dmm'];streams['speaker_3dmm']=((face[:,0] if face.ndim==3 else face)-mean)/std
  if item is None and not measurable:
   item=correct['exports'][i];verify_files([item]);save_cached(sidecar,ident,item)
  if item is None:
   noise=recording_noise(base.config,name,source['length']).cuda().double();outputs=[];begin=time.time()
   with torch.no_grad():
    for start in range(0,source['length'],750):
     n=min(750,source['length']-start);inputs=[]
     for key in ['speaker_audio','speaker_emotion','speaker_3dmm']:
      x=streams[key].cuda().double();pad=x.new_zeros(1,750,x.shape[-1]);pad[0,:n]=x[start:start+n];inputs.append(pad)
     z=noise.new_zeros(1,10,750,24);z[0,:,:n]=noise[:,start:start+n];es,q,d=sources.crop(name,start,n);es,changed=intervene(es,name,start,perturb,table,q[:n]);lengths=torch.tensor([n],device='cuda');offset=torch.tensor([start],device='cuda');frames=torch.tensor(q[None],device='cuda',dtype=torch.float64)
     y=model.sample(*inputs,lengths,[es],frames,[d],noise=z,position_offset=offset)
     if i==0 and start==0 and perturb=='correct':
      repeat=torch.cat([model.sample(*inputs,lengths,[es],frames,[d],noise=z[:,k:k+3],position_offset=offset) for k in range(0,10,3)],1);repeat_max=float((y-repeat).abs().max());assert repeat_max<1e-10
     outputs.append(y[0,:,:n].float().cpu())
   torch.cuda.synchronize();elapsed=time.time()-begin;generation_seconds+=elapsed;prediction=torch.cat(outputs,1);checks=check_prediction(prediction,source['length']);file=dest/f'{i:03d}.npy'
   # Atomic array + sidecar. A stray array from interruption can be safely replaced.
   tmp=file.with_suffix('.tmp')
   with tmp.open('wb') as f:np.save(f,prediction.numpy())
   tmp.replace(file);item=dict(path=str(file),sha256=sha(file),checks=checks,generation_seconds=elapsed);save_cached(sidecar,ident,item)
  else:verify_files([item]);prediction=torch.from_numpy(np.load(item['path']))
  if 'prediction' not in locals():prediction=torch.from_numpy(np.load(item['path']))
  preds.append(prediction);targets.append(torch.from_numpy(np.load(targets_payload['result']['files'][i]['path'])));speakers.append(streams['speaker_emotion']);exports.append(item)
  if perturb!='correct' and measurable:
   original=torch.from_numpy(np.load(correct['exports'][i]['path']));perturb_changes.append(dict(index=i,mean_abs_prediction_change=float((prediction-original).abs().mean())))
  del prediction
  print(label,i+1,'/80',flush=True)
 sys.path.insert(0,str(LEGACY));from framework.utils.compute_metrics import compute_metrics
 metrics=compute_metrics(speakers,preds,targets,threads=4,metric_names=manifest['metrics']);raw=raw_metrics(preds,targets,speakers);assert abs(metrics['FRC']-np.mean([r['FRC'] for r in raw]))<1e-6
 rows=compact(raw,preds,manifest['sources']);groups=summaries(rows,strata)
 assert abs(groups['all']['source_mean']['smse']-metrics['smse'])<1e-6 and abs(groups['all']['source_mean']['FRVar']-metrics['FRVar'])<1e-6
 result=dict(model=label,metrics=metrics,raw=raw,exports=exports,per_source=rows,subgroups=groups,seconds=time.time()-started,inference_seconds=sum(x.get('generation_seconds',0) for x in exports),candidate_chunk_error=repeat_max,peak_gpu_bytes=torch.cuda.max_memory_allocated(),perturbation=perturb,measurable_sources=eligible if perturb!='correct' else list(range(80)),perturbation_changes=perturb_changes)
 if perturb!='correct':
  result['diagnostic_only']=True;result['measurable_subset']=summaries(rows,dict(measurable=eligible));base_rows=correct['per_source'];result['paired_deltas']=[dict(index=i,FRC=rows[i]['FRC']-base_rows[i]['FRC'],paired_CCC_mean=rows[i]['paired_CCC_mean']-base_rows[i]['paired_CCC_mean']) for i in eligible]
 save_cached(task,identity,result);print(metrics,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--step',type=int,choices=[500,1000],required=True);p.add_argument('--perturb',choices=['correct','text','time'],default='correct');a=p.parse_args();main(a.arm,a.step,a.perturb)
