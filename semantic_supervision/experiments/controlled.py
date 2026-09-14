"""Matched 500-update E0/E-text/E-event pilot; fixed source evidence and T0 task loss."""
import argparse,json,hashlib,pickle,random,time,subprocess
from pathlib import Path
import numpy as np,torch
from reaction_flow.config import FlowConfig
from reaction_flow.data import FlowData,draw
from reaction_flow.flow_path import flow_path,flow_loss
from reaction_flow.sampler import load_checkpoint,metadata
from reaction_flow.task_dynamics_train import get_batch,task_parts
from reaction_flow.train import configure_flow
from semantic_supervision.models.auto_weak import read_weak,crop_events
from semantic_supervision.models.semantic_flow import SemanticFlow
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/reaction_flow/semantic_controlled_v1';PARENT=ROOT/'runs/reaction_flow/task_dynamics_v1/T0-task/attempt_000/checkpoints/step_014000.pt'
MODES={'E0':'null','E_text':'text','E_event':'event'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2));t.replace(p)
def read(path,m,split='train'):
 return read_weak(path,split=split,expected_cache_sha256=m['cache_sha256'],expected_media_hashes=m['media_hashes'],expected_policy_sha256=m['policy_sha256'])
def prepare():
 with (OUT/'checkpoints/event_typer.pkl').open('rb') as f:typer=pickle.load(f)
 index={};mapping={};cache=OUT/'train_cache';cache.mkdir(exist_ok=True)
 groups=[(ROOT/'runs/reaction_flow/semantic_auto_weak_v1',ROOT/'runs/reaction_flow/semantic_alignment_pilot_v1/job/job.json'),(ROOT/'runs/reaction_flow/semantic_auto_weak_v1/expansion/weak',ROOT/'runs/reaction_flow/semantic_auto_weak_v1/expansion/job/job.json')]
 for folder,job in groups:
  mapping.update({r['id']:r['relative'] for r in json.loads(job.read_text())['records']})
  for cid,m in json.loads((folder/'cache/index.json').read_text()).items():
   obj=json.loads((folder/'cache'/(cid+'.json')).read_text())
   types=typer.predict([e['text'] for e in obj['events']])
   for e,k in zip(obj['events'],types):e['event_type']=str(k);e['source_version_hashes'].append(m['cache_sha256'])
   obj['teacher_hashes'].append(sha(OUT/'checkpoints/event_typer.pkl'))
   p=cache/(cid+'.json');write(p,obj);index[cid]={**m,'cache_sha256':sha(p)}
 write(cache/'index.json',index)
 cfg=FlowConfig();data=FlowData(cfg,'cpu');lookup={str(p.relative_to(data.inner.pool.dataset.directory/'facial-attributes/speaker').with_suffix('')):i for i,p in enumerate(data.inner.pool.dataset.records)}
 cohort={};by_session={}
 for cid,m in index.items():
  if not read(cache/(cid+'.json'),m):continue
  rel=mapping[cid];vid=ROOT/'data/train/video-face-crop/speaker'/(rel+'.mp4')
  pts=[float(x['best_effort_timestamp_time']) for x in json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','json',str(vid)]))['frames']]
  for name in ['audio-features','facial-attributes','coefficients']:assert len(np.load(ROOT/'data/train'/name/'speaker'/(rel+'.npy'),mmap_mode='r'))==len(pts)
  cohort['speaker/'+rel]={'clip_id':cid,'dataset_index':lookup[rel],'frame_pts':pts,'cache':m}
  by_session.setdefault(rel.split('/')[0],[]).append(lookup[rel])
 rng=random.Random(123);records=[];sessions=sorted(by_session)
 for step in range(14001,14501):
  session=rng.choice(sessions);records.append({'step':step,'session_id':session,'source_indices':rng.choices(by_session[session],k=4),'crop_occurrences':[rng.randrange(1000000) for _ in range(4)],'target_slots':[rng.randrange(4) for _ in range(4)]})
 write(OUT/'cohort.json',cohort);write(OUT/'schedule.json',records)
 protocol={'parent_sha256':sha(PARENT),'steps_per_arm':500,'checkpoint_steps':[100,500],'arms':MODES,'batch_size':4,'crop_frames':750,'seed':123,'noise':'unchanged independent local recording/training noise','loss':'unchanged T0 FM + inherited lambda_ref * task_parts; no dynamic loss','optimizer':'inherit T0 AdamW velocity state, fresh semantic parameter group; both lr2e-5','dropout':.1,'schedule_sha256':sha(OUT/'schedule.json'),'cohort_sha256':sha(OUT/'cohort.json'),'event_typer_sha256':sha(OUT/'checkpoints/event_typer.pkl'),'cohort_sources':len(cohort),'semantic_regime':'auto_weak; hard gates unchanged; local typer applied to TRAIN and DEV','no_dev_tuning':True,'metrics':['FRC80','exact FRD20','S-MSE','FRVar'],'precision':'FP32 training; FP64 Euler16 evaluation -> FP32 metrics'}
 assert protocol['parent_sha256']=='e6199183cdc3526e1fc1390c89057e40514af8d2f2101d69a5e1dabb7f43f4d6';write(OUT/'PROTOCOL.json',protocol);print('prepared',len(cohort),'sources; 500 matched updates',flush=True)
def train(arm,steps=500,diagnostic=False):
 configure_flow();torch.manual_seed(123)
 protocol=json.loads((OUT/'PROTOCOL.json').read_text());assert sha(PARENT)==protocol['parent_sha256'];assert sha(OUT/'schedule.json')==protocol['schedule_sha256']
 base,saved=load_checkpoint(PARENT,'cuda:0');model=SemanticFlow(base,.1).cuda();model.train();opt=torch.optim.AdamW(base.velocity.parameters(),lr=2e-5,weight_decay=.01);opt.load_state_dict(saved['optimizer'])
 for g in opt.param_groups:g['lr']=2e-5
 opt.add_param_group({'params':model.semantic.parameters(),'lr':2e-5,'weight_decay':.01})
 cfg=FlowConfig();data=FlowData(cfg,'cuda:0');cohort=json.loads((OUT/'cohort.json').read_text());schedule=json.loads((OUT/'schedule.json').read_text());weight=saved['calibration']['lambda_ref'];start=time.time()
 dest=OUT/('diagnostics' if diagnostic else 'training')/arm;dest.mkdir(parents=True,exist_ok=True);(dest/'checkpoints').mkdir(exist_ok=True)
 if (dest/'finished.json').exists():return
 if (dest/'training.jsonl').exists():raise RuntimeError('Existing incomplete run; explicit checkpoint resume required')
 def state_hash(state):
  h=hashlib.sha256()
  for name,value in sorted(state.items()):h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
  return h.hexdigest()
 write(dest/'initial.json',{'parent_sha256':sha(PARENT),'base_state_sha256':state_hash(base.state_dict()),'semantic_state_sha256':state_hash(model.semantic.state_dict()),'schedule_sha256':protocol['schedule_sha256'],'cohort_sha256':protocol['cohort_sha256'],'mode':MODES[arm],'steps':steps,'diagnostic':diagnostic})
 with (dest/'training.jsonl').open('x') as log:
  for j,rec in enumerate(schedule[:steps],1):
   tick=time.time();b,y,n,_,_=get_batch(data,rec);events=[]
   for i,clip in enumerate(b['clip_id']):
    item=cohort[clip];es=read(OUT/'train_cache'/(item['clip_id']+'.json'),item['cache']);crop=int(b['crop_start'][i]);length=int(b['source_lengths'][i]);events.append(crop_events(es,item['frame_pts'],crop,crop+length,trajectory_frames=int(b['source_total_length'][i])))
   torch.manual_seed(123+j);opt.zero_grad(set_to_none=True)
   h,sm=model.condition(b['speaker_audio'],b['speaker_emotion'],b['speaker_3dmm'],b['source_lengths'],events,MODES[arm]);valid=torch.arange(cfg.T,device='cuda')[None]<n[:,None]
   target=base.transform(y).detach();noise=draw(target.shape,cfg.noise_seed,rec['step'],'FM_noise').cuda();tau=draw((cfg.B,),cfg.tau_seed,rec['step'],'FM_tau',False).cuda();u,v=flow_path(target,noise,tau,valid)
   estimate=base.velocity(u,tau,h,valid,sm,b['crop_start']);fm=flow_loss(estimate,v,valid)
   z=draw((cfg.B,4,cfg.T,24),cfg.rollout_seed,rec['step'],'task_rollout').cuda();pred=base.rollout(h,sm,z,16,b['crop_start'],gradient_checkpointing=True);task,parts=task_parts(pred,b);loss=fm+weight*task
   loss.backward();assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());opt.step()
   row={'step':j,'global_step':14000+j,'FM':float(fm),'task':float(task),'loss':float(loss),'lambda_ref':weight,'non_null_items':sum(bool(x) for x in events),'semantic_events':sum(len(x or []) for x in events),'seconds':time.time()-tick,'source_indices':rec['source_indices'],'crop_start':b['crop_start'].tolist()};log.write(json.dumps(row)+'\n');log.flush()
   if j==1 or j%25==0:print(arm,row,flush=True)
   write(dest/'monitor_latest.json',row)
   if j in [100,500] or j==steps:
    payload={'format_version':'auto-weak-flow-v1','base_metadata':metadata(base),'base_state':base.state_dict(),'semantic_state':model.semantic.state_dict(),'optimizer':opt.state_dict(),'arm':arm,'step':j,'protocol_sha256':sha(OUT/'PROTOCOL.json'),'parent_sha256':sha(PARENT),'diagnostic':diagnostic}
    p=dest/'checkpoints'/f'step_{j:06d}.pt';tmp=p.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(p);write(p.with_suffix('.json'),{'sha256':sha(p)})
 write(dest/'finished.json',{'arm':arm,'steps':steps,'seconds':time.time()-start,'diagnostic':diagnostic})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--arm',choices=list(MODES));p.add_argument('--steps',type=int,default=500);p.add_argument('--diagnostic',action='store_true');a=p.parse_args()
 if a.prepare:prepare()
 else:train(a.arm,a.steps,a.diagnostic)
