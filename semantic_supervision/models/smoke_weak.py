"""Real TRAIN B=2 integration diagnostic, not a task-performance experiment."""
import json,hashlib,subprocess,copy
from pathlib import Path
from dataclasses import replace
import numpy as np,torch
from reaction_flow.sampler import load_checkpoint
from reaction_flow.flow_path import flow_path,flow_loss
from .auto_weak import read_weak,crop_events
from .semantic_flow import SemanticFlow
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/reaction_flow/semantic_auto_weak_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    torch.manual_seed(123);torch.set_num_threads(4);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    idx=json.loads((OUT/'cache/index.json').read_text());job=json.loads((ROOT/'runs/reaction_flow/semantic_alignment_pilot_v1/job/job.json').read_text());mapping={r['id']:r for r in job['records']};samples=[];all_loaded=0
    for cid,m in sorted(idx.items()):
        es=read_weak(OUT/'cache'/(cid+'.json'),split='train',expected_cache_sha256=m['cache_sha256'],expected_media_hashes=m['media_hashes'],expected_policy_sha256=m['policy_sha256']);all_loaded+=len(es or [])
        if not es or len(samples)>=2:continue
        timed=[e for e in es if e['sync_status']['status']=='estimated' and e['timing_evidence']['start_s']>=e['sync_status']['effective_audio_range_s'][0] and e['timing_evidence']['end_s']<=e['sync_status']['effective_audio_range_s'][1]]
        if not timed:continue
        rel=mapping[cid]['relative'];data=ROOT/'data/train';v=data/'video-face-crop/speaker'/(rel+'.mp4')
        current=[sha(data/'audio/speaker'/(rel+'.wav')),sha(v),sha(data/'text/speaker'/(rel+'.txt'))];assert sorted(current)==sorted(m['media_hashes'])
        frames=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','json',str(v)]))['frames'];pts=np.array([float(f['best_effort_timestamp_time']) for f in frames])
        arrays=[np.load(data/name/'speaker'/(rel+'.npy'),allow_pickle=False).astype(np.float32) for name in ['audio-features','facial-attributes','coefficients']];arrays[2]=arrays[2].reshape(-1,58)
        assert all(len(a)==len(pts) for a in arrays)
        mean=np.load(ROOT/'external/FaceVerse/mean_face.npy');std=np.maximum(np.load(ROOT/'external/FaceVerse/std_face.npy'),1e-8);arrays[2]=((arrays[2]-mean)/std).astype(np.float32)
        offset=timed[0]['sync_status']['offset_s'];start=max(0,int(np.searchsorted(pts,timed[0]['timing_evidence']['start_s']+offset))-8);start=min(start,len(pts)-64)
        events=crop_events(es,pts,start,start+64,trajectory_frames=len(pts));assert events and any(e.interval is not None for e in events)
        # The paired target is read only here for the unchanged FM loss; never passed to cache/semantic APIs.
        target=np.load(data/'facial-attributes/listener'/(rel+'.npy'),allow_pickle=False).astype(np.float32);assert len(target)>=start+64
        samples.append({'clip_id':cid,'start':start,'arrays':[torch.from_numpy(a[start:start+64].copy()) for a in arrays],'target':torch.from_numpy(target[start:start+64].copy()),'events':events,'pts_sha256':hashlib.sha256(pts.tobytes()).hexdigest()})
    assert len(samples)==2
    checkpoint=ROOT/'runs/reaction_flow/task_dynamics_v1/T0-task/attempt_000/checkpoints/step_014000.pt';checkpoint_sha=sha(checkpoint)
    assert checkpoint_sha=='e6199183cdc3526e1fc1390c89057e40514af8d2f2101d69a5e1dabb7f43f4d6'
    base,_=load_checkpoint(checkpoint,'cuda:0');base.requires_grad_(False);model=SemanticFlow(base,dropout=.1).cuda();model.eval()
    arrays=[torch.stack([s['arrays'][i] for s in samples]).cuda() for i in range(3)];lengths=torch.tensor([64,64],device='cuda');events=[s['events'] for s in samples];offsets=torch.tensor([s['start'] for s in samples],device='cuda');target=torch.stack([s['target'] for s in samples]).cuda();valid=torch.ones(2,64,dtype=torch.bool,device='cuda')
    z=torch.randn(2,64,24,device='cuda');tau=torch.tensor([.3,.7],device='cuda');u,v=flow_path(base.transform(target),z,tau,valid)
    pred=model.velocity(u,tau,*arrays,lengths,events,position_offset=offsets);loss=flow_loss(pred,v,valid);loss.backward()
    norms={n:float(p.grad.norm()) if p.grad is not None else 0 for n,p in model.semantic.named_parameters()};assert all(x>0 and np.isfinite(x) for n,x in norms.items())
    assert not any(p.grad is not None for p in base.condition.parameters())
    shifted=[[replace(e,interval=(e.interval[0]+.4,e.interval[1]+.4)) if e.interval else e for e in es] for es in events]
    changed=[[replace(e,text='A different source sentence.',event_type='unknown') for e in es] for es in events]
    with torch.no_grad():
        sensitivity=float((pred-model.velocity(u,tau,*arrays,lengths,changed,position_offset=offsets)).abs().max())
        time_sensitivity=float((pred-model.velocity(u,tau,*arrays,lengths,shifted,position_offset=offsets)).abs().max())
    assert sensitivity>1e-7 and time_sensitivity>1e-7
    optim=torch.optim.AdamW(model.semantic.parameters(),lr=1e-4);losses=[]
    # Fixed mini-batch, three branch-only updates; not resumed T0 training.
    before={n:p.detach().clone() for n,p in model.semantic.named_parameters()}
    model.train()
    for _ in range(3):
        optim.zero_grad();p=model.velocity(u,tau,*arrays,lengths,events,position_offset=offsets);l=flow_loss(p,v,valid);l.backward();optim.step();losses.append(float(l))
    model.eval();noise=torch.randn(2,3,64,24,device='cuda')
    a=model.sample(*arrays,lengths,events,noise=noise,position_offset=offsets);b=model.sample(*arrays,lengths,events,noise=noise,position_offset=offsets);prefix=model.sample(*arrays,lengths,events,noise=noise[:,:2].contiguous(),position_offset=offsets)
    repeat=float((a-b).abs().max());prefix_error=float((a[:,:2]-prefix).abs().max());assert repeat==0 and prefix_error<1e-5
    result={'parent_sha256':checkpoint_sha,'batch_size':2,'crop_frames':64,'loaded_events_total':all_loaded,'mini_batch_events':[[{'event_id':e.event_id,'text':e.text,'interval':e.interval,'frame_interval':e.frame_interval} for e in es] for es in events],'source_crops':[{'clip_id':s['clip_id'],'start_frame':s['start'],'pts_sha256':s['pts_sha256']} for s in samples],'gradient_norms':norms,'frozen_condition_gradients':False,'content_type_perturbation_max':sensitivity,'time_perturbation_max':time_sensitivity,'same_input_noise_max':repeat,'k_prefix_max_error':prefix_error,'branch_optimizer_steps':3,'losses_diagnostic_only':losses,'semantic_parameters_changed':all(not torch.equal(before[n],p) for n,p in model.semantic.named_parameters()),'official_metrics_run':False,'long_training_started':False,'human_reviewed':False}
    (OUT/'mini_batch_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['mini_batch_events','gradient_norms']},ensure_ascii=False))
if __name__=='__main__':main()
