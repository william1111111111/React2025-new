"""Fixed seed123 matched R1/R2, complete schedule, checkpoint resume."""
import argparse,json,time,random,hashlib
from pathlib import Path
import numpy as np
import torch
from mam_target.model import make_model,metadata,load_checkpoint
from mam_target.data import TaskData
from mam_target.losses import objective,pair_costs
from hirp.train_phase25 import TrainConfig,rng_state,restore_rng,stream_hashes
from hirp.train_phase21 import configure
from hirp.phase22 import losses_for_batch
from hirp.paired_data import paired_model_inputs
from hirp.train_phase1 import state_hash
from hirp.phase15_audit import sha256_file
ROOT=Path('runs/mam_target/task_v1')
OLD=Path('runs/phase25/timescale_v1/seed_123')

def write(path,x):
    with Path(path).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)

def resources(arm,device):
    cfg=TrainConfig();manifest=json.loads((OLD/'manifest.json').read_text())
    records=json.loads((OLD/'schedule.json').read_text())['records'];noise=torch.from_numpy(np.load(OLD/'noise.npy'))
    assert len(records)==len(noise)==6000
    assert sha256_file(OLD/'noise.npy')==manifest['noise_file_sha256']
    data=TaskData(manifest,cfg,'C2' if arm=='R1' else 'C0',device)
    return cfg,manifest,records,noise,data

def calibrate(device):
    cfg,manifest,records,noises,data=resources('R2',device);m=make_model(device=device);m.train();samples=[]
    for rec,noise in zip(records[:3],noises[:3]):
        batch,_,targets,lengths,ids=data.task_batch(rec)
        p=m(**paired_model_inputs(batch),sample_count=4,noise=noise.to(device))
        c,d=pair_costs(p,targets,batch['source_lengths'],lengths)
        norms=[]
        for loss in [c.mean(),d.mean()]:
            g=torch.autograd.grad(loss,p,retain_graph=True)[0];norms.append(float(g.norm()))
        samples.append(dict(step=rec['step'],ccc=float(c.mean()),sdtw=float(d.mean()),prediction_gradient_norms=norms,targets=ids))
    ratio=float(np.median([r['prediction_gradient_norms'][0]/max(r['prediction_gradient_norms'][1],1e-12) for r in samples]))
    weights=dict(a=1.,b=ratio,beta=.25,gamma=.25,eta=.5,grid=32)
    write(ROOT/'weights.json',dict(weights=weights,calibration=samples,rule='a=1; b=median TRAIN prediction-gradient norm ratio over first3 schedule batches; beta=.25 gamma=.25 eta=.5 fixed before dev',validation_used=False))
    print('CALIBRATED',weights,flush=True)

def train(arm,device,stop,resume=None,label=None):
    cfg,manifest,records,noises,data=resources(arm,device);weights=json.loads((ROOT/'weights.json').read_text())['weights']
    out=ROOT/(label or arm);out.mkdir(exist_ok=True);attempt=out/f'attempt_{len(list(out.glob("attempt_*"))):03d}';attempt.mkdir();(attempt/'checkpoints').mkdir()
    random.seed(123);np.random.seed(123);m=make_model(device=device);initial=state_hash(m);opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=.01);step=0;rows=[]
    if resume:
        m,s=load_checkpoint(resume,device);assert s['arm']==arm and s['weights']==weights and s['split_hash']==manifest['split_hash']
        assert stream_hashes(records[:s['step']],noises[:s['step']])==s['consumed_hashes']
        opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=.01);opt.load_state_dict(s['optimizer']);step=s['step'];rows=s['rows'];initial=s['initialization_hash'];restore_rng(s['rng'])
    m.train();start=time.time();torch.cuda.reset_peak_memory_stats(device)
    try:
        with (attempt/'training.jsonl').open('x') as log:
            for row in rows:log.write(json.dumps(row)+'\n')
            while step<stop:
                rec=records[step];noise=noises[step].to(device);opt.zero_grad(set_to_none=True)
                diagnostic=(step==0 or (step+1)%100==0)
                noise.requires_grad_(diagnostic);captured={};hooks=[]
                if diagnostic:
                    def capture(name):
                        def hook(module,args,result):
                            value=result[0] if isinstance(result,tuple) else result
                            value.retain_grad();captured[name]=value
                        return hook
                    for name,mod in [('H',m.encoder),('decoded',m.decoder),('raw',m.output_head.raw_probe)]:hooks.append(mod.register_forward_hook(capture(name)))
                if arm=='R1':
                    batch,refs=data.batch(rec);v=losses_for_batch(m,batch,noise,data.scaler,'C2',refs,.1);loss=v['total'];metrics=dict(preserve=float(v['conditional']['loss']),group=float(v['group']['loss']));ids=rec['reference_ids']
                else:
                    batch,_,targets,lengths,ids=data.task_batch(rec);p=m(**paired_model_inputs(batch),sample_count=4,noise=noise)
                    v=objective(p,batch,targets,lengths,weights);loss=v['loss'];metrics={k:float(x) for k,x in v.items() if k!='loss'}
                loss.backward()
                if diagnostic:
                    pred=v['predictions'] if arm=='R1' else p
                    details={}
                    for name,x in captured.items():
                        mask=torch.arange(x.shape[-2],device=device)[None]<batch['source_lengths'][:,None]
                        if x.ndim==4:mask=mask[:,None].expand(-1,x.shape[1],-1)
                        vals=x.detach()[mask].flatten()
                        details[name]=dict(abs_quantiles=torch.quantile(vals.abs(),torch.tensor([.5,.9,.99],device=device)).tolist(),std=float(vals.std()),gradient_norm=float(x.grad.norm()))
                    details['noise_loss_vjp']=float(noise.grad.norm())
                    valid=torch.arange(pred.shape[2],device=device)[None]<batch['source_lengths'][:,None]
                    details['groups']={name:dict(sample_std=float(pred.detach()[...,a:b].std(1)[valid].mean()),mean=float(pred.detach()[...,a:b].permute(0,2,1,3)[valid].mean())) for name,a,b in [('AU',0,15),('VA',15,17),('expression',17,25)]}
                    write(attempt/f'diagnostic_{step+1:06d}.json',details)
                    for h in hooks:h.remove()
                if not torch.isfinite(loss) or any(not torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None):raise RuntimeError('nonfinite loss/gradient')
                norms={name:float(sum(p.grad.square().sum() for p in mod.parameters() if p.grad is not None).sqrt()) for name,mod in [('encoder',m.encoder),('decoder',m.decoder),('head',m.output_head)]}
                opt.step();step+=1
                row=dict(step=step,loss=float(loss),**metrics,gradients=norms,source_indices=rec['source_indices'],targets=ids,crop_start=batch['crop_start'].tolist(),source_lengths=batch['source_lengths'].tolist(),pair_lengths=batch['pair_lengths'].tolist())
                rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
                if step%50==0:print(arm,step,float(loss),time.time()-start,flush=True)
                if step in (128,500,1000,2000,4000,6000,stop):
                    saved=dict(**metadata(m),model=m.state_dict(),optimizer=opt.state_dict(),step=step,arm=arm,seed=123,weights=weights,rows=rows,rng=rng_state(),initialization_hash=initial,split_hash=manifest['split_hash'],scales=manifest['scales'],consumed_hashes=stream_hashes(records[:step],noises[:step]),requested_steps=6000,completed=step==6000)
                    path=attempt/'checkpoints'/f'step_{step:06d}.pt';tmp=path.with_suffix('.tmp');torch.save(saved,tmp);tmp.rename(path)
                    write(path.with_suffix('.json'),dict(path=str(path),sha256=sha256_file(path),step=step,arm=arm))
        assert step==stop==len(rows)
        torch.cuda.synchronize(device)
        write(attempt/'status.json',dict(actual_steps=step,requested_steps=6000,completed=step==6000,requested_segment_stop=stop,training_rows=len(rows),seconds=time.time()-start,peak_memory_bytes=torch.cuda.max_memory_allocated(device),initialization_hash=initial,parameters=sum(p.numel() for p in m.parameters()),valid_source_frames=sum(sum(r['source_lengths']) for r in rows),device=device))
    except BaseException as e:
        write(attempt/'failure.json',dict(step=step,error=repr(e),completed=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['R1','R2']);p.add_argument('--device',default='cuda:0');p.add_argument('--stop',type=int,default=6000);p.add_argument('--resume');p.add_argument('--label');p.add_argument('--calibrate',action='store_true');a=p.parse_args();configure();torch.use_deterministic_algorithms(True)
    if a.calibrate:calibrate(a.device)
    else:train(a.arm,a.device,a.stop,a.resume,a.label)
