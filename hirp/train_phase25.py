"""Phase25 T750 occurrence crops; frozen objectives and exact resume state."""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch.utils.data import default_collate
from . import HiRPConfig
from .phase22 import make_model, model_metadata, load_checkpoint, losses_for_batch
from .phase21 import CHANNEL_SCALE
from .phase21_diagnostics import component_gradients, output_path_diagnostic
from .train_phase21 import configure, sync, source_hashes
from .train_phase1 import to_device, state_hash
from .phase15_audit import canonical_hash, sha256_file
from .session_data import SessionPopulation
from .group_features import DescriptorScaler, reaction_descriptor

ROOT=Path(__file__).resolve().parents[1]


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


@dataclass(frozen=True)
class TrainConfig:
    init_seed:int=123
    sampler_seed:int=123
    noise_seed:int=123
    max_steps:int=6000
    lr:float=1e-4
    lambda_group:float=.1
    weight_decay:float=.01
    T:int=750
    K:int=4
    B:int=4
    eval_interval:int=500
    checkpoint_interval:int=500
    diagnostic_interval:int=100
    fixed_checkpoints:tuple=(128,500,1000,2000,4000,6000)

    def validate(self):
        if self.B!=4:raise ValueError('unchanged C2 estimator requires B=4')
        if min(self.max_steps,self.T,self.eval_interval,self.checkpoint_interval,self.diagnostic_interval)<1 or self.K<2:
            raise ValueError('invalid budget/interval/T/K')
        if self.lr<=0 or self.lambda_group<0:raise ValueError('invalid lr/lambda')


def schedule(pool,cfg,latent_dim):
    cfg.validate()
    # Independent source/reference substreams; no reference draws perturb sources.
    session_rng=torch.Generator().manual_seed(cfg.sampler_seed)
    source_rng=torch.Generator().manual_seed(cfg.sampler_seed+1000003)
    ref_rng=torch.Generator().manual_seed(cfg.sampler_seed+2000003)
    noise_rng=torch.Generator().manual_seed(cfg.noise_seed)
    sessions=sorted(pool.sources)
    weights=torch.tensor([len(pool.sources[s]) for s in sessions],dtype=torch.float64)
    records=[];noises=[]
    for i in range(cfg.max_steps):
        s=sessions[int(torch.multinomial(weights,1,generator=session_rng))]
        ids=pool.sources[s];refs=pool.references[s]
        records.append(dict(step=i+1,session_id=s,
            source_indices=[ids[j] for j in torch.randint(len(ids),(cfg.B,),generator=source_rng).tolist()],
            reference_ids=[refs[j] for j in torch.randperm(len(refs),generator=ref_rng)[:8].tolist()],crop_occurrences=[i*cfg.B+j for j in range(cfg.B)],crop_seed=cfg.sampler_seed,reference_crop_step=i,banks=[[0,1],[2,3]]))
        noises.append(torch.randn(cfg.B,cfg.K,latent_dim,generator=noise_rng))
    noise=torch.stack(noises)
    return records,noise,stream_hashes(records,noise)


def stream_hashes(records,noise):
    return dict(crop_hash=canonical_hash([{k:r[k] for k in ('crop_occurrences','crop_seed','reference_crop_step')} for r in records]),source_hash=canonical_hash([r['source_indices'] for r in records]),
                session_hash=canonical_hash([r['session_id'] for r in records]),
                reference_hash=canonical_hash([r['reference_ids'] for r in records]),
                noise_hash=hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest())


def rng_state():
    n=np.random.get_state()
    return dict(python=random.getstate(),numpy=[n[0],n[1].tolist(),n[2],n[3],n[4]],
                torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(s):
    random.setstate(s['python']);n=s['numpy'];np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(s['torch'].cpu())
    if s['cuda']:
        if len(s['cuda'])!=torch.cuda.device_count():raise ValueError('CUDA RNG topology differs on resume')
        torch.cuda.set_rng_state_all([x.cpu() for x in s['cuda']])


from .data_phase25 import RealData, fit_scaler


def prepare(run,cfg):
    cfg.validate();run.mkdir(parents=True,exist_ok=False)
    pool=SessionPopulation(ROOT/'data','train',cfg.T)
    scaler_path=run.parent/f'descriptor_stats_T{cfg.T}.json'
    if not scaler_path.exists():fit_scaler(pool,scaler_path)
    if json.loads(scaler_path.read_text())['clip_length']!=cfg.T:raise ValueError('scaler T mismatch')
    records,noise,hashes=schedule(pool,cfg,HiRPConfig().latent_dim)
    write(run/'schedule.json',dict(records=records,hashes=hashes))
    np.save(run/'noise.npy',noise.numpy())
    write(run/'config.json',asdict(cfg))
    # Full TRAIN file provenance; new schedules may access any source/ref.
    paths=set()
    for path in pool.dataset.records:
        rel=path.relative_to(pool.dataset.directory/'facial-attributes')
        paths.update((path,pool.dataset.directory/'audio-features'/rel,pool.dataset.directory/'coefficients'/rel))
    paths.update(pool.reference_paths.values())
    file_rows=[dict(path=str(p),sha256=sha256_file(p)) for p in sorted(paths)]
    normalization=[dict(path=str(ROOT/'external/FaceVerse'/n),sha256=sha256_file(ROOT/'external/FaceVerse'/n)) for n in ('mean_face.npy','std_face.npy')]
    manifest=dict(config=asdict(cfg),model_config=asdict(HiRPConfig()),scaler_path=str(scaler_path),
        scales=dict(channel_scale=list(CHANNEL_SCALE),descriptor_scaler_sha256=sha256_file(scaler_path),descriptor_dimension=75,training_T=cfg.T),
        split_hash=canonical_hash(file_rows),data_files=file_rows,normalization=normalization,
        population=pool.audit(),schedule=hashes,noise_file_sha256=sha256_file(run/'noise.npy'),source_sha256={str(p):sha256_file(p) for p in sorted((ROOT/'hirp').rglob('*.py'))},
        output_config=dict(head_pre_norm=True,projection_init='default',small_init_std=.01),prior_mode='standard_normal',
        crop_contract='source-only uniform start per seed/occurrence/ID; reference independent uniform start per step/ID; no paired-overlap resampling; raw-array LRU cache',budget=f'fixed {cfg.max_steps}-step budget; no validation-driven budget extension',
        evaluation='replay saved fixed checkpoints after training, development set only; no optimizer/RNG effect',
        eval_steps=sorted({s for s in cfg.fixed_checkpoints if s<=cfg.max_steps}|set(range(cfg.eval_interval,cfg.max_steps+1,cfg.eval_interval))|{cfg.max_steps}))
    write(run/'manifest.json',manifest)
    return manifest


def train(arm_dir,cfg,arm,records,noises,data,manifest,device,resume=None,stop_after=None,model_config=None):
    cfg.validate()
    if len(records)!=cfg.max_steps or len(noises)!=cfg.max_steps:raise ValueError('schedule length must equal requested steps')
    if arm not in ('C0','C1','C2'):raise ValueError('unknown arm')
    arm_dir.mkdir(parents=True,exist_ok=True)
    attempt=arm_dir/f'attempt_{len(list(arm_dir.glob("attempt_*"))):03d}'
    attempt.mkdir(exist_ok=False);(attempt/'checkpoints').mkdir()
    random.seed(cfg.init_seed);np.random.seed(cfg.init_seed)
    model=make_model(cfg.init_seed,model_config or HiRPConfig(**manifest['model_config']),device)
    initial_hash=state_hash(model);prior={k:v.detach().clone() for k,v in model.prior.state_dict().items()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    rows=[];global_step=0;previous_seconds=0.;checkpoint_paths=[]
    if resume:
        model,saved=load_checkpoint(resume,device)
        old=saved['train_config'];new=asdict(cfg)
        if any(old[k]!=v for k,v in new.items() if k!='max_steps') or cfg.max_steps<old['max_steps']:
            raise ValueError('resume config differs beyond nondecreasing max_steps')
        if saved['arm']!=arm or saved['split_hash']!=manifest['split_hash'] or saved['scales']!=manifest['scales']:
            raise ValueError('resume arm/split/scales mismatch')
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
        optimizer.load_state_dict(saved['optimizer'])
        global_step=saved['global_step'];rows=saved['training_rows'];previous_seconds=saved['training_seconds']
        if global_step!=len(rows) or global_step!=saved['schedule_position']:raise ValueError('resume step count mismatch')
        if stream_hashes(records[:global_step],noises[:global_step])!=saved['consumed_hashes']:raise ValueError('resume schedule prefix mismatch')
        initial_hash=saved['initialization_hash'];restore_rng(saved['rng'])
    # PyTorch 2.1's first AdamW construction lazily imports modules that consume
    # Python random state. Normalize that non-tensor stream after construction.
    # Torch RNG is untouched, preserving model initialization/dropout semantics.
    if not resume:random.seed(cfg.init_seed)
    model.train()
    weight=0. if arm=='C0' else cfg.lambda_group
    start_step=global_step;started=time.perf_counter();compute_seconds=0.
    if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats(device)
    all_sources=[i for r in records[:global_step] for i in r['source_indices']]
    def checkpoint(completed):
        sync(device)
        path=attempt/'checkpoints'/f'step_{global_step:06d}.pt'
        saved=dict(**model_metadata(model,manifest['scales']),model=model.state_dict(),optimizer=optimizer.state_dict(),
                   global_step=global_step,requested_steps=cfg.max_steps,actual_optimizer_steps=global_step,
                   schedule_position=global_step,train_config=asdict(cfg),rng=rng_state(),arm=arm,
                   split_hash=manifest['split_hash'],consumed_hashes=stream_hashes(records[:global_step],noises[:global_step]),
                   training_rows=rows,completed=completed,initialization_hash=initial_hash,
                   training_seconds=previous_seconds+time.perf_counter()-started)
        if path.exists():raise FileExistsError(path)
        torch.save(saved,path);checkpoint_paths.append(dict(step=global_step,path=str(path),sha256=sha256_file(path)))
    failure=None
    try:
        with (attempt/'training.jsonl').open('x') as log:
            for row in rows:log.write(json.dumps(row)+'\n')
            while global_step<cfg.max_steps:
                record=records[global_step];noise=noises[global_step].to(device)
                if record['step']!=global_step+1:raise ValueError('schedule step discontinuity')
                batch,refs=data.batch(record);optimizer.zero_grad(set_to_none=True)
                sync(device);step_start=time.perf_counter()
                values=losses_for_batch(model,batch,noise,data.scaler,arm,refs,weight)
                audit=component_gradients(model,values['conditional']['loss'],values['group']['loss'] if values['group'] else None,weight) if (global_step+1)%cfg.diagnostic_interval==0 or global_step==0 else None
                values['total'].backward()
                if not all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None):raise RuntimeError('nonfinite gradient')
                optimizer.step();sync(device);compute_seconds+=time.perf_counter()-step_start;global_step+=1
                row=dict(crop_start=batch['crop_start'].tolist(),source_lengths=batch['source_lengths'].tolist(),pair_lengths=batch['pair_lengths'].tolist(),step=global_step,loss=float(values['total']),conditional={k:float(v) for k,v in values['conditional'].items()},
                         group={k:float(v) for k,v in values['group'].items()} if values['group'] else None,
                         gradients=audit,source_indices=record['source_indices'],session_id=record['session_id'],
                         reference_ids=record['reference_ids'] if arm!='C0' else [])
                rows.append(row);all_sources.extend(record['source_indices'])
                log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                if global_step%cfg.diagnostic_interval==0:
                    before=state_hash(model);rstate=rng_state()
                    # Frozen-prior public forward has the same semantics under this legacy diagnostic.
                    diag=output_path_diagnostic(model,batch,noise)
                    assert before==state_hash(model)
                    assert torch.equal(rstate['torch'],torch.get_rng_state())
                    assert all(torch.equal(a,b) for a,b in zip(rstate['cuda'],torch.cuda.get_rng_state_all()))
                    write(attempt/f'diagnostic_{global_step:06d}.json',diag)
                    print(arm,global_step,'conditional',row['conditional']['loss'],flush=True)
                should_stop=stop_after is not None and global_step>=stop_after
                if global_step in cfg.fixed_checkpoints or global_step%cfg.checkpoint_interval==0 or global_step%cfg.eval_interval==0 or global_step==cfg.max_steps or should_stop:
                    checkpoint(global_step==cfg.max_steps)
                if should_stop:break
    except BaseException as exc:
        failure=repr(exc)
        write(attempt/'failure.json',dict(completed=False,actual_optimizer_steps=global_step,requested_steps=cfg.max_steps,
                                         training_rows=len(rows),error=failure,last_durable_checkpoints=checkpoint_paths))
        raise
    finally:
        sync(device)
        elapsed=time.perf_counter()-started
        completed=global_step==cfg.max_steps and failure is None
        if completed:assert global_step==cfg.max_steps==len(rows)
        assert all(torch.equal(prior[k],v) for k,v in model.prior.state_dict().items())
        summary=dict(completed=completed,requested_steps=cfg.max_steps,actual_optimizer_steps=global_step,training_rows=len(rows),
            initialization_hash=initial_hash,consumed_hashes=stream_hashes(records[:global_step],noises[:global_step]),
            source_occurrences=len(all_sources),valid_source_frame_exposures=sum(sum(r['source_lengths']) for r in rows),valid_pair_frame_exposures=sum(sum(r['pair_lengths']) for r in rows),tensor_frame_exposures=global_step*cfg.B*cfg.T,unique_sources=len(set(all_sources)),
            source_population=manifest['population']['source_population'],
            exposure_multiple=len(all_sources)/manifest['population']['source_population'],
            reference_exposures=sum(len(r['reference_ids']) for r in rows),
            unique_references=len({x for r in rows for x in r['reference_ids']}),parameter_count=sum(p.numel() for p in model.parameters()),
            prior_unchanged=True,attempt_steps=global_step-start_step,attempt_wall_seconds=elapsed,
            cumulative_wall_seconds=previous_seconds+elapsed,optimizer_compute_seconds=compute_seconds,
            attempt_sources_per_compute_second=(global_step-start_step)*cfg.B/compute_seconds if compute_seconds else None,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else None,
            invalid_pair_count=getattr(data,'invalid_pairs',0),device=device,checkpoints=checkpoint_paths,resumed_from=str(resume) if resume else None)
        write(attempt/'summary.json',summary)
    return summary,model,optimizer


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--arm',choices=['C0','C1','C2','all'],default='all');p.add_argument('--resume',type=Path)
    p.add_argument('--stop-after',type=int)
    for k,v in asdict(TrainConfig()).items():
        if k=='fixed_checkpoints':p.add_argument('--fixed-checkpoints',type=int,nargs='+',default=list(v))
        else:p.add_argument('--'+k.replace('_','-'),type=type(v),default=v)
    a=p.parse_args();configure();cfg=TrainConfig(**{k:tuple(getattr(a,k)) if k=='fixed_checkpoints' else getattr(a,k) for k in asdict(TrainConfig())})
    # A one-time external handoff may already own an arm. Do not start a second
    # optimizer from an older checkpoint when the original sequential driver
    # reaches that arm. The owner entered before this marker was installed.
    handoff=a.run/f'{a.arm}_concurrent_owner.json'
    if handoff.exists():
        import os
        if json.loads((a.run/'manifest.json').read_text())['config']!=json.loads(json.dumps(asdict(cfg))):
            raise ValueError('external handoff run config mismatch')
        owner=json.loads(handoff.read_text())
        finished=Path(owner['finished_path'])
        print('Waiting for external arm owner',owner['pid'],flush=True)
        while not finished.exists():
            try:os.kill(owner['pid'],0)
            except ProcessLookupError:raise RuntimeError('external owner exited without completion record')
            time.sleep(2)
        result=json.loads(finished.read_text())
        if not result.get('completed'):raise RuntimeError('external training failed; inspect its records before explicit recovery')
        summary=json.loads(Path(owner['summary_path']).read_text())
        requested_stop=a.stop_after or cfg.max_steps
        if summary['actual_optimizer_steps']>=requested_stop:
            assert summary['actual_optimizer_steps']==summary['training_rows']
            print('Arm already completed by external owner',summary['actual_optimizer_steps'],flush=True)
            return
        if not a.resume:
            raise RuntimeError('continuation beyond external segment requires explicit resume')
    if (a.run/'manifest.json').exists():
        manifest=json.loads((a.run/'manifest.json').read_text())
        expected=json.loads(json.dumps(asdict(cfg)))
        if manifest['config']!=expected:raise ValueError('existing run config mismatch; budget extension requires a new run manifest')
    else:manifest=prepare(a.run,cfg)
    for row in manifest['data_files']+manifest['normalization']:
        if sha256_file(row['path'])!=row['sha256']:raise ValueError('data provenance mismatch')
    if sha256_file(manifest['scaler_path'])!=manifest['scales']['descriptor_scaler_sha256']:raise ValueError('scaler mismatch')
    records=json.loads((a.run/'schedule.json').read_text())['records'];noises=torch.from_numpy(np.load(a.run/'noise.npy'))
    if stream_hashes(records,noises)!=manifest['schedule']:raise ValueError('schedule/noise hash mismatch')
    if a.resume and a.arm=='all':raise ValueError('resume requires one explicit arm')
    for arm in ('C0','C1','C2') if a.arm=='all' else (a.arm,):
        if (a.run/arm).exists() and not a.resume:raise FileExistsError('arm already exists; explicit resume required')
        data=RealData(manifest,cfg,arm,a.device)
        result,model,opt=train(a.run/arm,cfg,arm,records,noises,data,manifest,a.device,a.resume,a.stop_after)
        del data,model,opt
        if a.device.startswith('cuda'):torch.cuda.empty_cache()
        print('RESULT',arm,json.dumps(result),flush=True)


if __name__=='__main__':main()
