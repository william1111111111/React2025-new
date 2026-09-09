"""Bounded matched B2/B3/B4 pilot with a pretraining gradient-ratio stop gate."""
import argparse
from dataclasses import asdict
import hashlib,json,time
from pathlib import Path
import subprocess
import numpy as np
import torch
from torch.utils.data import default_collate
from . import HiRPNet
from .paired_data import paired_model_inputs
from .session_data import SessionPopulation,make_schedule
from .group_features import reaction_descriptor,DescriptorScaler,fit_descriptor_scaler
from .group_scores import paired_correspondence,b2_score,b3_score,b4_score,toy_hierarchy
from .phase1_diagnostics import prior_mode,gradient_norm
from .phase15_audit import sha256_file,canonical_hash,feature_paths
from .train_phase1 import to_device,state_hash


def description(pred,lengths,scaler):
    if pred.ndim==4:lengths=lengths[:,None].expand(pred.shape[:2])
    return scaler(*reaction_descriptor(pred,lengths))


def losses_for_batch(model,batch,noise,scaler,arm,references=None):
    # No reference or paired tensor enters either forward call.
    with prior_mode(model,'A0'):
        central=model(**paired_model_inputs(batch),sample_count=1,noise=torch.zeros_like(noise[:,:1]))[:,0]
        predictions=model(**paired_model_inputs(batch),sample_count=noise.shape[1],noise=noise)
    pair=paired_correspondence(central,batch['paired_target'],batch['pair_lengths'])
    generated,valid=description(predictions,batch['source_lengths'],scaler)
    if arm=='B2':
        if references is not None:raise ValueError('B2 must not receive population references')
        paired,pair_valid=description(batch['paired_target'],batch['pair_lengths'],scaler)
        dist=b2_score(generated,valid,paired,pair_valid)
    elif arm in ('B3','B4'):
        if references is None:raise ValueError('session population required')
        dist=(b3_score if arm=='B3' else b4_score)(generated,valid,*references)
    else:raise ValueError('unknown Phase 2 arm')
    return pair,dist


def norm_breakdown(model):
    return dict(total=gradient_norm(model),stems=gradient_norm(model.stems),encoder=gradient_norm(model.encoder),
                decoder=gradient_norm(model.decoder),output_head=gradient_norm(model.output_head),prior=gradient_norm(model.prior))


def seeded_model(device):
    torch.manual_seed(123);torch.cuda.manual_seed_all(123)
    return HiRPNet().to(device).train()


def prepare_step(record,source_cache,ref_cache,arm,device):
    batch=to_device(default_collate([source_cache[i] for i in record['source_indices']]),device)
    refs=None
    if arm!='B2':
        selected=[ref_cache[ref] for ref in record['reference_ids']]
        refs=(torch.stack([r[0] for r in selected]).to(device),torch.stack([r[1] for r in selected]).to(device))
    return batch,refs


def preflight(arm,record,noise,scaler,source_cache,ref_cache,device):
    model=seeded_model(device)
    batch,refs=prepare_step(record,source_cache,ref_cache,arm,device)
    pair,dist=losses_for_batch(model,batch,noise.to(device),scaler,arm,refs)
    model.zero_grad(set_to_none=True);pair['overall'].backward(retain_graph=True)
    pair_finite=all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
    pn=norm_breakdown(model)
    model.zero_grad(set_to_none=True);dist['loss'].backward()
    dist_finite=all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
    dn=norm_breakdown(model)
    ratio=max(pn['total']/dn['total'],dn['total']/pn['total']) if min(pn['total'],dn['total'])>0 else None
    passed=bool(pair_finite and dist_finite and ratio is not None and ratio<=100)
    result=dict(arm=arm,pair_loss=pair['overall'].item(),dist_loss=dist['loss'].item(),
                pair_gradient_norms=pn,dist_gradient_norms=dn,gradient_ratio=ratio,
                finite_gradients=pair_finite and dist_finite,gate_passed=passed,initialization_hash=state_hash(model))
    del model
    if device.startswith('cuda'):torch.cuda.empty_cache()
    return result


def train_arm(arm,records,noises,scaler,source_cache,ref_cache,device,output,manifest_sha,scaler_sha):
    model=seeded_model(device);initial=state_hash(model)
    prior_before={k:v.detach().clone() for k,v in model.prior.state_dict().items()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    sessions=[];sources=[];used_refs=[];noise_hash=hashlib.sha256();rows=[]
    if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize(device)
    started=time.perf_counter()
    for record,noise in zip(records,noises):
        sessions.append(record['session_id']);sources.append(record['source_indices'])
        noise_hash.update(noise.numpy().tobytes())
        if arm!='B2':used_refs.append(record['reference_ids'])
        batch,refs=prepare_step(record,source_cache,ref_cache,arm,device)
        optimizer.zero_grad(set_to_none=True)
        pair,dist=losses_for_batch(model,batch,noise.to(device),scaler,arm,refs)
        total=pair['overall']+dist['loss']
        total.backward()
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise RuntimeError('nonfinite training gradient')
        row=dict(step=record['step'],session_id=record['session_id'],loss=total.item(),
                 pair_loss=pair['overall'].item(),pair_AU=pair['AU'].item(),pair_VA=pair['VA'].item(),pair_expression=pair['expression'].item(),
                 dist_loss=dist['loss'].item(),dist_cross=dist['cross'].item(),dist_self=dist['self'].item(),
                 gradient_norm=gradient_norm(model),prior_gradient_norm=gradient_norm(model.prior))
        optimizer.step();rows.append(row)
        if record['step']%16==0:print(json.dumps(dict(arm=arm,**row)),flush=True)
    if device.startswith('cuda'):torch.cuda.synchronize(device)
    elapsed=time.perf_counter()-started
    memory=torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else None
    assert all(torch.equal(prior_before[k],v) for k,v in model.prior.state_dict().items())
    checkpoint=output/'checkpoints'/f'{arm}.pt'
    torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},config=asdict(model.config),
                    arm=arm,prior_mode='A0',steps=len(records),training_manifest_sha256=manifest_sha,scaler_sha256=scaler_sha),checkpoint)
    (output/f'{arm}_training.jsonl').write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in rows))
    result=dict(initialization_hash=initial,parameter_count=sum(p.numel() for p in model.parameters()),steps=len(rows),
                session_schedule_hash=canonical_hash(sessions),source_occurrence_hash=canonical_hash(sources),noise_hash=noise_hash.hexdigest(),
                reference_selection_hash=canonical_hash(used_refs) if used_refs else None,
                population_reference_lookups=sum(map(len,used_refs)),descriptor_scaler_hash=scaler_sha,
                optimizer=dict(name='AdamW',lr=3e-4,weight_decay=.01),prior_mode='A0',prior_parameters_unchanged=True,
                training_seconds=elapsed,peak_allocated_bytes=memory,checkpoint_sha256=sha256_file(checkpoint),first=rows[0],last=rows[-1])
    del model,optimizer
    if device.startswith('cuda'):torch.cuda.empty_cache()
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('runs/phase2'));parser.add_argument('--device',default='cuda:1')
    args=parser.parse_args();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    root=Path(__file__).resolve().parents[1];output=args.output
    output.mkdir(parents=True,exist_ok=False);(output/'checkpoints').mkdir()
    train=SessionPopulation(root/'data','train');val=SessionPopulation(root/'data','val')
    phase15=json.loads((root/'runs/phase15/manifest.json').read_text())
    val_indices=phase15['val_indices']
    assert [val.dataset[i]['clip_id'] for i in val_indices]==phase15['validation']['clip_id']
    records,noises,hashes=make_schedule(train)
    (output/'schedule.json').write_text(json.dumps(dict(records=records,hashes=hashes),indent=2))
    np.save(output/'training_noise.npy',noises.numpy())
    scaler_path=output/'descriptor_stats.json';scaler=fit_descriptor_scaler(train,scaler_path)
    scaler_hash=sha256_file(scaler_path)
    # Population descriptor preparation is shared preprocessing. B2's training
    # path receives no refs and performs zero population-reference lookups.
    ref_ids=sorted({ref for record in records for ref in record['reference_ids']})
    ref_cache={};ref_crops=[]
    with torch.no_grad():
        for ref in ref_ids:
            item=train.load_reference(ref);ref_cache[ref]=scaler(*reaction_descriptor(item['reaction'],item['length']))
            ref_crops.append(dict(record_id=ref,crop_start=item['crop_start'],length=int(item['length'])))
    source_ids=sorted({i for record in records for i in record['source_indices']})
    source_cache={i:train.dataset[i] for i in source_ids}
    selected_paths=set()
    for pool,indices in [(train,source_ids),(val,val_indices)]:
        for index in indices:selected_paths.update(feature_paths(pool.dataset,index))
    selected_paths.update(train.reference_paths.values());selected_paths.update(val.reference_paths.values())
    data_files=[dict(path=str(p.relative_to(root)),sha256=sha256_file(p)) for p in sorted(selected_paths)]
    manifest=dict(phase=2,base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                  branch=subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip(),
                  config=dict(T=128,K_train=4,B=4,R_max=8,steps=128,lambda_dist=1.,prior_mode='A0',
                              optimizer='AdamW',lr=3e-4,weight_decay=.01),
                  device=args.device,torch_version=torch.__version__,
                  train_pool=train.audit(),validation_pool=val.audit(),validation_indices=val_indices,
                  validation_ids=phase15['validation']['clip_id'],validation_ids_hash=canonical_hash(phase15['validation']['clip_id']),
                  source_crops={str(i):dict(clip_id=v['clip_id'],crop_start=int(v['crop_start']),source_lengths=int(v['source_lengths']),pair_lengths=int(v['pair_lengths'])) for i,v in source_cache.items()},
                  reference_crops=ref_crops,schedule_hashes=hashes,scaler_hash=scaler_hash,
                  train_scaler_source_hash=json.loads(scaler_path.read_text())['source_data_hash'],
                  data_files=data_files,data_file_list_hash=canonical_hash(data_files),
                  normalization=[dict(path=f'external/FaceVerse/{n}',sha256=sha256_file(root/'external/FaceVerse'/n)) for n in ('mean_face.npy','std_face.npy')],
                  seeds=dict(initialization=123,session=2001,source=2002,reference=2003,noise=789),
                  source_sha256={str(p.relative_to(root)):sha256_file(p) for p in sorted((root/'hirp').rglob('*.py'))})
    (output/'training_manifest.json').write_text(json.dumps(manifest,indent=2))
    manifest_hash=sha256_file(output/'training_manifest.json')
    (output/'toy_hierarchy.json').write_text(json.dumps(toy_hierarchy(),indent=2))
    scaler=scaler.to(args.device)
    audits={arm:preflight(arm,records[0],noises[0],scaler,source_cache,ref_cache,args.device) for arm in ('B2','B3','B4')}
    (output/'gradient_preflight.json').write_text(json.dumps(audits,indent=2,allow_nan=False))
    print('GRADIENT_PREFLIGHT',json.dumps(audits),flush=True)
    if not all(row['gate_passed'] for row in audits.values()):
        (output/'training_summary.json').write_text(json.dumps(dict(stopped_before_training=True,reason='gradient ratio >100, zero or nonfinite; lambda unchanged'),indent=2))
        print('STOPPED: gradient gate failed; no arm trained',flush=True);return
    results={arm:train_arm(arm,records,noises,scaler,source_cache,ref_cache,args.device,output,manifest_hash,scaler_hash) for arm in ('B2','B3','B4')}
    for key in ('initialization_hash','parameter_count','steps','session_schedule_hash','source_occurrence_hash','noise_hash','descriptor_scaler_hash'):
        assert len({row[key] for row in results.values()})==1,key
    assert results['B3']['reference_selection_hash']==results['B4']['reference_selection_hash']==hashes['reference_selection_hash']
    assert results['B2']['population_reference_lookups']==0
    (output/'training_summary.json').write_text(json.dumps(dict(stopped_before_training=False,matched_controls_verified=True,arms=results),indent=2))
    print('PHASE2_TRAINING_COMPLETE',flush=True)


if __name__=='__main__':main()
