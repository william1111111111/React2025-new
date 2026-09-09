"""Matched short three-arm audit training; no new training loss or regularizer."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from . import HiRPNet
from .paired_data import PairedReactionDataset, paired_model_inputs, pair_valid_mask
from .phase1_diagnostics import prior_mode, gradient_norm, prior_statistics
from .phase15_audit import (audit_random_crops, canonical_hash, data_provenance,
                            sha256_file, stratified_indices)
from .train_phase1 import load_subset, serializable_batch, state_hash, take_batch, to_device
from .losses import paired_energy_score


def train_arm(arm, train_cpu, device, output, manifest_hash):
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    model = HiRPNet().to(device)
    init = state_hash(model)
    weights_before = {name:p.detach().clone() for name,p in model.prior.named_parameters() if name.endswith('weight')}
    optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    batch_all = to_device(train_cpu,device)
    schedule_rng, noise_rng = torch.Generator().manual_seed(456),torch.Generator().manual_seed(789)
    schedule_hash, noise_hash = hashlib.sha256(),hashlib.sha256()
    records=[]
    for step in range(1,65):
        indices=torch.randperm(len(train_cpu['clip_id']),generator=schedule_rng)[:4]
        noise=torch.randn(4,4,32,generator=noise_rng)
        schedule_hash.update(indices.numpy().tobytes()); noise_hash.update(noise.numpy().tobytes())
        batch=take_batch(batch_all,indices.to(device))
        optimizer.zero_grad(set_to_none=True)
        model.train()
        with prior_mode(model,arm):
            aux=model(**paired_model_inputs(batch),noise=noise.to(device),return_aux=True)
        details=paired_energy_score(aux['predictions'],batch['paired_target'],pair_valid_mask(batch),return_details=True)
        details['loss'].backward()
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise RuntimeError('nonfinite gradient')
        row=dict(step=step, **{k:v.item() for k,v in details.items()},
                 prior_mu_gradient_norm=gradient_norm(model.prior.mu),
                 prior_log_sigma_gradient_norm=gradient_norm(model.prior.log_sigma),
                 prior_weight_gradient_norm=sum(p.grad.float().norm().item() for name,p in model.prior.named_parameters()
                                                if name.endswith('weight') and p.grad is not None),
                 **prior_statistics(aux['latent_mu'].detach(),aux['latent_log_sigma'].detach()))
        if arm != 'A0' and (row['prior_mu_gradient_norm']<=0 or row['prior_log_sigma_gradient_norm']<=0):
            raise RuntimeError('prior bias path died')
        if arm == 'A_global' and row['prior_weight_gradient_norm'] != 0:
            raise RuntimeError('global prior weights must not learn')
        optimizer.step()
        records.append(row)
        if step%16==0:
            print(json.dumps(dict(arm=arm,**row)),flush=True)
    unchanged=all(torch.equal(weights_before[name],p) for name,p in model.prior.named_parameters() if name in weights_before)
    if arm in ('A0','A_global') and not unchanged:
        raise RuntimeError('unused prior weights changed, possibly from weight decay')
    checkpoint=output/'checkpoints'/f'{arm}.pt'
    model_state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    torch.save(dict(model=model_state,config=asdict(model.config),arm=arm,step=64,
                    initial_state_sha256=init,training_manifest_sha256=manifest_hash),checkpoint)
    (output/f'{arm}_training.jsonl').write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in records))
    result=dict(initial_state_sha256=init,final_state_sha256=state_hash(model),
                parameter_count=sum(p.numel() for p in model.parameters()),steps=64,
                schedule_sha256=schedule_hash.hexdigest(),training_noise_sha256=noise_hash.hexdigest(),
                prior_weights_unchanged=unchanged,checkpoint=str(checkpoint.name),
                checkpoint_sha256=sha256_file(checkpoint),first=records[0],last=records[-1])
    del optimizer,model
    if str(device).startswith('cuda'): torch.cuda.empty_cache()
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:1')
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    root=Path(__file__).resolve().parents[1]
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    if head != 'e03870fe01c8afdd0f633b8ee90acfb6da13c49b':
        print('NOTE: running on a later source commit; manifest records it',flush=True)
    train_ds=PairedReactionDataset(root/'data','train')
    val_ds=PairedReactionDataset(root/'data','val')
    train_indices=torch.randperm(len(train_ds),generator=torch.Generator().manual_seed(2026))[:32].tolist()
    val_indices=stratified_indices(val_ds,4,1501)
    if not 64<=len(val_indices)<=128:
        raise RuntimeError('validation selection must contain 64-128 records')
    audits=[audit_random_crops(dataset) for dataset in (train_ds,val_ds)]
    print('RANDOM_CROP_PREFLIGHT',json.dumps(audits),flush=True)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'checkpoints').mkdir()
    (args.output/'random_crop_audit.json').write_text(json.dumps(audits,indent=2))
    train,val=load_subset(root,'train',train_indices),load_subset(root,'val',val_indices)
    manifest=dict(phase='1.5',git_head=head,device=args.device,T=128,K_train=4,steps=64,batch_size=4,
                  optimizer=dict(name='AdamW',lr=3e-4,weight_decay=.01),
                  seeds=dict(initialization=123,selection_train=2026,selection_validation=1501,
                             training_schedule=456,training_noise=789,evaluation_noise=321,shuffle=1502),
                  train_indices=train_indices,val_indices=val_indices,
                  train=serializable_batch(train),validation=serializable_batch(val),
                  random_crop_audit=audits,torch_version=torch.__version__,
                  source_sha256={str(p.relative_to(root)):sha256_file(p) for p in sorted((root/'hirp').rglob('*.py'))},
                  **data_provenance(root,[(train_ds,train_indices),(val_ds,val_indices)]))
    manifest['selection_sha256']=canonical_hash(dict(train=manifest['train']['clip_id'],validation=manifest['validation']['clip_id']))
    immutable=args.output/'training_manifest.json'
    immutable.write_text(json.dumps(manifest,indent=2,allow_nan=False))
    manifest_hash=sha256_file(immutable)
    results={arm:train_arm(arm,train,args.device,args.output,manifest_hash) for arm in ('A0','A_global','A1')}
    for key in ('initial_state_sha256','parameter_count','schedule_sha256','training_noise_sha256'):
        assert len({r[key] for r in results.values()})==1,key
    (args.output/'training_summary.json').write_text(json.dumps(dict(matched_controls_verified=True,arms=results),indent=2))
    manifest.update(training_manifest_sha256=manifest_hash,
                    checkpoints={arm:dict(path=f'checkpoints/{arm}.pt',sha256=r['checkpoint_sha256']) for arm,r in results.items()})
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('TRAINING_COMPLETE',str(args.output),flush=True)


if __name__=='__main__': main()
