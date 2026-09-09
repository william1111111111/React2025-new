"""Read-only audit-checkpoint evaluation with common random numbers, no fitting."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from . import HiRPNet,HiRPConfig
from .paired_data import paired_model_inputs
from .phase1_diagnostics import prior_mode,prior_statistics
from .phase15_audit import sha256_file,verify_provenance
from .phase15_interventions import within_session_permutation,intervention_prior,override_prior
from .phase15_metrics import channel_metrics,aggregate_channels,real_real_reference,distribution
from .train_phase1 import load_subset,take_batch,to_device,state_hash


@torch.no_grad()
def collect_priors(model,batch,arm,device,batch_size=4):
    mus,logs=[],[]
    for start in range(0,len(batch['clip_id']),batch_size):
        indices=torch.arange(start,min(start+batch_size,len(batch['clip_id'])))
        small=to_device(take_batch(batch,indices),device)
        with prior_mode(model,arm):
            aux=model.sample(**paired_model_inputs(small),sample_count=1,
                             noise=torch.zeros(len(indices),1,model.config.latent_dim,device=device),return_aux=True)
        mus.append(aux['latent_mu'].cpu()); logs.append(aux['latent_log_sigma'].cpu())
    return torch.cat(mus),torch.cat(logs)


@torch.no_grad()
def evaluate_case(model,batch,mu,logs,noise_bank,alpha,device):
    if noise_bank.shape[1]<2:
        raise ValueError('evaluation requires K >= 2')
    records=[]; raw_summaries=[]
    for start in range(0,len(batch['clip_id']),4):
        indices=torch.arange(start,min(start+4,len(batch['clip_id'])))
        small=to_device(take_batch(batch,indices),device)
        noise=noise_bank.to(device).expand(len(indices),-1,-1)*alpha
        captured=[]
        handle=model.output_head.stochastic.register_forward_hook(lambda module,inputs,output:captured.append(output.detach()))
        try:
            with override_prior(model,mu[indices].to(device),logs[indices].to(device)):
                aux=model.sample(**paired_model_inputs(small),sample_count=noise.shape[1],noise=noise,return_aux=True)
        finally:
            handle.remove()
        metrics=channel_metrics(aux['predictions'],small['paired_target'],small['pair_lengths'])
        for j,index in enumerate(indices.tolist()):
            valid=aux['valid_mask'][j]
            raw=captured[0][j,:,valid].float()
            residual=dict(stochastic_raw_abs_mean=raw.abs().mean().item(),
                          stochastic_residual_abs_mean=raw.tanh().abs().mean().item(),
                          tanh_saturation_fraction=(raw.tanh().abs()>=.99).float().mean().item())
            raw_summaries.append(residual)
            records.append(dict(clip_id=batch['clip_id'][index],session_id=batch['session_id'][index],
                                channels=metrics[j],residual=residual))
    aggregate=aggregate_channels([r['channels'] for r in records])
    prior=prior_statistics(mu,logs)
    prior['effective_sigma_mean']=logs.exp().mean().item()*alpha
    return dict(channels=aggregate,prior=prior,
                residual={key:float(np.mean([r[key] for r in raw_summaries])) for key in raw_summaries[0]},
                spread_distributions={name:distribution([r['channels'][name]['self_distance'] for r in records])
                                      for name in aggregate},per_example=records)


def paired_difference(reference,other):
    """Other minus correct, paired by identical inputs and random numbers."""
    return {name:{key:distribution([b['channels'][name][key]-a['channels'][name][key]
                                  for a,b in zip(reference['per_example'],other['per_example'])])
                  for key in ('ES','cross_distance','self_distance')}
            for name in reference['channels']}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--output',type=Path,help='new evaluation output directory; never overwrite')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    manifest=json.loads((args.run/'manifest.json').read_text())
    if sha256_file(args.run/'training_manifest.json')!=manifest['training_manifest_sha256']:
        raise ValueError('training manifest hash mismatch')
    verify_provenance(root,manifest)
    output=args.output if args.output is not None else args.run/'evaluation'
    output.mkdir(parents=True,exist_ok=False)
    batch=load_subset(root,'val',manifest['val_indices'])
    assert batch['clip_id']==manifest['validation']['clip_id']
    bank=torch.randn(1,64,32,generator=torch.Generator().manual_seed(321))
    np.save(output/'common_noise_K64.npy',bank.numpy())
    permutation=within_session_permutation(batch['session_id'],1502)
    shuffle=[dict(recipient=batch['clip_id'][i],donor=batch['clip_id'][j]) for i,j in enumerate(permutation.tolist())]
    real=real_real_reference(batch)
    (output/'real_real_reference.json').write_text(json.dumps(real,indent=2,allow_nan=False))
    all_results={}; hashes_after={}
    for arm in ('A0','A_global','A1'):
        checkpoint=args.run/manifest['checkpoints'][arm]['path']
        if sha256_file(checkpoint)!=manifest['checkpoints'][arm]['sha256']:
            raise ValueError('checkpoint hash mismatch')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if saved['training_manifest_sha256']!=manifest['training_manifest_sha256'] or saved['arm']!=arm:
            raise ValueError('checkpoint provenance mismatch')
        model=HiRPNet(HiRPConfig(**saved['config'])).to(args.device).eval()
        model.load_state_dict(saved['model'],strict=True)
        state_before=state_hash(model)
        mu,logs=collect_priors(model,batch,arm,args.device)
        np.savez(output/f'{arm}_prior_cache.npz',mu=mu.numpy(),log_sigma=logs.numpy())
        for k in (32,64):
            cases=[('correct',1.)]
            if arm=='A1':
                cases += [('shuffled',1.),('dataset_mean',1.)]
                if k==32:
                    cases += [('correct',a) for a in (0.,.25,.5,1.5,2.)]
            for intervention,alpha in cases:
                chosen_mu,chosen_logs=intervention_prior(mu,logs,intervention,permutation)
                name=f'{arm}_{intervention}_alpha{alpha:g}_K{k}'
                result=evaluate_case(model,batch,chosen_mu,chosen_logs,bank[:,:k],alpha,args.device)
                result.update(arm=arm,intervention=intervention,alpha=alpha,K=k)
                all_results[name]=result
                (output/f'{name}.json').write_text(json.dumps(result,indent=2,allow_nan=False))
                print(json.dumps(dict(case=name,**result['channels']['overall'])),flush=True)
        assert state_before==state_hash(model),'evaluation changed model state'
        hashes_after[arm]=sha256_file(checkpoint)
        assert hashes_after[arm]==manifest['checkpoints'][arm]['sha256']
        del model,saved
        if args.device.startswith('cuda'): torch.cuda.empty_cache()
    deltas={}
    for k in (32,64):
        reference=all_results[f'A1_correct_alpha1_K{k}']
        for label in ('shuffled','dataset_mean'):
            deltas[f'{label}_minus_correct_K{k}']=paired_difference(reference,all_results[f'A1_{label}_alpha1_K{k}'])
    summary={name:{key:value for key,value in result.items() if key!='per_example'} for name,result in all_results.items()}
    (output/'summary.json').write_text(json.dumps(dict(cases=summary,paired_intervention_deltas=deltas),indent=2,allow_nan=False))
    evaluation_manifest=dict(training_manifest_sha256=manifest['training_manifest_sha256'],
        data_file_list_sha256=manifest['data_file_list_sha256'],checkpoint_hashes_after=hashes_after,
        checkpoint_state_unchanged=True,validation_ids=batch['clip_id'],validation_sessions=batch['session_id'],
        common_noise_sha256=sha256_file(output/'common_noise_K64.npy'),shuffle=shuffle,
        mean_prior_definition='arithmetic mean mu and arithmetic mean sigma over all 80 validation inputs, no targets',
        sigma_sweep_definition='mu + alpha * sigma * epsilon; alpha multiplies noise, no extra log_sigma clamp',
        metric_mask='pair_lengths for channel metrics and real reference; source_lengths for stochastic residual',
        source_sha256={str(p.relative_to(root)):sha256_file(p) for p in sorted((root/'hirp').rglob('*.py'))})
    (output/'manifest.json').write_text(json.dumps(evaluation_manifest,indent=2))
    print('EVALUATION_COMPLETE',str(output),flush=True)


if __name__=='__main__':main()
