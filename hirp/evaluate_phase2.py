"""Read-only fixed-80 validation of correspondence and session distributions."""
import argparse,json,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate
from . import HiRPNet,HiRPConfig
from .session_data import SessionPopulation
from .group_features import DescriptorScaler,reaction_descriptor
from .group_scores import (paired_correspondence,descriptor_es,descriptor_distances,session_mixture_score)
from .train_phase2 import description
from .paired_data import paired_model_inputs
from .phase1_diagnostics import prior_mode
from .phase15_metrics import channel_metrics,aggregate_channels,trajectory_summary,distribution
from .phase15_interventions import within_session_permutation
from .phase15_audit import sha256_file,canonical_hash
from .train_phase1 import to_device,take_batch,state_hash


def numbers(values):return {key:value.item() if torch.is_tensor(value) else value for key,value in values.items()}


def mean_dict(rows):return {key:float(np.mean([row[key] for row in rows])) for key in rows[0]}


@torch.no_grad()
def reference_cache_and_baseline(pool,scaler):
    cache={};crops=[];sessions={}
    for session in sorted(pool.references):
        ref_ids=pool.references[session];features=[];masks=[];stats=[]
        for ref_id in ref_ids:
            item=pool.load_reference(ref_id)
            phi,valid=scaler(*reaction_descriptor(item['reaction'],item['length']))
            features.append(phi);masks.append(valid)
            crops.append(dict(record_id=ref_id,crop_start=item['crop_start'],length=int(item['length'])))
            stats.append({name:trajectory_summary(item['reaction'][None,:int(item['length']),channels],name)
                          for name,channels in [('overall',slice(0,25)),('AU',slice(0,15)),('VA',slice(15,17)),('expression',slice(17,25))]})
        features,masks=torch.stack(features),torch.stack(masks)
        cache[session]=(features,masks)
        n=len(ref_ids);off=torch.triu_indices(n,n,1)
        distances=descriptor_distances(features,masks,features,masks)[off[0],off[1]].tolist()
        sessions[session]=dict(reference_count=n,unordered_pair_count=len(distances),reference_ids=ref_ids,
             real_real_descriptor=distribution(distances),descriptor_pair_distances=distances,
             trajectories={g:{key:float(np.mean([row[g][key] for row in stats if row[g][key] is not None]))
                               if any(row[g][key] is not None for row in stats) else None
                               for key in stats[0][g]} for g in stats[0]})
    aggregate=dict(reference_count=len(pool.reference_paths),pair_count=sum(s['unordered_pair_count'] for s in sessions.values()),
        real_real_descriptor_session_macro_mean=float(np.mean([s['real_real_descriptor']['mean'] for s in sessions.values() if s['real_real_descriptor']['count']])),
        trajectories={g:mean_dict([s['trajectories'][g] for s in sessions.values()]) for g in ('overall','AU','VA','expression')})
    return cache,dict(sessions=sessions,aggregate=aggregate,reference_crops=crops,
                     definition='Uniform unique references within session; summaries macro-average sessions. Different interactions, not conditional repeats.')


@torch.no_grad()
def evaluate_arm(arm,model,batch,refs,scaler_cpu,scaler_gpu,bank,permutation,device):
    centers=[];features=[];validities=[];channel_rows=[];residual_rows=[]
    if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize(device)
    started=time.perf_counter()
    for start in range(0,len(batch['clip_id']),4):
        indices=torch.arange(start,min(start+4,len(batch['clip_id'])))
        small=to_device(take_batch(batch,indices),device)
        capture=[]
        with prior_mode(model,'A0'):
            central=model(**paired_model_inputs(small),sample_count=1,noise=torch.zeros(len(indices),1,32,device=device))[:,0]
            hook=model.output_head.stochastic.register_forward_hook(lambda module,inputs,output:capture.append(output.detach()))
            try:pred=model(**paired_model_inputs(small),sample_count=32,noise=bank.to(device).expand(len(indices),-1,-1))
            finally:hook.remove()
        feat,valid=description(pred,small['source_lengths'],scaler_gpu)
        centers.append(central.cpu());features.append(feat.cpu());validities.append(valid.cpu())
        channel_rows.extend(channel_metrics(pred,small['paired_target'],small['pair_lengths']))
        for j,length in enumerate(small['source_lengths'].tolist()):
            raw=capture[0][j,:,:length].float()
            residual_rows.append(dict(stochastic_raw_abs_mean=raw.abs().mean().item(),
                stochastic_residual_abs_mean=raw.tanh().abs().mean().item(),tanh_saturation_fraction=(raw.tanh().abs()>=.99).float().mean().item()))
    central,phi,pvalid=torch.cat(centers),torch.cat(features),torch.cat(validities)
    paired,pair_valid=description(batch['paired_target'],batch['pair_lengths'],scaler_cpu)
    per_input=[];groups=defaultdict(list)
    for i,session in enumerate(batch['session_id']):groups[session].append(i)
    for i in range(len(central)):
        j=int(permutation[i])
        correct=numbers(paired_correspondence(central[i:i+1],batch['paired_target'][i:i+1],batch['pair_lengths'][i:i+1]))
        joint=torch.minimum(batch['pair_lengths'][i:i+1],batch['source_lengths'][j:j+1])
        shuffled=numbers(paired_correspondence(central[j:j+1],batch['paired_target'][i:i+1],joint))
        cp=numbers(descriptor_es(phi[i],pvalid[i],paired[i:i+1],pair_valid[i:i+1]))
        sp=numbers(descriptor_es(phi[j],pvalid[j],paired[i:i+1],pair_valid[i:i+1]))
        population=numbers(descriptor_es(phi[i],pvalid[i],*refs[batch['session_id'][i]]))
        per_input.append(dict(clip_id=batch['clip_id'][i],session_id=batch['session_id'][i],shuffle_donor=batch['clip_id'][j],
                              central_correct=correct,central_shuffled=shuffled,paired_descriptor_correct=cp,
                              paired_descriptor_shuffled=sp,per_input_population=population,
                              diversity=channel_rows[i],residual=residual_rows[i]))
    per_session=[]
    for session,indices in sorted(groups.items()):
        score=numbers(session_mixture_score(phi[indices],pvalid[indices],*refs[session]))
        selected=[per_input[i] for i in indices]
        per_session.append(dict(session_id=session,source_count=len(indices),reference_count=len(refs[session][0]),
             marginal=score,central_correct=mean_dict([r['central_correct'] for r in selected]),
             central_shuffled=mean_dict([r['central_shuffled'] for r in selected]),
             paired_descriptor_correct=mean_dict([r['paired_descriptor_correct'] for r in selected]),
             paired_descriptor_shuffled=mean_dict([r['paired_descriptor_shuffled'] for r in selected])))
    aggregate=dict(pair=mean_dict([r['central_correct'] for r in per_input]),
                   shuffled_pair=mean_dict([r['central_shuffled'] for r in per_input]),
                   paired_descriptor=mean_dict([r['paired_descriptor_correct'] for r in per_input]),
                   shuffled_paired_descriptor=mean_dict([r['paired_descriptor_shuffled'] for r in per_input]),
                   session_marginal=mean_dict([r['marginal'] for r in per_session]),
                   per_input_population=mean_dict([r['per_input_population'] for r in per_input]),
                   diversity=aggregate_channels(channel_rows),residual=mean_dict(residual_rows))
    for label,correct,shuffled in [('central',aggregate['pair']['overall'],aggregate['shuffled_pair']['overall']),
                                  ('paired_descriptor',aggregate['paired_descriptor']['loss'],aggregate['shuffled_paired_descriptor']['loss'])]:
        aggregate[label+'_conditionality']=dict(correct=correct,shuffled=shuffled,ratio=shuffled/correct if correct else None,difference=shuffled-correct)
    if device.startswith('cuda'):torch.cuda.synchronize(device)
    return dict(arm=arm,aggregate=aggregate,per_input=per_input,per_session=per_session,
                evaluation_seconds=time.perf_counter()-started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else None)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,default=Path('runs/phase2'))
    parser.add_argument('--device',default='cuda:2');parser.add_argument('--output',type=Path)
    args=parser.parse_args();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    root=Path(__file__).resolve().parents[1];run=args.run
    manifest=json.loads((run/'training_manifest.json').read_text());trained=json.loads((run/'training_summary.json').read_text())
    if trained.get('stopped_before_training'):raise RuntimeError('training was stopped by gradient gate; no evaluation')
    scaler_path=run/'descriptor_stats.json'
    assert sha256_file(scaler_path)==manifest['scaler_hash']
    assert canonical_hash(manifest['data_files'])==manifest['data_file_list_hash']
    for record in manifest['data_files']+manifest['normalization']:
        assert sha256_file(root/record['path'])==record['sha256'],record['path']
    output=args.output or run/'evaluation';output.mkdir(parents=True,exist_ok=False)
    pool=SessionPopulation(root/'data','val');batch=default_collate([pool.dataset[i] for i in manifest['validation_indices']])
    assert batch['clip_id']==manifest['validation_ids']
    scaler_cpu=DescriptorScaler.load(scaler_path);scaler_gpu=DescriptorScaler.load(scaler_path).to(args.device)
    refs,real=reference_cache_and_baseline(pool,scaler_cpu)
    (output/'real_reference.json').write_text(json.dumps(real,indent=2,allow_nan=False))
    # Reuse exactly Phase 1.5's frozen common random numbers and clip ordering.
    bank=torch.from_numpy(np.load(root/'runs/phase15/evaluation/common_noise_K64.npy'))[:,:32].clone()
    np.save(output/'common_noise_K32.npy',bank.numpy())
    permutation=within_session_permutation(batch['session_id'],1502)
    results={};checkpoint_hashes={}
    for arm in ('B2','B3','B4'):
        path=run/'checkpoints'/f'{arm}.pt';digest=sha256_file(path)
        assert digest==trained['arms'][arm]['checkpoint_sha256']
        saved=torch.load(path,map_location='cpu',weights_only=True)
        assert saved['scaler_sha256']==manifest['scaler_hash'] and saved['prior_mode']=='A0'
        assert saved['training_manifest_sha256']==sha256_file(run/'training_manifest.json')
        model=HiRPNet(HiRPConfig(**saved['config'])).to(args.device).eval();model.load_state_dict(saved['model'],strict=True)
        before=state_hash(model)
        result=evaluate_arm(arm,model,batch,refs,scaler_cpu,scaler_gpu,bank,permutation,args.device)
        assert before==state_hash(model) and sha256_file(path)==digest
        checkpoint_hashes[arm]=digest;results[arm]=result
        (output/f'{arm}.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        print('EVAL',arm,json.dumps(result['aggregate']),flush=True)
        del model,saved
        if args.device.startswith('cuda'):torch.cuda.empty_cache()
    (output/'summary.json').write_text(json.dumps({arm:r['aggregate'] for arm,r in results.items()},indent=2))
    evaluation_manifest=dict(validation_ids=batch['clip_id'],validation_ids_hash=canonical_hash(batch['clip_id']),
        scaler_hash=sha256_file(scaler_path),checkpoint_hashes=checkpoint_hashes,K=32,prior_mode='A0',
        common_noise_hash=sha256_file(output/'common_noise_K32.npy'),
        shuffle=[dict(recipient=batch['clip_id'][i],donor=batch['clip_id'][j]) for i,j in enumerate(permutation.tolist())],
        session_marginal_self='uniform fixed-source mixture; exclude equal epsilon indices across all source pairs',
        reference_mass='uniform unique val listener clips within each session; all available refs',
        aggregation='macro over 20 validation sessions (four sources each)',
        conditionality_gap='ratio shuffled/correct; additionally report shuffled-correct difference',
        source_sha256={str(p.relative_to(root)):sha256_file(p) for p in sorted((root/'hirp').rglob('*.py'))})
    (output/'manifest.json').write_text(json.dumps(evaluation_manifest,indent=2))
    print('PHASE2_EVALUATION_COMPLETE',flush=True)


if __name__=='__main__':main()
