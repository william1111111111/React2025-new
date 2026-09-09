"""Development-set evaluation via the public sampler; locked banks and shuffles."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import default_collate
from .phase22 import load_checkpoint,eval_adapter
from .phase21 import CHANNEL_SCALE
from .train_phase22 import ROOT,write
from .train_phase21 import configure,sync
from .train_phase1 import to_device,take_batch,state_hash
from .paired_data import paired_model_inputs
from .session_data import SessionPopulation
from .group_features import DescriptorScaler
from .group_scores import descriptor_es,session_mixture_score,paired_correspondence
from .losses import paired_energy_score
from .train_phase2 import description
from .phase15_metrics import channel_metrics,aggregate_channels
from .phase15_interventions import within_session_permutation
from .phase15_audit import sha256_file,canonical_hash
from .evaluate_phase2 import reference_cache_and_baseline,numbers,mean_dict

@torch.no_grad()
def evaluate_model(model,batch,refs,scaler_cpu,scaler_gpu,bank,permutation,device,k=32):
    if k<2:raise ValueError('Phase 2.1 evaluation requires K >= 2')
    outputs=[];centers=[];channel_rows=[];phis=[];validities=[]
    sync(device);started=time.perf_counter()
    if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats(device)
    for start in range(0,len(batch['clip_id']),4):
        indices=torch.arange(start,min(start+4,len(batch['clip_id'])))
        small=to_device(take_batch(batch,indices),device)
        noise=bank[:,:k].to(device).expand(len(indices),-1,-1)
        pred=eval_adapter(model,**paired_model_inputs(small),noise=noise)
        center=eval_adapter(model,**paired_model_inputs(small),noise=torch.zeros_like(noise[:,:1]))[:,0]
        phi,valid=description(pred,small['source_lengths'],scaler_gpu)
        outputs.append(pred.cpu());centers.append(center.cpu());phis.append(phi.cpu());validities.append(valid.cpu())
        channel_rows.extend(channel_metrics(pred,small['paired_target'],small['pair_lengths']))
    p=torch.cat(outputs);central=torch.cat(centers);phi=torch.cat(phis);valid=torch.cat(validities)
    per_input=[];groups=defaultdict(list)
    t=p.shape[2]
    for i,session in enumerate(batch['session_id']):
        groups[session].append(i)
        j=int(permutation[i]);length=batch['pair_lengths'][i:i+1]
        common=torch.minimum(length,batch['source_lengths'][j:j+1])
        target=batch['paired_target'][i:i+1]
        mask=torch.arange(t)[None]<length[:,None]
        common_mask=torch.arange(t)[None]<common[:,None]
        def score(pred,mask):
            return numbers(paired_energy_score(pred,target,mask,channel_scale=CHANNEL_SCALE,return_details=True))
        per_input.append(dict(clip_id=batch['clip_id'][i],session_id=session,shuffle_donor=batch['clip_id'][j],
            pair_length=int(length),common_shuffle_length=int(common),
            conditional=score(p[i:i+1],mask),
            conditional_correct_common=score(p[i:i+1],common_mask),
            conditional_shuffled=score(p[j:j+1],common_mask),
            central=numbers(paired_correspondence(central[i:i+1],target,length)),
            central_correct_common=numbers(paired_correspondence(central[i:i+1],target,common)),
            central_shuffled=numbers(paired_correspondence(central[j:j+1],target,common)),
            per_input_population=numbers(descriptor_es(phi[i],valid[i],*refs[session])),
            diversity=channel_rows[i]))
    sessions=[]
    for session,indices in sorted(groups.items()):
        selected=[per_input[i] for i in indices]
        marginal=numbers(session_mixture_score(phi[indices],valid[indices],*refs[session]))
        within=float(np.mean([r['per_input_population']['self'] for r in selected]))
        n=len(indices)
        between=(marginal['self']-within/n)/(1-1/n)
        row=dict(session_id=session,source_count=n,reference_count=len(refs[session][0]),marginal=marginal,
                 spread=dict(within_input_descriptor_self=within,between_input_descriptor_distance=between))
        for key in ('conditional','conditional_correct_common','conditional_shuffled','central','central_correct_common',
                    'central_shuffled','per_input_population'):
            row[key]=mean_dict([r[key] for r in selected])
        sessions.append(row)
    aggregate={key:mean_dict([r[key] for r in sessions]) for key in ('marginal','spread','conditional',
        'conditional_correct_common','conditional_shuffled','central','central_correct_common','central_shuffled','per_input_population')}
    for prefix,field in [('conditional','loss'),('central','overall')]:
        a=aggregate[prefix+'_correct_common'][field];b=aggregate[prefix+'_shuffled'][field]
        aggregate[prefix+'_shuffle_gap']=dict(correct=a,shuffled=b,ratio=b/a if a else None,difference=b-a)
    aggregate['channels']=aggregate_channels(channel_rows)
    sync(device)
    return dict(all_predictions=p,aggregate=aggregate,per_input=per_input,per_session=sessions,
        evaluation_seconds=time.perf_counter()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else None)


def freeze_plan(root):
    protocol=json.loads((root/'locked_protocol.json').read_text())
    old=json.loads((ROOT/'runs/phase21/conditional_score_v0/training_manifest.json').read_text())
    pool=SessionPopulation(ROOT/'data','val',128)
    batch=default_collate([pool.dataset[i] for i in old['validation_indices']])
    assert batch['clip_id']==old['validation_ids']
    out=root/'evaluation_plan';out.mkdir(exist_ok=False)
    plans=[]
    for seed in protocol['evaluation_noise_seeds']:
        bank=torch.randn(1,64,32,generator=torch.Generator().manual_seed(seed))
        path=out/f'noise_{seed}.npy';np.save(path,bank.numpy())
        plans.append(dict(seed=seed,path=str(path),sha256=sha256_file(path)))
    perms=[dict(seed=s,donors=within_session_permutation(batch['session_id'],s).tolist()) for s in protocol['permutation_seeds']]
    assert len({tuple(p['donors']) for p in perms})==len(perms)
    paths=set(pool.reference_paths.values())
    for i in old['validation_indices']:
        path=pool.dataset.records[i];rel=path.relative_to(pool.dataset.directory/'facial-attributes')
        paths.update([path,pool.dataset.directory/'audio-features'/rel,pool.dataset.directory/'coefficients'/rel])
    files=[dict(path=str(p),sha256=sha256_file(p)) for p in sorted(paths)]
    train=SessionPopulation(ROOT/'data','train',128)
    # Audit candidate unused validation clips without treating session IDs as participant IDs.
    train_names={p.stem for p in train.dataset.records};val_names={p.stem for p in pool.dataset.records}
    development=set(old['validation_indices'])
    candidate=[dict(index=i,clip_id=p.relative_to(pool.dataset.directory/'facial-attributes').with_suffix('').as_posix()) for i,p in enumerate(pool.dataset.records) if i not in development]
    write(out/'confirmation_candidate_audit.json',dict(candidate_unused_val=candidate,
        train_val_recording_basename_overlap=sorted(train_names&val_names),
        train_val_session_overlap=sorted(set(train.sources)&set(pool.sources)),
        participant_mapping_available=False,independent_confirmation_approved=False,
        reason='No authoritative participant/interaction crosswalk for these Camera/session feature directories. CSV identifiers use a different dataset naming scheme. Candidate list is sealed but NOT used as independent confirmation.',
        resampling_unit='session clusters for development summaries; repeated participants across sessions cannot be ruled out; frame counts are not independent units'))
    write(out/'manifest.json',dict(role='development, repeatedly used 80 clips',indices=old['validation_indices'],clip_ids=batch['clip_id'],
        clip_ids_hash=canonical_hash(batch['clip_id']),source_lengths=batch['source_lengths'].tolist(),pair_lengths=batch['pair_lengths'].tolist(),
        sessions=batch['session_id'],banks=plans,permutations=perms,data_files=files,data_hash=canonical_hash(files),
        normalization=old['normalization'],scaler_sha256=old['scaler_hash'],T=128,main_K=32,stability_K=64,
        fixed_visualization_indices=[0,20,40,60],aggregation='macro 20 sessions; bank0/K32 is learning-curve reference; final all predeclared banks/derangements',
        group_self='exclude equal noise index across every source pair'))


@torch.no_grad()
def extra_shuffles(pred,batch,permutations):
    result=[];t=pred.shape[2]
    for perm in permutations:
        rows=[]
        for i,j in enumerate(perm['donors']):
            n=min(int(batch['pair_lengths'][i]),int(batch['source_lengths'][j]));mask=torch.arange(t)[None]<n
            target=batch['paired_target'][i:i+1]
            c=float(paired_energy_score(pred[i:i+1],target,mask,channel_scale=CHANNEL_SCALE))
            s=float(paired_energy_score(pred[j:j+1],target,mask,channel_scale=CHANNEL_SCALE))
            rows.append(dict(clip_id=batch['clip_id'][i],session_id=batch['session_id'][i],donor_id=batch['clip_id'][j],
                             common_length=n,correct=c,shuffled=s,gap=s-c))
        grouped=defaultdict(list)
        for row in rows:grouped[row['session_id']].append(row)
        sessions=[dict(session_id=s,**{key:float(np.mean([r[key] for r in grouped[s]])) for key in ('correct','shuffled','gap')}) for s in sorted(grouped)]
        result.append(dict(seed=perm['seed'],per_input=rows,per_session=sessions,mean_gap=float(np.mean([r['gap'] for r in sessions]))))
    return result


def evaluate(root,seed,device):
    plan=json.loads((root/'evaluation_plan/manifest.json').read_text())
    run=root/f'seed_{seed}';manifest=json.loads((run/'manifest.json').read_text())
    for row in plan['data_files']+plan['normalization']:
        if sha256_file(row['path'])!=row['sha256']:raise ValueError('evaluation data hash mismatch')
    for row in plan['banks']:
        if sha256_file(row['path'])!=row['sha256']:raise ValueError('bank hash mismatch')
    if sha256_file(manifest['scaler_path'])!=plan['scaler_sha256']:raise ValueError('scaler hash mismatch')
    pool=SessionPopulation(ROOT/'data','val',128);batch=default_collate([pool.dataset[i] for i in plan['indices']])
    assert batch['clip_id']==plan['clip_ids']
    scaler=DescriptorScaler.load(manifest['scaler_path']);scaler_gpu=DescriptorScaler.load(manifest['scaler_path']).to(device)
    refs,real=reference_cache_and_baseline(pool,scaler)
    out=run/'evaluation';out.mkdir(exist_ok=True)
    if not (out/'real_reference.json').exists():write(out/'real_reference.json',real)
    completed=[]
    for arm in ('C0','C1','C2'):
        summaries=sorted((run/arm).glob('attempt_*/summary.json'))
        all_ckpts={}
        for summary in summaries:
            for ck in json.loads(summary.read_text())['checkpoints']:all_ckpts[ck['step']]=ck
        for step in manifest['eval_steps']:
            if step not in all_ckpts:raise ValueError(f'missing {arm} checkpoint {step}')
            ck=all_ckpts[step]
            if sha256_file(ck['path'])!=ck['sha256']:raise ValueError('checkpoint fingerprint mismatch')
            model,saved=load_checkpoint(ck['path'],device)
            if saved['global_step']!=step or saved['scales']!=manifest['scales']:raise ValueError('loaded step/scales mismatch')
            before=state_hash(model)
            for bi,bank_info in enumerate(plan['banks'] if step==manifest['config']['max_steps'] else plan['banks'][:1]):
                bank=torch.from_numpy(np.load(bank_info['path']))
                for k in ((32,64) if step==manifest['config']['max_steps'] else (32,)):
                    label=f'{arm}_step{step}_bank{bi}_K{k}';path=out/(label+'.json')
                    if path.exists():
                        previous=json.loads(path.read_text())
                        if previous['checkpoint_sha256']!=ck['sha256']:raise ValueError('existing evaluation checkpoint mismatch')
                        completed.append(dict(arm=arm,step=step,bank=bi,K=k,path=str(path)));continue
                    r=evaluate_model(model,batch,refs,scaler,scaler_gpu,bank[:,:k],torch.tensor(plan['permutations'][0]['donors']),device,k)
                    pred=r.pop('all_predictions')
                    r['permutations']=extra_shuffles(pred,batch,plan['permutations'])
                    r.update(arm=arm,training_seed=seed,step=step,bank_seed=bank_info['seed'],K=k,checkpoint_sha256=ck['sha256'],
                             evaluation_plan_sha256=sha256_file(root/'evaluation_plan/manifest.json'))
                    if step==manifest['config']['max_steps'] and bi==0 and k==32:
                        ids=plan['fixed_visualization_indices']
                        np.savez_compressed(out/(label+'_fixed_cases.npz'),predictions=pred[ids,:10].numpy(),
                            targets=batch['paired_target'][ids].numpy(),lengths=batch['pair_lengths'][ids].numpy(),
                            clip_ids=np.array([batch['clip_id'][i] for i in ids]))
                    assert state_hash(model)==before
                    write(path,r);completed.append(dict(arm=arm,step=step,bank=bi,K=k,path=str(path)))
                    print('EVAL',seed,label,'conditional',r['aggregate']['conditional']['loss'],'group',r['aggregate']['marginal']['loss'],flush=True)
            assert sha256_file(ck['path'])==ck['sha256']
            del model,saved
            if device.startswith('cuda'):torch.cuda.empty_cache()
    write(out/'completed.json',completed)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--seed',type=int,default=123)
    p.add_argument('--device',default='cuda:4');p.add_argument('--prepare',action='store_true')
    a=p.parse_args();configure()
    if a.prepare:freeze_plan(a.root)
    else:evaluate(a.root,a.seed,a.device)


if __name__=='__main__':main()
