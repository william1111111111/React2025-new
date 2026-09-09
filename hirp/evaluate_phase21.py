"""Read-only fixed-80 Phase 2.1 validation, all-K conditional trajectory ES."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.utils.data import default_collate
from . import HiRPConfig
from .phase21 import OutputConfig, make_model, forward_a0, CHANNEL_SCALE
from .phase21_diagnostics import output_path_diagnostic
from .paired_data import paired_model_inputs
from .session_data import SessionPopulation
from .group_features import DescriptorScaler
from .group_scores import descriptor_es, session_mixture_score, paired_correspondence
from .losses import paired_energy_score
from .train_phase2 import description
from .train_phase21 import configure, write, source_hashes, sync
from .train_phase1 import to_device, take_batch, state_hash
from .phase15_audit import sha256_file, canonical_hash
from .phase15_metrics import channel_metrics, aggregate_channels
from .phase15_interventions import within_session_permutation
from .evaluate_phase2 import reference_cache_and_baseline, numbers, mean_dict


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
        pred=forward_a0(model,**paired_model_inputs(small),sample_count=k,noise=noise)
        center=forward_a0(model,**paired_model_inputs(small),sample_count=1,noise=torch.zeros_like(noise[:,:1]))[:,0]
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
    return dict(aggregate=aggregate,per_input=per_input,per_session=sessions,
        evaluation_seconds=time.perf_counter()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else None)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,default=Path('runs/phase21/conditional_score_v0'))
    parser.add_argument('--device',default='cuda:2');parser.add_argument('--output',type=Path)
    args=parser.parse_args();configure();started=time.time()
    root=Path(__file__).resolve().parents[1];run=args.run
    manifest=json.loads((run/'training_manifest.json').read_text())
    trained=json.loads((run/'training_summary.json').read_text())
    assert sha256_file(run/'descriptor_stats.json')==manifest['scaler_hash']
    for row in manifest['data_files']+manifest['normalization']:
        assert sha256_file(root/row['path'])==row['sha256'],row['path']
    out=args.output or run/'evaluation';out.mkdir(parents=True,exist_ok=False)
    pool=SessionPopulation(root/'data','val')
    batch=default_collate([pool.dataset[i] for i in manifest['validation_indices']])
    assert batch['clip_id']==manifest['validation_ids']
    scaler=DescriptorScaler.load(run/'descriptor_stats.json')
    scaler_gpu=DescriptorScaler.load(run/'descriptor_stats.json').to(args.device)
    refs,real=reference_cache_and_baseline(pool,scaler)
    write(out/'real_reference.json',real)
    bank=torch.from_numpy(np.load(root/'runs/phase2/evaluation/common_noise_K32.npy'))
    np.save(out/'common_noise_K32.npy',bank.numpy())
    permutation=within_session_permutation(batch['session_id'],1502)
    results={};checkpoints={}
    for arm in ('C0','C1','C2'):
        path=run/'checkpoints'/f'{arm}.pt';digest=sha256_file(path)
        assert digest==trained['arms'][arm]['checkpoint_sha256']
        saved=torch.load(path,map_location='cpu',weights_only=True)
        assert saved['manifest_hash']==sha256_file(run/'training_manifest.json')
        assert saved['output_config']==manifest['output_config'] and saved['prior_mode']=='A0'
        assert saved['channel_scale']==list(CHANNEL_SCALE)
        model=make_model(args.device,OutputConfig(**saved['output_config']),HiRPConfig(**saved['config'])).eval()
        model.load_state_dict(saved['model'],strict=True);before=state_hash(model)
        result=evaluate_model(model,batch,refs,scaler,scaler_gpu,bank,permutation,args.device)
        probe=to_device(take_batch(batch,torch.arange(4)),args.device)
        result['heldout_output_path_probe']=output_path_diagnostic(model,probe,bank[:,:4].to(args.device).expand(4,-1,-1))
        assert state_hash(model)==before and sha256_file(path)==digest
        write(out/f'{arm}.json',result);results[arm]=result['aggregate'];checkpoints[arm]=digest
        print('EVAL',arm,json.dumps(result['aggregate']),flush=True)
        del model,saved
        if args.device.startswith('cuda'):torch.cuda.empty_cache()
    write(out/'summary.json',results)
    write(out/'manifest.json',dict(validation_ids=batch['clip_id'],validation_ids_hash=canonical_hash(batch['clip_id']),
        validation_indices=manifest['validation_indices'],K=32,prior_mode='A0',channel_scale=list(CHANNEL_SCALE),
        scaler_sha256=sha256_file(run/'descriptor_stats.json'),checkpoint_hashes=checkpoints,
        noise_sha256=sha256_file(out/'common_noise_K32.npy'),
        shuffle=[dict(recipient=batch['clip_id'][i],donor=batch['clip_id'][j]) for i,j in enumerate(permutation.tolist())],
        conditionality_comparison_mask='min(recipient pair length, donor source length), same for correct and shuffled',
        group_self='unchanged session_mixture_score: exclude same noise index for every source pair',
        aggregation='macro 20 sessions x 4 clips; unique uniform within-session VAL refs',source_sha256=source_hashes()))
    with (run/'commands.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(argv=[sys.executable,'-m','hirp.evaluate_phase21']+sys.argv[1:],
            started_unix=started,completed_unix=time.time(),exit_code=0))+'\n')


if __name__=='__main__':main()
