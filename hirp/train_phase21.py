"""Staged, bounded Phase 2.1 experiment. Every stage refuses to overwrite results."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from torch.utils.data import default_collate
from .phase21 import OutputConfig, make_model, losses_for_batch, CHANNEL_SCALE
from .phase21_diagnostics import output_path_diagnostic, component_gradients
from .phase1_diagnostics import gradient_norm
from .phase15_audit import sha256_file, canonical_hash
from .session_data import SessionPopulation, make_schedule
from .group_features import DescriptorScaler, reaction_descriptor
from .train_phase1 import to_device, state_hash

ROOT = Path(__file__).resolve().parents[1]
BASE = '8fbc5febddb14b0b54d27633a3edb95227cc9834'
ARMS = ('C0','C1','C2')


def write(path,value):
    with path.open('x') as stream: json.dump(value,stream,indent=2,allow_nan=False)


def configure():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


def source_hashes():
    return {str(p.relative_to(ROOT)):sha256_file(p) for p in sorted((ROOT/'hirp').rglob('*.py'))}


def prepare(run,device):
    run.mkdir(parents=True,exist_ok=False)
    old=ROOT/'runs/phase2'
    manifest=json.loads((old/'training_manifest.json').read_text())
    # Verify every actual selected datum, normalization and scaler, not just IDs.
    for row in manifest['data_files']+manifest['normalization']:
        assert sha256_file(ROOT/row['path'])==row['sha256'],row['path']
    assert sha256_file(old/'descriptor_stats.json')==manifest['scaler_hash']
    for name in ('descriptor_stats.json','schedule.json','training_noise.npy'):
        shutil.copyfile(old/name,run/name)
    pool=SessionPopulation(ROOT/'data','train')
    records,noise,hashes=make_schedule(pool)
    saved=json.loads((run/'schedule.json').read_text())
    assert records==saved['records'] and hashes==saved['hashes']
    assert np.array_equal(noise.numpy(),np.load(run/'training_noise.npy'))
    for record in records:
        for i in record['source_indices']:
            assert pool.dataset[i]['pair_lengths']>0
    history={str(p.relative_to(ROOT)):sha256_file(p) for p in sorted((ROOT/'runs').rglob('*'))
             if p.is_file() and not p.is_relative_to(run.resolve())}
    write(run/'historical_artifacts.json',history)
    manifest.update(phase='2.1',base_commit=BASE,
        local_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        config=dict(T=128,K_train=4,B=4,R_max=8,pilot_steps=128,smoke_steps=32,lr=1e-4,
                    weight_decay=.01,optimizer='AdamW',prior_mode='A0',channel_scale=list(CHANNEL_SCALE),
                    precision='float32; AMP tested separately, not used in pilot',gradient_audit_interval=16,
                    preflight_batches=8,lambda_candidates=[.01,.03,.1,.3,1.],gradient_target_ratio=.5),
        old_definition='central_RMS(G(X,0),paired_Y) + descriptor_distribution_loss',
        new_definition='paired_trajectory_ES(all K stochastic predictions,Y) + optional lambda*group_descriptor_ES',
        center_role='diagnostic only; not an expectation or median',
        scaler_provenance='unchanged frozen TRAIN-only Phase 2 scaler, hash-verified; never refit',
        smoke_selection_rule='Both use conditional ES only. Viable iff finite nonzero group noise VJP, '
            'residual derivative median > 1e-3 and every group boundary < .95. Prefer pre-norm if old fails '
            'or it reduces residual saturation by at least .05; otherwise retain old. If neither passes stop.',
        device=device,source_sha256=source_hashes())
    write(run/'preparation_manifest.json',manifest)
    (run/'checkpoints').mkdir();(run/'smoke').mkdir()


def load_data(run,device):
    pool=SessionPopulation(ROOT/'data','train')
    records=json.loads((run/'schedule.json').read_text())['records']
    noise=torch.from_numpy(np.load(run/'training_noise.npy'))
    scaler=DescriptorScaler.load(run/'descriptor_stats.json')
    ids=sorted({i for r in records for i in r['source_indices']})
    sources={i:pool.dataset[i] for i in ids}
    refs={}
    # Shared preprocessing; C0's step path never indexes this cache.
    with torch.no_grad():
        for ref_id in sorted({i for r in records for i in r['reference_ids']}):
            item=pool.load_reference(ref_id)
            refs[ref_id]=scaler(*reaction_descriptor(item['reaction'],item['length']))
    return records,noise,scaler.to(device),sources,refs


def step_batch(record,sources,refs,arm,device):
    batch=to_device(default_collate([sources[i] for i in record['source_indices']]),device)
    if arm=='C0':return batch,None
    chosen=[refs[r] for r in record['reference_ids']]
    return batch,tuple(torch.stack([r[j] for r in chosen]).to(device) for j in (0,1))


def sync(device):
    if device.startswith('cuda'):torch.cuda.synchronize(device)


def train(run,label,arm,output_config,weight,steps,data,device,manifest_hash=None):
    records,noises,scaler,sources,refs=data
    logfile=run/(label+'_training.jsonl')
    if logfile.exists():raise FileExistsError(logfile)
    model=make_model(device,output_config).train()
    initial=state_hash(model);prior={k:v.detach().clone() for k,v in model.prior.state_dict().items()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    probe_batch,_=step_batch(records[0],sources,refs,'C0',device)
    probe_noise=noises[0].to(device)
    probes=[dict(step=0,**output_path_diagnostic(model,probe_batch,probe_noise))]
    sessions=[];occurrences=[];references=[];noise_hash=hashlib.sha256();rows=[]
    if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats(device)
    sync(device);started=time.perf_counter()
    with logfile.open('x') as stream:
        for record,noise in zip(records[:steps],noises[:steps]):
            batch,selected=step_batch(record,sources,refs,arm,device)
            optimizer.zero_grad(set_to_none=True)
            values=losses_for_batch(model,batch,noise.to(device),scaler,arm,selected,lambda_group=weight)
            audit_step=record['step']==1 or record['step']%16==0
            audit=component_gradients(model,values['conditional']['loss'],
                                     values['group']['loss'] if values['group'] else None,weight) if audit_step else None
            values['total'].backward()
            finite=all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
            if not finite:raise RuntimeError('nonfinite gradient; stopping bounded run')
            row=dict(step=record['step'],session_id=record['session_id'],loss=values['total'].item(),
                conditional={k:v.item() for k,v in values['conditional'].items()},
                group={k:v.item() for k,v in values['group'].items()} if values['group'] else None,
                total_gradient_norm=gradient_norm(model),prior_gradient_norm=gradient_norm(model.prior),
                component_gradients=audit)
            optimizer.step()
            stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();rows.append(row)
            sessions.append(record['session_id']);occurrences.append(record['source_indices'])
            noise_hash.update(noise.numpy().tobytes())
            if arm!='C0':references.append(record['reference_ids'])
            if record['step']%16==0 or record['step']==steps:
                probes.append(dict(step=record['step'],**output_path_diagnostic(model,probe_batch,probe_noise)))
                print(label,record['step'],json.dumps(row['conditional']),flush=True)
    sync(device);elapsed=time.perf_counter()-started
    assert all(torch.equal(prior[k],v) for k,v in model.prior.state_dict().items())
    path=run/'checkpoints'/(label+'.pt');path.parent.mkdir(exist_ok=True)
    if path.exists():raise FileExistsError(path)
    torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
        config=asdict(model.config),output_config=asdict(output_config),prior_mode='A0',arm=arm,steps=steps,
        channel_scale=list(CHANNEL_SCALE),lambda_group=weight,manifest_hash=manifest_hash),path)
    write(run/(label+'_path_diagnostics.json'),probes)
    result=dict(arm=arm,output_config=asdict(output_config),initialization_hash=initial,
        parameter_count=sum(p.numel() for p in model.parameters()),steps=steps,
        optimizer=dict(name='AdamW',lr=1e-4,weight_decay=.01),channel_scale_hash=canonical_hash(CHANNEL_SCALE),
        session_schedule_hash=canonical_hash(sessions),source_occurrence_hash=canonical_hash(occurrences),
        noise_hash=noise_hash.hexdigest(),reference_selection_hash=canonical_hash(references) if references else None,
        population_reference_lookups=sum(map(len,references)),scaler_hash=sha256_file(run/'descriptor_stats.json'),
        checkpoint_sha256=sha256_file(path),prior_unchanged=True,first=rows[0],last=rows[-1],
        elapsed_seconds_including_diagnostic_probes=elapsed,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else None)
    del model,optimizer
    if device.startswith('cuda'):torch.cuda.empty_cache()
    return result,probes[-1]


def viable(probe):
    return (all(v['finite'] and v['vjp_norm']>1e-8 for v in probe['noise_jacobian'].values())
            and probe['residual']['derivative']['abs_quantiles']['p50']>1e-3
            and all(v['boundary_fraction']<.95 for v in probe['outputs'].values()))


def smoke(run,device):
    if (run/'smoke_summary.json').exists():raise FileExistsError('smoke already completed')
    data=load_data(run,device)
    probe,_=step_batch(data[0][0],data[3],data[4],'C0',device)
    historical={}
    for arm in ('B2','B3','B4'):
        path=ROOT/'runs/phase2/checkpoints'/f'{arm}.pt'
        if not path.exists():
            historical[arm]=dict(unavailable=True);continue
        digest=sha256_file(path)
        saved=torch.load(path,map_location='cpu',weights_only=True)
        model=make_model(device)
        # No norm params in this configuration; strict load + identical state.
        model.load_state_dict(saved['model'],strict=True)
        before=state_hash(model)
        historical[arm]=dict(checkpoint_sha256=digest,objective_for_gradient='new paired trajectory ES, read-only probe',
                            diagnostic=output_path_diagnostic(model,probe,data[1][0].to(device)))
        assert before==state_hash(model) and digest==sha256_file(path)
        del model,saved
    write(run/'historical_output_path.json',historical)
    results={};last={}
    for name,config in [('old_head',OutputConfig()),('pre_norm_head',OutputConfig(head_pre_norm=True))]:
        results[name],last[name]=train(run,name,'C0',config,0.,32,data,device)
    eligible={name:viable(p) for name,p in last.items()}
    chosen=None
    if eligible['old_head']:chosen='old_head'
    if eligible['pre_norm_head'] and (not eligible['old_head'] or
        last['pre_norm_head']['residual']['tanh_saturation_fraction'] <= last['old_head']['residual']['tanh_saturation_fraction']-.05):
        chosen='pre_norm_head'
    write(run/'smoke_summary.json',dict(results=results,viable=eligible,selected=chosen,
        selection_uses='fixed TRAIN probe only, predeclared rule; no validation accessed',
        output_config=results[chosen]['output_config'] if chosen else None))
    print('SMOKE',eligible,'selected',chosen,flush=True)


def preflight(run,device):
    selected=json.loads((run/'smoke_summary.json').read_text())
    if selected['selected'] is None:raise RuntimeError('neither output path viable; no pilot permitted')
    config=OutputConfig(**selected['output_config']);data=load_data(run,device)
    records,noises,scaler,sources,refs=data;rows=[]
    for arm in ('C1','C2'):
        model=make_model(device,config).train()
        for record,noise in zip(records[:8],noises[:8]):
            batch,reference=step_batch(record,sources,refs,arm,device)
            values=losses_for_batch(model,batch,noise.to(device),scaler,arm,reference,lambda_group=1.)
            audit=component_gradients(model,values['conditional']['loss'],values['group']['loss'],1.)
            if not audit['finite']:raise RuntimeError('nonfinite preflight gradient')
            rows.append(dict(arm=arm,step=record['step'],session_id=record['session_id'],
                             conditional_loss_magnitude=values['conditional']['loss'].item(),group_loss_magnitude=values['group']['loss'].item(),**audit))
        del model
    ratios=[row['weighted_group_over_conditional'] for row in rows]
    candidates=[.01,.03,.1,.3,1.]
    median=float(np.median(ratios))
    weight=min(candidates,key=lambda w:abs(np.log(w*median/.5)))
    manifest=json.loads((run/'preparation_manifest.json').read_text())
    manifest.update(output_config=asdict(config),lambda_group=weight,source_sha256=source_hashes(),
        selection=dict(rule='nearest log-distance to median weighted group/conditional grad ratio .5 over 8 train batches x C1/C2',
                       unweighted_median_ratio=median,candidate_weighted_medians={str(w):w*median for w in candidates},
                       no_dynamic_controller=True,head_selection=selected['selected']))
    write(run/'gradient_preflight.json',dict(rows=rows,selected_lambda=weight,median_unweighted_ratio=median))
    write(run/'training_manifest.json',manifest)
    print('LOCKED lambda',weight,'median weighted ratio',weight*median,flush=True)


def pilot(run,device):
    manifest=json.loads((run/'training_manifest.json').read_text())
    data=load_data(run,device);results={}
    for arm in ARMS:
        results[arm],_=train(run,arm,arm,OutputConfig(**manifest['output_config']),manifest['lambda_group'],
                            128,data,device,sha256_file(run/'training_manifest.json'))
    for key in ('initialization_hash','parameter_count','steps','session_schedule_hash','source_occurrence_hash',
                'noise_hash','scaler_hash','channel_scale_hash'):
        assert len({r[key] for r in results.values()})==1,key
    assert results['C1']['reference_selection_hash']==results['C2']['reference_selection_hash']
    assert results['C0']['population_reference_lookups']==0
    write(run/'training_summary.json',dict(matched_controls_verified=True,arms=results))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=Path('runs/phase21/conditional_score_v0'))
    parser.add_argument('--device',default='cuda:2')
    parser.add_argument('--stage',choices=('prepare','smoke','preflight','pilot'),required=True)
    args=parser.parse_args();configure()
    stage=globals()[args.stage]
    started=time.time();stage(args.run,args.device)
    with (args.run/'commands.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(argv=[sys.executable,'-m','hirp.train_phase21']+sys.argv[1:],
                                    started_unix=started,completed_unix=time.time(),exit_code=0))+'\n')


if __name__=='__main__':main()
