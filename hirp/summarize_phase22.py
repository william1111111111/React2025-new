"""Summarize only completed runs; seed variability and conditional bootstrap stay separate."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .train_phase22 import write

ROOT=Path('runs/phase22/replicated_v1')
ARMS=('C0','C1','C2')


def read(path):return json.loads(path.read_text())


def main():
    out=ROOT/'report';out.mkdir(exist_ok=False)
    seeds=[s for s in (123,42,2026) if (ROOT/f'seed_{s}/evaluation/completed.json').exists()]
    if not seeds:raise RuntimeError('no completed evaluated seeds')
    final={};table=[];learning=[];training=[];audits={}
    for seed in seeds:
        run=ROOT/f'seed_{seed}';summaries={}
        for arm in ARMS:
            summary=read(sorted((run/arm).glob('attempt_*/summary.json'))[-1]);summaries[arm]=summary
            assert summary['completed'] and summary['actual_optimizer_steps']==summary['requested_steps']==summary['training_rows']==2000
            for p in sorted((run/'evaluation').glob(f'{arm}_step*_bank0_K32.json')):
                r=read(p);a=r['aggregate'];learning.append(dict(seed=seed,arm=arm,step=r['step'],conditional=a['conditional']['loss'],group=a['marginal']['loss']))
            rows=[json.loads(s) for s in sorted((run/arm).glob('attempt_*/training.jsonl'))[-1].read_text().splitlines()]
            training.extend(dict(seed=seed,arm=arm,step=r['step'],conditional=r['conditional']['loss'],total=r['loss']) for r in rows)
            for k in (32,64):
                cases=[read(run/'evaluation'/f'{arm}_step2000_bank{b}_K{k}.json') for b in (0,1)]
                final[seed,arm,k]=cases
                aggregates=[c['aggregate'] for c in cases]
                row=dict(seed=seed,arm=arm,K=k,actual_steps=summary['actual_optimizer_steps'],source_occurrences=summary['source_occurrences'],
                    unique_sources=summary['unique_sources'],reference_exposures=summary['reference_exposures'],parameters=summary['parameter_count'],
                    compute_seconds=summary['optimizer_compute_seconds'],wall_seconds=summary['attempt_wall_seconds'],
                    sources_per_second=summary['attempt_sources_per_compute_second'],peak_MiB=summary['peak_allocated_bytes']/2**20)
                for metric,key in [('conditional','conditional'),('marginal','marginal'),('spread','spread')]:
                    for field in aggregates[0][key]:row[metric+'_'+field]=float(np.mean([a[key][field] for a in aggregates]))
                for field in ('correct','shuffled','gap'):
                    row['shuffle_'+field]=float(np.mean([np.mean([s[field] for s in p['per_session']]) for c in cases for p in c['permutations']]))
                for group in ('AU','VA','expression'):
                    for field in aggregates[0]['channels'][group]:row[group+'_'+field]=float(np.mean([a['channels'][group][field] for a in aggregates]))
                table.append(row)
        for key in ('initialization_hash','consumed_hashes','actual_optimizer_steps','parameter_count'):
            assert all(summaries[a][key]==summaries['C0'][key] for a in ARMS),key
        assert summaries['C0']['reference_exposures']==0
        assert summaries['C1']['reference_exposures']==summaries['C2']['reference_exposures']
        audits[str(seed)]={a:{k:v for k,v in s.items() if k!='checkpoints'} for a,s in summaries.items()}
    if len(seeds)>1:
        assert len({audits[str(s)]['C0']['initialization_hash'] for s in seeds})==len(seeds)
        for key in ('source_hash','noise_hash','reference_hash'):
            assert len({audits[str(s)]['C0']['consumed_hashes'][key] for s in seeds})==len(seeds)
    write(out/'matched_controls.json',dict(seeds=seeds,all_assertions_passed=True,details=audits))
    for name,rows in [('per_seed_metrics',table),('development_learning_curves',learning),('training_curves',training)]:
        with (out/(name+'.csv')).open('x') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    grouped={}
    metrics=['conditional_loss','conditional_cross_distance','conditional_self_distance','marginal_loss','marginal_cross','marginal_self','shuffle_correct','shuffle_shuffled','shuffle_gap','spread_within_input_descriptor_self','spread_between_input_descriptor_distance']
    for k in (32,64):
        grouped[str(k)]={}
        for arm in ARMS:
            rows=[r for r in table if r['arm']==arm and r['K']==k]
            grouped[str(k)][arm]={name:dict(values=[r[name] for r in rows],mean=float(np.mean([r[name] for r in rows])),std=float(np.std([r[name] for r in rows],ddof=1)) if len(rows)>1 else None) for name in metrics}
    robustness=[]
    for (seed,arm,k),cases in final.items():
        for bank,c in enumerate(cases):
            robustness.append(dict(seed=seed,arm=arm,K=k,bank=bank,conditional=c['aggregate']['conditional']['loss'],marginal=c['aggregate']['marginal']['loss'],shuffle_gaps={str(p['seed']):p['mean_gap'] for p in c['permutations']}))
    write(out/'noise_permutation_stability.json',robustness)
    write(out/'seed_summary.json',dict(seeds=seeds,definition='average two pre-fixed banks within model; shuffle averages three fixed derangements; sample std across training seeds',metrics=grouped))
    # Paired contrasts computed BEFORE bootstrap, within each fixed training seed.
    boot={};rng=np.random.default_rng(22201);ix=rng.integers(0,20,(10000,20))
    for seed in seeds:
        per={}
        for arm in ARMS:
            cases=final[seed,arm,32]
            ids=[r['session_id'] for r in cases[0]['per_session']]
            assert all([r['session_id'] for r in c['per_session']]==ids for c in cases)
            per[arm]={metric:np.mean([[s[metric]['loss'] for s in c['per_session']] for c in cases],axis=0) for metric in ('conditional','marginal')}
            per[arm]['gap']=np.mean([[s['gap'] for s in p['per_session']] for c in cases for p in c['permutations']],axis=0)
        for base in ('C0','C1'):
            for metric in ('conditional','marginal','gap'):
                delta=per['C2'][metric]-per[base][metric];lo,hi=np.quantile(delta[ix].mean(1),[.025,.975])
                boot[f'seed{seed}_C2_minus_{base}_{metric}']=dict(mean=float(delta.mean()),lower=float(lo),upper=float(hi),session_deltas=delta.tolist())
    write(out/'paired_session_bootstrap.json',dict(seed=22201,resamples=10000,unit='20 session clusters',
        caveat='conditional on each trained model and fixed development list/banks/shuffles; unknown participant links may induce additional dependencies; not training-seed inference',contrasts=boot))
    colors=dict(C0='#2563eb',C1='#d97706',C2='#059669')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for arm in ARMS:
        for ax,metric in zip(axes,('conditional','group')):
            for seed in seeds:
                rows=sorted([r for r in learning if r['arm']==arm and r['seed']==seed],key=lambda r:r['step'])
                ax.plot([r['step'] for r in rows],[r[metric] for r in rows],color=colors[arm],alpha=.3,lw=1)
            steps=sorted({r['step'] for r in learning});means=[np.mean([r[metric] for r in learning if r['arm']==arm and r['step']==s]) for s in steps]
            ax.plot(steps,means,'o-',color=colors[arm],label=arm)
            ax.set(xlabel='Actual optimizer steps',ylabel=metric+' ES (development)',title='Fixed bank 0, K=32');ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=180);fig.savefig(out/'learning_curves.svg');plt.close(fig)
    fig,axes=plt.subplots(len(seeds),1,figsize=(10,3*len(seeds)),squeeze=False)
    for row,seed in zip(axes,seeds):
        ax=row[0]
        for arm in ARMS:
            vals=[r['conditional'] for r in training if r['seed']==seed and r['arm']==arm]
            smoothed=np.convolve(vals,np.ones(100)/100,mode='valid');ax.plot(np.arange(100,len(vals)+1),smoothed,label=arm,color=colors[arm])
        ax.set(title=f'Seed {seed}; 100-step moving average',xlabel='Actual steps',ylabel='TRAIN paired ES');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'training_curves.png',dpi=180);plt.close(fig)
    for seed in seeds:
        for arm in ARMS:
            file=ROOT/f'seed_{seed}/evaluation/{arm}_step2000_bank0_K32_fixed_cases.npz';d=np.load(file)
            fig,axes=plt.subplots(3,4,figsize=(14,7),sharex=True)
            for col in range(4):
                n=int(d['lengths'][col]);x=np.arange(n)
                for ax,ch,label in zip(axes[:,col],(0,15,17),('AU0','Valence','Expression0')):
                    ax.plot(x,d['predictions'][col,:,:n,ch].T,color=colors[arm],alpha=.3,lw=.7)
                    ax.plot(x,d['targets'][col,:n,ch],color='black',lw=1.2,label='paired target')
                    ax.set_ylabel(label);ax.grid(alpha=.15)
                axes[0,col].set_title(str(d['clip_ids'][col]).split('/')[-1],fontsize=7)
                axes[-1,col].set_xlabel('Crop-relative frame')
            fig.suptitle(f'{arm}, seed {seed}, step 2000: 10 fixed noise samples; black = paired target')
            fig.tight_layout();fig.savefig(out/f'fixed_cases_seed{seed}_{arm}.png',dpi=150);plt.close(fig)
    lines=['# HiRP Phase 2.2 replicated development pilot','',
        'Baseline: acfdda2ef02ebf5340ed13c78084a5d1bc518043. Branch: agent/hirp-phase22-replicated-pilot. No later commits or uncommitted user files at entry. CODEX_PHASE22.md was not found; the full user-supplied task text was used. Review of Phase 2.1 logs was not an independent rerun of historical training.', '',
        '## Sampling and training contracts','',
        'HiRP22 stores standard_normal / conditional_gaussian as a public forward contract. Standard normal uses z=epsilon without executing the retained prior module. The same inherited sample, eval adapter and strict loader share this rule. Historical A0 migration requires the matching manifest and scale metadata. Missing semantics fail explicitly. No historical model, objective, checkpoint or run was rewritten.', '',
        'Actual GPU equivalence: forward/sample/adapter/roundtrip vs historical forward_a0 maximum error 0 for initial and C0/C1/C2 checkpoints at K=1/4/10/32. Candidate chunking maximum error 7.3761e-7 (declared atol 2e-6). Old bare sample mismatch is retained in public_contract_errors.json, not silently corrected in historical code.', '',
        'All three real-data GPU 8-step continuous versus 4+4 resumed runs had exactly identical model parameters. CPU regression verifies optimizer state and per-step rows, with dropout active. An additional cold-start audit found that the first PyTorch 2.1 AdamW construction consumed Python RNG (tensor states/results were identical). The final runner re-seeds only Python RNG after optimizer construction on a fresh run; resumed RNG is restored unchanged. rng_fix_resume/verification.json verifies model, optimizer, all RNG states and logs exactly, and unchanged model/optimizer versus original smoke. The original mismatch record remains in resume_gpu_optimizer_verification.json. Checkpoints retain all configs, scales, split hash, optimizer, global step, full Python/NumPy/CPU/CUDA RNG, and consumed schedule/noise hashes. Diagnostics preserve parameters and RNG. Premature stop reports completed=false. Budget extension uses a NEW manifest/run and explicit resume, never overwrites historical checkpoints.', '',
        'Fixed training: T=128, K=4, B=4, pre-norm/default projection, A0, FP32, AdamW lr=1e-4 wd=.01; C0 lambda=0, C1/C2 lambda=.1. Each seed drives initialization, sampler and independent Gaussian noise; source/reference substreams are separately derived. Full 2000-entry schedules are generated, not repeated from Phase 2. C0 does not load reference tensors. C1/C2 references are unique per step and matched. Source occurrences are IID with replacement under p(session)=n_s/N. The unchanged cross-bank estimator and equal-noise-index exclusion are preserved.', '',
        '## Completed results','',f'Completed training seeds: {seeds}. Each completed arm has requested=actual=training_rows=2000; 8000 source occurrences / 1660 = 4.819277 exposure multiples, not epochs. See matched_controls.json for full assertions and resource accounting.', '',
        'Main final numbers average two pre-fixed independent banks at K=32, and three pre-fixed same-session derangements for the shuffle columns. Per-seed values precede mean/std; raw per-example/session values remain in seed_*/evaluation. This is the repeatedly used 80-clip development set, NOT hidden or independent test.', '',
        '| Seed | Arm | Conditional ES | Cross | Self | Group ES | Group cross | Group self | Shuffle gap |', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in table:
        if r['K']==32:lines.append('| '+str(r['seed'])+' | '+r['arm']+' | '+' | '.join(f'{r[k]:.6f}' for k in ('conditional_loss','conditional_cross_distance','conditional_self_distance','marginal_loss','marginal_cross','marginal_self','shuffle_gap'))+' |')
    lines+=['','| Arm | Conditional ES mean ± seed std | Group ES mean ± seed std | Shuffle gap mean ± seed std |','|---|---:|---:|---:|']
    for arm in ARMS:
        d=grouped['32'][arm];lines.append('| '+arm+' | '+' | '.join(f"{d[k]['mean']:.6f} ± {d[k]['std']:.6f}" if d[k]['std'] is not None else f"{d[k]['mean']:.6f} (one seed)" for k in ('conditional_loss','marginal_loss','shuffle_gap'))+' |')
    lines+=['','![Development learning curves](learning_curves.png)','','![Training curves](training_curves.png)','',
        'K=64 values, channel ES/distribution/velocity/entropy, within/between spread, per-seed exposure and resource metrics are in per_seed_metrics.csv. All 45 fixed checkpoint evaluations are included, plus final bank/K stability cases. Fixed trajectory panels show 10 predeclared noise samples and paired targets for indices 0,20,40,60; they are illustrative development cases, not selected best samples.', '',
        '## Interpretation limits and remaining work','',
        'Direct paired gap_C2-gap_C1 and gap_C2-gap_C0 session bootstrap is in paired_session_bootstrap.json, separately for each training seed. Its intervals are conditional on model/development cases/evaluation draws; they do not replace seed replication. Mean/std over three training seeds is descriptive, not a strong population significance claim. Self is a legitimate score term; conditional and marginal scores must be interpreted together with shape diagnostics.', '',
        'Feature-byte hashes and split-local paths are verified; the extraction model/version is not independently established. The data audit found overlapping recording basenames across split/session labels, but no byte-identical files in the audited development inputs and full VAL listener reference set versus TRAIN. Participant/interaction crosswalk is unavailable and CSV identifiers use a different naming scheme. The unused VAL candidate list was sealed, but no independent confirmation claim or evaluation was made. Session resampling cannot rule out cross-session participant dependence.', '',
        'Not run: independent confirmation, official metrics/full-sequence generation or T=750 validation, budgets beyond 2000, lambda sweeps, architecture changes, push. Official adapter K=10 source-only contract is tested; local legacy loader/postprocessor sources and assets were inspected without importing/modifying metric code. Assets exist, but independent provenance and long-sequence integration remain unresolved. See official_adapter_audit.json.', '',
        'Full command logs, runtime source archives, initial/final history hashes and checkpoint fingerprints accompany the run. GPU compute_seconds measures synchronized optimization including scheduled gradient probes; wall seconds additionally include IO/diagnostics/checkpointing, not preparation/evaluation. No best-checkpoint selection or added budget used development outcomes.']
    (out/'PHASE22_REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
