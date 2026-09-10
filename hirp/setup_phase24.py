import json,random,sys,subprocess
from pathlib import Path
import numpy as np
import torch
from .phase24 import *
from .evaluate_phase23 import points


def main():
    plan=json.loads(PLAN.read_text());normalization(plan)
    write(ROOT/'initial_audit.json',dict(baseline='8d7e788a88b494882bed1f5565807796da5fb071',head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),status=subprocess.check_output(['git','status','--short'],text=True),original_worktree='clean before Phase24 creation',agents='No AGENTS.md in workspace/ancestors; .agents and .codex empty',historical_reports={str(p):sha256_file(p) for p in Path('runs').glob('phase*/**/*REPORT.md') if 'phase24' not in str(p)}))
    candidates=[]
    for seed in (123,42,2026):
        for p in points(OLD,seed):
            if p['lambda_group'] not in (0.,.03,.1):continue
            model,m=load_checkpoint(p['checkpoint']['path'])
            row=dict(seed=seed,arm=p['arm'],lambda_group=p['lambda_group'],stored_lambda_group=m['train_config']['lambda_group'],steps=2000,checkpoint=p['checkpoint']['path'],checkpoint_sha256=sha256_file(p['checkpoint']['path']),origin=p['origin'],**{k:m[k] for k in ('prior_mode','output_config','config','scales')})
            validate_candidate(row,m);candidates.append(row);del model,m
    assert len(candidates)==15
    write(ROOT/'candidate_manifest.json',candidates)
    (ROOT/'noise_banks').mkdir()
    banks=[]
    for seed in range(24001,24009):
        x=torch.randn(1,64,32,generator=torch.Generator().manual_seed(seed)).numpy();path=ROOT/'noise_banks'/f'{seed}.npy'
        with path.open('xb') as f:np.save(f,x)
        banks.append(dict(seed=seed,path=str(path),sha256=sha256_file(path)))
    write(ROOT/'mc_protocol.json',dict(banks=banks,K=64,T=128,max_banks=8,models=15,new_training_steps=0,plan_path=str(PLAN),plan_sha256=sha256_file(PLAN),aggregation='per-bank U-statistic then mean 8; shared noise index excluded for all source pairs in group self',paired_interval='Student t df7 Monte Carlo CI across independent paired bank differences; no population/seed uncertainty',halves=[[24001,24002,24003,24004],[24005,24006,24007,24008]],relative_C0_screen=.01,selection='fixed before new evaluations'))
    # Use actual official constructor's video population/order; no source from confirmation is generated.
    sys.path.insert(0,str(LEGACY))
    from regnn.eval_conditional_regnn_official_test import initialize_official_hydra_runtime
    initialize_official_hydra_runtime(LEGACY)
    from dataset.react_2025 import ReactionDataset
    dataset=ReactionDataset(root_dir=str(Path('data').resolve()),split='val',clip_length=750,load_video_s=False,load_video_l=False)
    lookup={str(p):i for i,p in enumerate(dataset.speaker_path_list)};sources=[]
    full=json.loads((OLD/'task_development_full/manifest.json').read_text())['sources']
    for i,clip in enumerate(plan['clip_ids']):
        idx=lookup[clip];paired=dataset.listener_path_list[idx];pool=dataset.gt_path_list[idx];seed=24100+i
        selected=official_selection(paired,pool,random.Random(seed));targets=[]
        for rank,ref in enumerate(selected):
            path=Path('data/val/facial-attributes')/ref.with_suffix('.npy')
            targets.append(dict(rank=rank,path=str(path),sha256=sha256_file(path),role=ref.parts[0],session=ref.parts[1],length=len(np.load(path,mmap_mode='r')),source='paired' if rank==0 else 'same-session official video pool',selection_seed=seed,occurrence_weight=.1))
        assert len(targets)==10 and all(t['role']=='listener' and t['session']==clip.split('/')[1] for t in targets)
        sources.append(dict(**full[i],targets=targets,pool_order=[str(x) for x in pool],selection_seed=seed,official_dataset_index=idx))
    write(ROOT/'multitarget_development_manifest.json',dict(population='Development-80, official test selection rule applied to VAL; not challenge/hidden test',sources=sources,selection='paired +9; random.sample if pool excluding paired >=9 else random.choices; singleton may repeat paired; no extra dedup',normalization=plan['normalization'],K=10,noise=plan['banks'][0],noise_indices=list(range(10)),time_chunk=750,processor_length=1000,processor_seed=24201,metrics=['FRC','FRVar','smse','temporal_smse','TLCC'],FRD='not budgeted: exact all-pair variable-length DTW 16x80x100 is quadratic; no surrogate substituted',loader=dict(path=inspect.getfile(ReactionDataset),sha256=sha256_file(inspect.getfile(ReactionDataset)))))
    write(ROOT/'dependency_snapshot.json',dict(hirp=dependencies(),legacy=legacy_snapshot(),policy='source files only, no identity/result/log circular dependency; offline reporting/setup/tests excluded'))
    print('FROZEN',len(candidates),'models',len(banks),'banks',len(sources),'sources',flush=True)


if __name__=='__main__':main()
