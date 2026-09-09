"""Freeze the finite campaign, population contract and exact checkpoint reuse."""
import json,subprocess,hashlib
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from .train_phase22 import write as _write,TrainConfig,schedule
from .session_data import SessionPopulation
from .phase15_audit import sha256_file,canonical_hash
from . import HiRPConfig

ROOT=Path('runs/phase23/tradeoff_v1')
OLD=Path('runs/phase22/replicated_v1')


def write(path,value):
    if path.exists():
        if path.name=='initial_audit.json':return
        if canonical_hash(json.loads(path.read_text()))!=canonical_hash(value):raise ValueError('setup identity mismatch: '+str(path))
        return
    _write(path,value)


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    write(ROOT/'initial_audit.json',dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),status=subprocess.check_output(['git','status','--short'],text=True),
        later_commits=subprocess.check_output(['git','log','135220fae7f9018eec8f8119a72b4e2d074f713b..HEAD','--oneline'],text=True),
        historical_files={str(p):sha256_file(p) for base in ('phase1','phase15','phase2','phase21','phase22') for p in Path('runs',base).rglob('*') if p.is_file()}))
    plan_path=OLD/'evaluation_plan/manifest.json';plan=json.loads(plan_path.read_text());candidate_path=OLD/'evaluation_plan/confirmation_candidate_audit.json';candidate=json.loads(candidate_path.read_text())
    write(ROOT/'evaluation_population_contract.json',dict(
        development=dict(name='Development-80',approved=True,scope='paired development comparison only; repeated-use, not participant independent',ids=plan['clip_ids'],plan_path=str(plan_path),sha256=sha256_file(plan_path)),
        confirmation=dict(status='pending',approved=False,candidate_manifest=str(candidate_path),candidate_sha256=sha256_file(candidate_path),candidate_ids=candidate['candidate_unused_val'],executed=False),
        identity_mapping=dict(raw_interaction='Camera basename not established as globally unique interaction ID',speaker_id=None,listener_id=None,role='speaker/listener directory role only',session='directory label, not proven participant identity',
          crops_role_grouping='one source-centred crop per Development-80 source; reverse roles not added; underlying cross-session/camera relationship unknown',
          csv_crosswalk='no authoritative Camera-to-CSV mapping found; no fuzzy matching inferred',
          cross_split='basename overlaps and different bytes do not establish or exclude same interaction'),
        evidence=[dict(path=str(p),sha256=sha256_file(p)) for p in [OLD/'split_overlap_details.json',OLD/'feature_provenance_audit.json',candidate_path]],
        searched_metadata=['data/{train,val,test}.csv','data/data_indices.csv','data/{train,val,test}/whisper_batch_log.csv'],
        features='existing wav2vec-width audio / attributes / FaceVerse-normalized coefficients; extraction version not independently established',
        weights=dict(source='uniform 4 fixed sources within session, macro 20 sessions',reference='uniform full session VAL listener reference pool',same_finite_population=False),
        sensitivity='six exhaustive 2-of-4 source subsets within Development-80, unchanged full reference pool; no candidate confirmation inputs consumed'))
    protocol=dict(seeds=[123,42,2026],arms=['C0','C1','C2'],lambdas=[.03,.1,.3],new_lambdas=[.03,.3],max_new_training_runs=12,
        max_steps=2000,T=128,K_train=4,B=4,lr=1e-4,weight_decay=.01,prior_mode='standard_normal',head='pre-norm/default',
        main_K=32,stability_K=64,bank_files=plan['banks'],permutations=plan['permutations'],eval_steps=[128,500,1000,1500,2000],
        final_aggregation='mean two pre-fixed banks; three shuffles averaged separately; training seeds are only replication unit',
        curve_aggregation='bank0 only, explicit in CSV/plot labels',engineering_tolerance=dict(relative_C0_conditional_ES=.01,
        status='post-Phase22 development choice, frozen before new lambda outputs; not statistical equivalence'),
        auxiliary=dict(covariance='population raw-unit covariance, no fitted validation scales',autocorrelation_lags=[1,4,16],source_subset_repetitions=6),
        official_integration=dict(population='Development-80 only',K=10,chunk_T=750,global_noise='same sample-index epsilon for all time chunks',
            metrics=['FRC','FRVar','smse','temporal_smse','TLCC'],targets='paired listener only; not official ten-appropriate-target protocol',
            processor='unchanged legacy target alignment processor; no target-based prediction selection',external_MAM=False))
    write(ROOT/'protocol.json',protocol)
    (ROOT/'PROTOCOL.md').write_text('''# Phase 2.3 locked finite protocol

Question: does cross-input marginal matching improve group fit under a controlled conditional-quality constraint?

Keep HiRP22 pre-norm/default, A0, the original all-sample conditional ES and C1/C2 group estimators. Three seeds 123/42/2026, 2000 actual steps, T128/K4/B4, AdamW 1e-4/wd .01. Reuse C0 and lambda .1 after fingerprint/config/schedule checks. At most 12 new scratch trainings: C1/C2 × lambda .03/.3 × three seeds. Finish seed 123 before other seeds. No extensions or checkpoint mixtures.

Final frontier: both fixed banks averaged at K32, K64 stability. Curves: bank0 only, explicitly labelled. All discrete points remain visible. Relative conditional ES ≤ C0×1.01 is a post-Phase22 engineering screen, not preregistered historical evidence or statistical equivalence. Report individual seed differences, uncertainty and failure to match; do not interpolate points.

Development-80 remains the only source evaluation population. Confirmation candidates stay pending and unused. Six exhaustive 2-of-4 source mixtures per session assess finite source-subset sensitivity against the same full reference pool. These are not independent datasets/seeds. Raw covariance and lag 1/4/16 autocorrelations are fixed evaluation-only summaries with no validation scale fitting.

Task integration uses the existing nine checkpoints first: K10, full source records, zero-padded T750 chunks, fixed global epsilon per sample index across all time chunks. Entire valid source sequences are exported; no silent time truncation. Time-chunk and full-context predictions need not agree. The old official chunk convention creates an extra zero-valid tail at exact multiples: retain its padding layout but do not call the generator on zero-valid chunks; trim that tail on reassembly. Attribute metric implementations and target alignment processor stay unchanged. Targets are the single paired development listener; therefore this is a development full-sequence protocol, not the official ten-appropriate-target benchmark. FRC metric matching is used as implemented, never to rerank predictions. FRD/rendering/video-realism metrics and external MAM comparison are outside this bounded initial integration.

Cache reuse requires the complete content/implementation identity and result hash. Missing identity, different inputs/configuration or tampering fails. Full reruns use a new evaluation-name directory. Repeating a fully completed identical invocation is a no-op.
''')
    reuse=[];pool=SessionPopulation(Path('data'),'train')
    for seed in protocol['seeds']:
        run=OLD/f'seed_{seed}';m=json.loads((run/'manifest.json').read_text())
        cfg=TrainConfig(init_seed=seed,sampler_seed=seed,noise_seed=seed)
        rec,noise,hashes=schedule(pool,cfg,32)
        assert hashes==m['schedule'] and np.array_equal(noise.numpy(),np.load(run/'noise.npy'))
        assert json.loads(json.dumps(asdict(cfg)))==m['config']
        for row in m['data_files']+m['normalization']:assert sha256_file(row['path'])==row['sha256']
        for arm in protocol['arms']:
            summary=json.loads((run/arm/'attempt_000/summary.json').read_text());assert summary['completed'] and summary['actual_optimizer_steps']==2000
            ck=summary['checkpoints'][-1];assert sha256_file(ck['path'])==ck['sha256']
            saved=torch.load(ck['path'],map_location='cpu',weights_only=True)
            assert saved['prior_mode']=='standard_normal' and saved['output_config']==m['output_config'] and canonical_hash(saved['config'])==canonical_hash(m['model_config'])
            assert saved['train_config']['init_seed']==seed and saved['global_step']==len(saved['training_rows'])==2000
            assert saved['consumed_hashes']==hashes and saved['initialization_hash']==summary['initialization_hash']
            reuse.append(dict(seed=seed,arm=arm,lambda_group=0 if arm=='C0' else .1,checkpoint=ck,
                              run=str(run),manifest_sha256=sha256_file(run/'manifest.json'),initialization_hash=saved['initialization_hash'],checks='config/data/initialization/schedule/noise/step/scales verified; same unchanged train_phase22 tensor objective'))
    write(ROOT/'reused_checkpoints.json',reuse)


if __name__=='__main__':main()
