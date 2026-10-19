"""Matched conditional-assignment vs shuffled-condition teacher experiment."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from coreset_reactor.teacher_cache import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def comparison(result, baseline):
    for key in ('clip_id', 'target_cache_key'):
        assert [r[key] for r in result['rows']] == [r[key] for r in baseline['rows']]
    assert len(result['rows']) == 571
    summary = {}
    # Session-cluster bootstrap, not treating contexts in a session as IID.
    sessions = np.array([r['clip_id'].split('/')[1] for r in result['rows']])
    groups = sorted(set(sessions))
    rng = np.random.default_rng(20260925)
    draws = rng.integers(0, len(groups), size=(4000, len(groups)))
    for key in ('FRC', 'exact_FRD', 'FRDiv'):
        values = lambda d: np.asarray([sum(r['frc']) if key == 'FRC' else r[key] for r in d['rows']])
        a, b = values(result), values(baseline)
        delta = a-b
        sums = np.array([delta[sessions==s].sum() for s in groups])
        counts = np.array([(sessions==s).sum() for s in groups])
        bootstrap = sums[draws].sum(1)/counts[draws].sum(1)
        summary[key] = {'delta_mean': float(delta.mean()),
                        'session_bootstrap_delta_ci95': np.quantile(bootstrap, [.025,.975]).tolist(),
                        'context_delta_quantiles_p10_p50_p90': np.quantile(delta, [.1,.5,.9]).tolist(),
                        'context_improvement_fraction': float(np.mean(delta < 0 if key=='exact_FRD' else delta > 0)),
                        'per_session_delta': {s: float(delta[sessions==s].mean()) for s in groups}}
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--asset-mam-root', type=Path, required=True)
    p.add_argument('--convergence-root', type=Path, required=True)
    p.add_argument('--target-cache', type=Path, required=True)
    p.add_argument('--gpus', type=int, nargs='+', default=[1,5,6])
    args = p.parse_args()
    for k in ('root','data_root','asset_mam_root','convergence_root','target_cache'):
        setattr(args,k,getattr(args,k).resolve())
    diagnostic = json.loads((args.root/'summary.json').read_text())
    if diagnostic['arms']['conditional']['fraction_nonanchor_slots_changed'] < .01:
        raise ValueError('No meaningful assignment intervention; do not spend training budget')
    run = args.root/'pilot600'
    run.mkdir(exist_ok=False)
    expected = json.loads((args.convergence_root/'arms/T1/rank_00/conditional-step000600-seed1.json').read_text())
    env0 = dict(os.environ, MAM_REACTOR_ROOT=str(args.asset_mam_root),
                PYTHONPATH=f'{ROOT/"mam_reactor"}:{ROOT}', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    names = ['coreset_reactor/conditional_slot_manifest.py', 'coreset_reactor/run_conditional_slot_pilot.py',
             'coreset_reactor/eval_expert_student_fast.py', 'coreset_reactor/fast_frc.py',
             'coreset_reactor/oracle_select.py', 'coreset_reactor/evaluate.py',
             'mam_reactor/regnn/train_conditional_regnn.py', 'mam_reactor/regnn/conditional_data.py',
             'mam_reactor/regnn/conditional_model.py']
    source_hashes = {n: sha256_file(ROOT/n) for n in names}
    for name in names:
        dst = run/'source_snapshot'/name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, dst)
    provenance = {'git_sha': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                  'source_hashes': source_hashes, 'manifest_summary_sha256': sha256_file(args.root/'summary.json'),
                  'arms': ['conditional','shuffled'], 'steps':600,'seed':1,'batch_size':32,
                  'formal_schedule_epochs':50, 'gpus':args.gpus,
                  'new_updates_per_arm':5400, 'total_updates_with_reused_paired_anchor':6000,
                  'loss': 'unchanged Mam MSE+(1-CCC)+.05 velocity; relative_time_masked',
                  'q0': 'exact T1_600 paired anchor reused',
                  'eval': 'full VAL571 all-session GT; eval_seed1234; not official sampled10',
                  'primary_numeric_target': 'FRC>=1.10; .17<=FRDiv<=.20; FRD<=T1_600',
                  'mechanism_test': 'conditional improves quality vs T1 AND shuffled without losing target membership',
                  'selection': 'one TRAIN-frozen assignment formula; no validation hyperparameter search',
                  'speed_caveat':'concurrent GPU/CPU workloads; not an isolated benchmark'}
    (run/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')

    def execute(command,gpu,log):
        log.with_suffix('.command.json').write_text(json.dumps(command,indent=2)+'\n')
        with log.open('x') as f:
            subprocess.run(command,cwd=ROOT,env=dict(env0,CUDA_VISIBLE_DEVICES=str(gpu)),
                           stdout=f,stderr=subprocess.STDOUT,check=True)

    def train_queue(gpu,tasks):
        for arm,rank in tasks:
            directory=run/arm/f'rank_{rank:02d}'
            directory.mkdir(parents=True)
            manifest=args.root/'manifests'/arm/f'rank_{rank:02d}.json'
            assert sha256_file(manifest)==diagnostic['arms'][arm]['manifest_hashes'][str(rank)]
            command=[sys.executable,'-m','regnn.train_conditional_regnn',
                     '--data-dir',str(args.data_root),'--run-dir',str(directory),
                     '--epochs','50','--batch-size','32','--workers','8','--precision','bf16',
                     '--seed','1','--fixed-pair-seed','1','--fixed-pair-rank',str(rank),
                     '--fixed-pair-count','10','--fixed-pair-manifest',str(manifest),
                     '--fixed-pair-alignment-policy','relative_time_masked',
                     '--graph-layers','2','--prediction-head','linear',
                     '--max-train-steps','600','--snapshot-steps','600','--save-every','5','--print-every','100']
            execute(command,gpu,directory/'train.log')
            audit=json.loads((directory/'conditional-step000600-seed1.json').read_text())
            for key in ('initial_state_sha256','observed_source_schedule_sha256','global_step','parameter_count'):
                if audit[key]!=expected[key]:
                    raise ValueError(f'Matched audit failed: {arm}/{rank}/{key}')
            print(f'{arm} q{rank}: 600 steps, initialization/source/crop audit passed',flush=True)

    started=time.monotonic()
    tasks=[(a,r) for a in ('conditional','shuffled') for r in range(1,10)]
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures=[pool.submit(train_queue,gpu,tasks[i::len(args.gpus)]) for i,gpu in enumerate(args.gpus)]
        for task in futures: task.result()
    provenance['training_wall_seconds']=time.monotonic()-started
    provenance['matched_training_checks']='passed'
    provenance['audits']={f'{a}/q{r}':json.loads((run/a/f'rank_{r:02d}/conditional-step000600-seed1.json').read_text())
                         for a,r in tasks}
    (run/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')

    def evaluate(arm,gpu):
        checkpoints=[args.convergence_root/'arms/T1/rank_00/conditional-step000600-seed1.pth']+[
            run/arm/f'rank_{r:02d}/conditional-step000600-seed1.pth' for r in range(1,10)]
        for stage in ('fast','full'):
            output=run/arm/f'val571_{stage}.json'
            command=[sys.executable,'-m','coreset_reactor.eval_expert_student_fast',
                     '--teacher-checkpoints',*map(str,checkpoints),'--data-root',str(args.data_root),
                     '--asset-mam-root',str(args.asset_mam_root),'--device','cuda:0',
                     '--eval-seed','1234','--output',str(output),'--target-cache',str(args.target_cache)]
            if stage=='full': command+=['--exact-frd','--metric-workers','16']
            execute(command,gpu,output.with_suffix('.log'))
            print(json.dumps({'arm':arm,'stage':stage,'metrics':json.loads(output.read_text())['metrics']}),flush=True)
        return json.loads((run/arm/'val571_full.json').read_text())

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={a:pool.submit(evaluate,a,gpu) for a,gpu in zip(('conditional','shuffled'),args.gpus)}
        results={a:job.result() for a,job in futures.items()}
    baselines={a:json.loads((args.convergence_root/f'eval/{a}_s000600_val571_full.json').read_text()) for a in ('T0','T1')}
    summary={'metrics':{a:d['metrics'] for a,d in {**baselines,**results}.items()},
             'vs_T1':{a:comparison(d,baselines['T1']) for a,d in results.items()},
             'conditional_vs_shuffled':comparison(results['conditional'],results['shuffled']),
             'numeric_goal':{a:d['metrics']['FRC']>=1.10 and .17<=d['metrics']['FRDiv']<=.20 and
                             d['metrics']['exact_FRD']<=baselines['T1']['metrics']['exact_FRD'] for a,d in results.items()},
             'wall_seconds':time.monotonic()-started}
    (run/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
