"""REACT2024 replication of frozen mode-prior teachers and matched student KD.

Keep the existing T1 target manifests, paired anchor, 600-step teacher budget,
and 10-epoch student recipes. No hyperparameter sweep or inference rescaling.
Native REACT2024 neighbour-GT scores are NOT interchangeable with this protocol.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--old-teacher-root', type=Path, required=True)
    parser.add_argument('--asset-mam-root', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs=3, default=[1, 5, 6])
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=False)
    started = time.time()
    env = dict(os.environ, MAM_REACTOR_ROOT=str(args.asset_mam_root),
               PYTHONPATH=f'{ROOT / "mam_reactor"}:{ROOT}', OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')

    def stage(name, **kwargs):
        write(args.root / 'status.json', {'stage': name, 'pid': os.getpid(),
              'started_unix': started, 'updated_unix': time.time(), **kwargs})
        print(name, flush=True)

    def execute(command, gpu, log):
        log.parent.mkdir(parents=True, exist_ok=True)
        write(log.with_suffix('.command.json'), command)
        with log.open('x') as f:
            subprocess.run(command, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)),
                           stdout=f, stderr=subprocess.STDOUT, check=True)

    try:
        stage('preflight')
        configs = [read(args.old_teacher_root / f'rank_{r:02d}/config.json') for r in range(10)]
        cfg = configs[0]
        data_root = Path(cfg['training_config']['data_dir'])
        allowlist = cfg['session_allowlist']
        expected_train = cfg['dataset_size']
        expected_val = len(list((data_root / 'val/facial-attributes/speaker').glob('*/*.npy')))
        if expected_val != 1124 or expected_train != 6340:
            raise ValueError('Unexpected REACT2024 split sizes')
        original_split = Path(cfg['training_config']['session_split_manifest'])
        split = args.root / 'session_split.json'
        shutil.copy2(original_split, split)
        old, manifests = [], []
        for rank, c in enumerate(configs):
            for key in ('dataset_size', 'session_allowlist', 'model_config', 'parameter_count'):
                if c[key] != cfg[key]:
                    raise ValueError(f'Old teacher mismatch: {rank}/{key}')
            t = c['training_config']
            for key, value in {'batch_size':32, 'epochs':50, 'max_train_steps':600,
                               'precision':'bf16', 'seed':1, 'lr':.0001,
                               'fixed_pair_alignment_policy':'relative_time_masked'}.items():
                if t[key] != value:
                    raise ValueError(f'Unexpected old training config {rank}/{key}')
            history = [json.loads(line) for line in (args.old_teacher_root /
                        f'rank_{rank:02d}/train_metrics.jsonl').read_text().splitlines() if line.strip()]
            if history[-1]['global_step'] != 600:
                raise ValueError('Historical baseline not 600 updates')
            paths = list((args.old_teacher_root / f'rank_{rank:02d}').glob('conditional-epoch*.pth'))
            if len(paths) != 1:
                raise ValueError('Ambiguous old checkpoint')
            dest = args.root / 'frozen_T1' / f'rank_{rank:02d}'
            dest.mkdir(parents=True)
            ckpt = dest / paths[0].name
            shutil.copy2(paths[0], ckpt)
            manifest = dest / 'manifest.json'
            shutil.copy2(t['fixed_pair_manifest'], manifest)
            shutil.copy2(args.old_teacher_root / f'rank_{rank:02d}/config.json', dest / 'config.json')
            old.append(ckpt)
            manifests.append(manifest)
        sources = [Path(__file__), *[ROOT / 'coreset_reactor' / n for n in (
            'train_latent_reference_teacher.py', 'train_mode_reference_teacher.py',
            'teacher_cache.py', 'train_expert_student.py', 'relation_distillation.py',
            'run_mode_reference_student.py',
            'eval_expert_student_fast.py', 'eval_random_reference_frc.py', 'fast_frc.py',
            'evaluate.py', 'paired_data.py', 'oracle_select.py')]]
        sources += list((ROOT / 'mam_reactor/regnn').glob('*.py'))
        sources += list((ROOT / 'mam_reactor/framework/metrics').rglob('*.py'))
        source_hashes = {}
        for p in sources:
            relative = p.relative_to(ROOT)
            dest = args.root / 'source_snapshot' / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
            source_hashes[str(relative)] = digest(p)
        provenance = {'dataset':'REACT2024 converted_t1_v2', 'data_root':data_root,
            'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'source_hashes':source_hashes, 'old_teacher_hashes':[digest(p) for p in old],
            'manifest_hashes':[digest(p) for p in manifests], 'split_sha256':digest(split),
            'train_contexts':expected_train, 'val_contexts':expected_val,
            'training_eligibility':'Same T1 session allowlist; excluded RECOLA sessions with <10 GT',
            'teacher_steps':600, 'paired_anchor':'Frozen REACT2024 T1 q0, unchanged',
            'teacher_objective':'fixed MSE + mode-prior reference CCC + unchanged masked velocity',
            'temperature':.05, 'prior_effective_count':8, 'hyperparameter_search':False,
            'student_epochs':10, 'student_steps':math.ceil(expected_train/4)*10,
            'student_arms':['old_contrast','new_basic','new_contrast'],
            'student_initialization':'same REACT2024 T1 paired anchor for every arm',
            'evaluation':'full VAL1124 center750 all-session opposite-role GT; AU round; exact DTW-FRD',
            'native_2024_metrics':'not this experiment; no neighbour-GT or FRDvs equivalence claimed',
            'matching_caveat':'Historical teachers have no init/source hash. Config/manifest matched; new teachers and students record exact hashes.',
            'training_cost':'same teacher updates; all-reference CCC costs more than one-reference CCC',
            'reference_scaling':'TRAIN-only mean/std50D, including all TRAIN reference groups',
            'smoke_updates':2, 'inference':'no GT/no scaling/no noise; ten outputs'}
        write(args.root / 'provenance.json', provenance)

        def teacher_command(rank, directory, steps):
            return [sys.executable,'-m','coreset_reactor.train_mode_reference_teacher',
                '--data-dir',str(data_root),'--run-dir',str(directory),'--epochs','50',
                '--batch-size','32','--workers','8','--precision','bf16','--seed','1',
                '--fixed-pair-seed','1','--fixed-pair-rank',str(rank),'--fixed-pair-count','10',
                '--fixed-pair-manifest',str(manifests[rank]),
                '--fixed-pair-alignment-policy','relative_time_masked','--graph-layers','2',
                '--prediction-head','linear','--max-train-steps',str(steps),
                '--snapshot-steps',str(steps),'--save-every','5','--print-every','100',
                '--reference-temperature','.05','--reference-effective-count','8',
                '--session-split-manifest',str(split),'--session-split-name','B_fit']

        stage('two_step_teacher_smoke')
        execute(teacher_command(1,args.root/'smoke',2),args.gpus[0],args.root/'smoke/train.log')
        stage('training_nine_mode_reference_teachers')
        def teacher_queue(gpu, ranks):
            for rank in ranks:
                d = args.root / 'teachers' / f'rank_{rank:02d}'
                execute(teacher_command(rank,d,600),gpu,d/'train.log')
                print(f'teacher rank {rank} complete',flush=True)
        with ThreadPoolExecutor(max_workers=3) as pool:
            tasks = [pool.submit(teacher_queue,g,list(range(i+1,10,3))) for i,g in enumerate(args.gpus)]
            for task in tasks:
                task.result()
        audits = [read(args.root/'teachers'/f'rank_{r:02d}/conditional-step000600-seed1.json') for r in range(1,10)]
        for key in ('initial_state_sha256','observed_source_schedule_sha256','global_step','parameter_count'):
            if len({a[key] for a in audits}) != 1:
                raise ValueError(f'New teacher matching failed: {key}')
        write(args.root/'teacher_audits.json',audits)
        new = [old[0]] + [args.root/'teachers'/f'rank_{r:02d}/conditional-step000600-seed1.pth' for r in range(1,10)]
        target_cache = args.root/'val1124_gt_cache'

        def evaluate(label, paths, gpu, is_student=False):
            d = args.root/'eval'
            for kind in ('fast','full'):
                output = d/f'{label}_{kind}.json'
                command = [sys.executable,'-m','coreset_reactor.eval_expert_student_fast',
                    *(['--checkpoint',str(paths[0])] if is_student else ['--teacher-checkpoints',*map(str,paths)]),
                    '--data-root',str(data_root),'--asset-mam-root',str(args.asset_mam_root),
                    '--output',str(output),'--target-cache',str(target_cache),'--device','cuda:0',
                    '--eval-seed','1234']
                if kind=='full': command+=['--exact-frd','--metric-workers','12']
                execute(command,gpu,output.with_suffix('.log'))
                if read(output)['metrics']['contexts'] != expected_val:
                    raise ValueError('Incomplete validation set')
                print(json.dumps({'arm':label,'stage':kind,'metrics':read(output)['metrics']}),flush=True)

        def cache(label, checkpoints, gpu):
            execute([sys.executable,'-m','coreset_reactor.teacher_cache',
                '--data-root',str(data_root),'--teacher-checkpoints',*map(str,checkpoints),
                '--output-dir',str(args.root/f'{label}_cache'),'--device','cuda:0',
                '--batch-size','4','--workers','4','--session-allowlist',*allowlist],gpu,args.root/f'{label}_cache.log')
        def teacher_evaluations():
            evaluate('old_teacher',old,args.gpus[2])
            evaluate('new_teacher',new,args.gpus[2])
        stage('generating_train_caches_and_evaluating_teachers')
        with ThreadPoolExecutor(max_workers=3) as pool:
            tasks=[pool.submit(cache,'old',old,args.gpus[0]),pool.submit(cache,'new',new,args.gpus[1]),
                   pool.submit(teacher_evaluations)]
            for task in tasks: task.result()
        from coreset_reactor.run_mode_reference_student import view_rows
        old_cache,new_cache=[args.root/f'{label}_cache/manifest.json' for label in ('old','new')]
        if list(view_rows(old_cache))!=list(view_rows(new_cache)):
            raise ValueError('Old/new teacher cache source views mismatch')
        if any(read(p)['contexts']!=expected_train for p in (old_cache,new_cache)):
            raise ValueError('Incorrect TRAIN eligibility filter')
        stage('training_three_matched_students_then_full_val')
        def student(arm,gpu):
            d=args.root/'students'/arm
            manifest=old_cache if arm=='old_contrast' else new_cache
            basic=arm=='new_basic'
            execute([sys.executable,'-m','coreset_reactor.train_expert_student',
                '--data-root',str(data_root),'--cache-manifest',str(manifest),'--run-dir',str(d),
                '--device','cuda:0','--epochs','10','--batch-size','4','--workers','4',
                '--lr','.0001','--seed','1','--expert-embedding-dim','64',
                '--initialize-backbone',str(old[0]),'--relation-objective',
                'distance' if basic else 'centered_residual','--relation-grad-ratio',
                '0' if basic else '.25','--print-every','100','--save-every','5'],gpu,args.root/f'{arm}_train.log')
            evaluate(arm,[d/'student.pth'],gpu,True)
        arms=['old_contrast','new_basic','new_contrast']
        with ThreadPoolExecutor(max_workers=3) as pool:
            tasks=[pool.submit(student,a,g) for a,g in zip(arms,args.gpus)]
            for task in tasks: task.result()
        prov=[read(args.root/'students'/a/'provenance.json') for a in arms]
        for key in ('initial_student_state_sha256','observed_source_schedule_sha256','initialization_checkpoint_hash'):
            if len({p[key] for p in prov})!=1: raise ValueError(f'Student matching failed: {key}')
        for arm in arms:
            if read(args.root/'students'/arm/'train_metrics.json')[-1]['step']!=math.ceil(expected_train/4)*10:
                raise ValueError('Incorrect student update budget')
        labels=['old_teacher','new_teacher',*arms]
        files=[args.root/'eval'/f'{label}_full.json' for label in labels]
        stage('random_gt_frc_100_draws_full_val1124')
        execute([sys.executable,'-m','coreset_reactor.eval_random_reference_frc',
            '--evaluations',*map(str,files),'--names',*labels,'--target-cache',str(target_cache),
            '--output-dir',str(args.root/'random_frc100'),'--draws','100','--expected-contexts','1124'],
            args.gpus[0],args.root/'random_frc100.log')
        metrics={label:read(path)['metrics'] for label,path in zip(labels,files)}
        summary={'dataset':'REACT2024','contexts':expected_val,'metrics':metrics,
                 'matched_student_checks':'passed','wall_seconds':time.time()-started}
        write(args.root/'summary.json',summary)
        lines=['# REACT2024 mode-reference teacher → student','',
            'Full VAL1124; center750; all-session opposite-role GT; AU round; exact DTW-FRD.',
            'Not native neighbour-GT/FRDvs, not official TEST. Single seed; no inference scaling.', '',
            '| Arm | FRC | exact FRD | FRDiv | FRVar |','|---|---:|---:|---:|---:|']
        for arm,m in metrics.items():
            lines.append(f"| {arm} | {m['FRC']:.6f} | {m['exact_FRD']:.6f} | {m['FRDiv']:.6f} | {m['FRVar']:.6f} |")
        lines+=['','Teacher pilot: 600 updates, unchanged T1 target manifests and paired anchor.',
            'Students: matched initialization/source views/order, 10 epochs / 15850 updates.',
            'All-reference teacher CCC adds compute; equal updates are not equal training time.',
            'See per-arm config, source snapshots, and random_frc100 for provenance, speed and random-reference FRC.',
            'Historical teacher lacks initial/source hashes; do not claim bitwise matched retraining.','']
        (args.root/'REACT2024_MODE_REFERENCE_DISTILLATION.md').write_text('\n'.join(lines))
        stage('complete',summary=summary)
    except Exception as error:
        stage('failed',error=repr(error))
        raise


if __name__=='__main__':
    main()
