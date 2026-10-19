"""Distill a user-accepted teacher tradeoff without rewriting its failed gate.

Reuse the count8 student recipe exactly, recalibrating only TRAIN target scales
as required by the new teacher cache. No teacher retraining or VAL labels in KD.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from coreset_reactor.teacher_cache import sha256_file
from coreset_reactor.run_mode_reference_student import view_rows
from coreset_reactor.run_reference_count_ablation import quality_gate

ROOT=Path(__file__).resolve().parents[1]


def read(path): return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-run',type=Path,required=True)
    p.add_argument('--teacher-run',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=5)
    args=p.parse_args()
    base=args.baseline_run.resolve(); teacher_root=args.teacher_run.resolve()
    args.root=args.root.resolve(); args.root.mkdir(parents=True,exist_ok=False)
    started=time.time()
    def write(path,value):
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,default=str)+'\n');temp.replace(path)
    def stage(name,**extra):
        write(args.root/'status.json',{'stage':name,'pid':os.getpid(),'started_unix':started,
              'updated_unix':time.time(),**extra});print(name,flush=True)
    try:
        stage('preflight')
        teacher=read(teacher_root/'eval/teacher_full.json')
        checkpoints=[Path(path) for path in teacher['checkpoint']]
        if len(checkpoints)!=10 or [sha256_file(c) for c in checkpoints]!=teacher['checkpoint_hashes']:
            raise ValueError('Teacher bank hash/order mismatch')
        train_cmd=read(base/'new_contrast_train.command.json')
        data=Path(train_cmd[train_cmd.index('--data-root')+1])
        evaluation_cmd=read(base/'eval/new_teacher_full.command.json')
        assets=Path(evaluation_cmd[evaluation_cmd.index('--asset-mam-root')+1])
        initial=Path(train_cmd[train_cmd.index('--initialize-backbone')+1])
        old_prov=read(base/'students/new_contrast/provenance.json')
        if sha256_file(initial)!=old_prov['initialization_checkpoint_hash']:
            raise ValueError('Baseline student initializer changed')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),MAM_REACTOR_ROOT=str(assets),
                 PYTHONPATH=f'{ROOT/"mam_reactor"}:{ROOT}',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                 OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        def execute(name,command):
            write(args.root/f'{name}.command.json',command)
            with (args.root/f'{name}.log').open('x') as log:
                subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        sources={}
        for folder in ('coreset_reactor','mam_reactor/regnn','mam_reactor/framework/metrics'):
            for source in (ROOT/folder).glob('*.py'):
                relative=source.relative_to(ROOT); dest=args.root/'source_snapshot'/relative
                dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
                sources[str(relative)]=sha256_file(source)
        write(args.root/'experiment.json',{'teacher_run':teacher_root,'baseline_run':base,
            'teacher_hashes':teacher['checkpoint_hashes'],'teacher_metrics':teacher['metrics'],
            'authorization':'User explicitly accepted count4 quality/diversity tradeoff and requested student distillation.',
            'historical_teacher_strict_gate_passed':read(teacher_root/'summary.json')['teacher_strict_gate_passed'],
            'do_not_relabel_as_strict_win':True,'student_recipe':'centered_residual, TRAIN gradient ratio .25',
            'epochs':10,'expected_updates':15850,'train_contexts':6340,'seed':1,
            'initialization_sha256':sha256_file(initial),'gpu':args.gpu,
            'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'source_hashes':sources,'inference':'one student, ten outputs; no GT/no noise/no scale'})
        stage('generating_train_cache')
        allowlist=read(base/'frozen_T1/rank_00/config.json')['session_allowlist']
        execute('cache',[sys.executable,'-m','coreset_reactor.teacher_cache','--data-root',str(data),
            '--teacher-checkpoints',*map(str,checkpoints),'--output-dir',str(args.root/'train_cache'),
            '--device','cuda:0','--batch-size','4','--workers','4','--session-allowlist',*allowlist])
        cache=args.root/'train_cache/manifest.json'
        manifest=read(cache)
        if manifest['teacher_checkpoint_hashes']!=teacher['checkpoint_hashes'] or manifest['contexts']!=6340:
            raise ValueError('Unexpected training cache')
        new_views=list(view_rows(cache));old_views=list(view_rows(base/'new_cache/manifest.json'))
        if new_views!=old_views: raise ValueError('TRAIN source/crop schedule mismatch')
        write(args.root/'view_audit.json',{'matched_views':len(new_views),'matched':True})
        stage('training_student')
        train_cmd[train_cmd.index('--cache-manifest')+1]=str(cache)
        train_cmd[train_cmd.index('--run-dir')+1]=str(args.root/'student')
        execute('train',train_cmd)
        new_prov=read(args.root/'student/provenance.json')
        for key in ('initial_student_state_sha256','observed_source_schedule_sha256','initialization_checkpoint_hash'):
            if new_prov[key]!=old_prov[key]:raise ValueError(f'Matched training failed: {key}')
        if read(args.root/'student/train_metrics.json')[-1]['step']!=15850:
            raise ValueError('Training budget mismatch')
        target_cache=base/'val1124_gt_cache'
        for kind in ('fast','full'):
            stage(f'evaluating_val1124_{kind}')
            cmd=[sys.executable,'-m','coreset_reactor.eval_expert_student_fast',
                '--checkpoint',str(args.root/'student/student.pth'),'--data-root',str(data),
                '--asset-mam-root',str(assets),'--output',str(args.root/f'val1124_{kind}.json'),
                '--target-cache',str(target_cache),'--device','cuda:0','--eval-seed','1234']
            if kind=='full':cmd+=['--exact-frd','--metric-workers','12']
            execute(f'eval_{kind}',cmd)
        result=read(args.root/'val1124_full.json');baseline=read(base/'eval/new_contrast_full.json')
        if len(result['rows'])!=1124:raise ValueError('Incomplete VAL')
        for key in ('clip_id','target_cache_key'):
            if [r[key] for r in result['rows']]!=[r[key] for r in baseline['rows']]:
                raise ValueError(f'Matched evaluation failed: {key}')
        stage('random_gt_frc100')
        files=[base/'eval/new_teacher_full.json',teacher_root/'eval/teacher_full.json',
               base/'eval/new_contrast_full.json',args.root/'val1124_full.json']
        names=['teacher_count8','teacher_count4','student_count8','student_count4']
        execute('random_frc100',[sys.executable,'-m','coreset_reactor.eval_random_reference_frc',
            '--evaluations',*map(str,files),'--names',*names,'--target-cache',str(target_cache),
            '--output-dir',str(args.root/'random_frc100'),'--draws','100','--expected-contexts','1124'])
        metrics={name:read(path)['metrics'] for name,path in zip(names,files)}
        summary={'dataset':'REACT2024','contexts':1124,'matched_checks':'passed','metrics':metrics,
                 'student_strict_gate_passed':quality_gate(result['metrics'],baseline['metrics']),
                 'teacher_tradeoff_user_accepted':True,'wall_seconds':time.time()-started}
        write(args.root/'summary.json',summary)
        lines=['# REACT2024 accepted count4 teacher → student','',
            'Full VAL1124, all-session GT, center750, AU rounded, exact DTW-FRD.',
            'Single seed, not native neighbour-GT / official TEST. No inference rescaling.',
            'Count4 teacher failed the original strict gate; user explicitly accepted the tradeoff.',
            'Matched contrast KD: identical initialization and 6340 source/crop views, 10 epochs / 15850 updates.',
            'TRAIN-only scale/gradient calibration repeated for new targets. All old weights retained.','',
            '| Model | FRC | exact FRD | FRDiv | FRVar |','|---|---:|---:|---:|---:|']
        for name,m in metrics.items():
            lines.append(f"| {name} | {m['FRC']:.6f} | {m['exact_FRD']:.6f} | {m['FRDiv']:.6f} | {m['FRVar']:.6f} |")
        (args.root/'REACT2024_COUNT4_STUDENT.md').write_text('\n'.join(lines)+'\n')
        stage('complete',summary=summary)
    except Exception as error:
        stage('failed',error=repr(error));raise


if __name__=='__main__':main()
