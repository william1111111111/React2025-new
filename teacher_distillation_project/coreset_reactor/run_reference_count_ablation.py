"""One matched entropy-count ablation; distill only after a strict teacher gate."""
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

from coreset_reactor.teacher_cache import sha256_file
from coreset_reactor.run_mode_reference_student import view_rows

ROOT=Path(__file__).resolve().parents[1]


def read(path): return json.loads(path.read_text())


def quality_gate(candidate, baseline):
    return (candidate['FRC']>=baseline['FRC'] and
            candidate['exact_FRD']<=baseline['exact_FRD'] and
            candidate['FRDiv']>baseline['FRDiv'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-run',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--effective-count',type=float,default=4)
    p.add_argument('--gpus',type=int,nargs=3,default=[1,5,6])
    args=p.parse_args()
    if not 1<=args.effective_count<8: raise ValueError('This bounded ablation narrows the count8 baseline')
    args.root=args.root.resolve(); args.root.mkdir(parents=True,exist_ok=False)
    base=args.baseline_run.resolve(); started=time.time()
    def write(path,value):
        temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(value,indent=2,default=str)+'\n'); temp.replace(path)
    def stage(name,**extra):
        write(args.root/'status.json',{'stage':name,'pid':os.getpid(),'started_unix':started,
                                      'updated_unix':time.time(),**extra}); print(name,flush=True)
    old_eval=read(base/'eval/new_teacher_full.json')
    template=read(base/'teachers/rank_01/train.command.json')
    data=Path(template[template.index('--data-dir')+1])
    eval_template=read(base/'eval/new_teacher_full.command.json')
    assets=Path(eval_template[eval_template.index('--asset-mam-root')+1])
    env=dict(os.environ,MAM_REACTOR_ROOT=str(assets),PYTHONPATH=f'{ROOT/"mam_reactor"}:{ROOT}',
             OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    def execute(command,gpu,log):
        log.parent.mkdir(parents=True,exist_ok=True)
        write(log.with_suffix('.command.json'),command)
        with log.open('x') as f:
            subprocess.run(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),
                           stdout=f,stderr=subprocess.STDOUT,check=True)
    try:
        stage('training_nine_teachers')
        snapshots={}
        for folder in ('coreset_reactor','mam_reactor/regnn','mam_reactor/framework/metrics'):
            for src in (ROOT/folder).glob('*.py'):
                dest=args.root/'source_snapshot'/src.relative_to(ROOT)
                dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dest)
                snapshots[str(src.relative_to(ROOT))]=sha256_file(src)
        write(args.root/'provenance.json',{'baseline_run':base,'changed_parameter':{'effective_count':[8,args.effective_count]},
            'rationale':'TRAIN prior-overlap diagnostic: REACT2024 count4 overlaps like REACT2025 count8; hypothesis, not causal proof.',
            'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'source_hashes':snapshots,'strict_gate':'FRC>=count8, exact_FRD<=count8, FRDiv>count8',
            'inference':'unchanged; no scaling/no noise/no GT', 'single_seed':1})
        def queue(gpu,ranks):
            for rank in ranks:
                cmd=read(base/f'teachers/rank_{rank:02d}/train.command.json')
                directory=args.root/f'teachers/rank_{rank:02d}'
                cmd[cmd.index('--run-dir')+1]=str(directory)
                cmd[cmd.index('--reference-effective-count')+1]=str(args.effective_count)
                execute(cmd,gpu,directory/'train.log')
                a=read(directory/'conditional-step000600-seed1.json')
                b=read(base/f'teachers/rank_{rank:02d}/conditional-step000600-seed1.json')
                for k in ('initial_state_sha256','observed_source_schedule_sha256','global_step','parameter_count'):
                    if a[k]!=b[k]: raise ValueError(f'Matched teacher audit failed {rank}/{k}')
                print(f'q{rank} complete, matching passed',flush=True)
        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs=[pool.submit(queue,g,list(range(i+1,10,3))) for i,g in enumerate(args.gpus)]
            for job in jobs: job.result()
        teachers=[Path(old_eval['checkpoint'][0])]+[
            args.root/f'teachers/rank_{r:02d}/conditional-step000600-seed1.pth' for r in range(1,10)]
        if sha256_file(teachers[0])!=old_eval['checkpoint_hashes'][0]: raise ValueError('Paired anchor changed')
        target_cache=base/'val1124_gt_cache'
        def evaluate(label,paths,gpu,student=False):
            for kind in ('fast','full'):
                out=args.root/f'eval/{label}_{kind}.json'
                cmd=[sys.executable,'-m','coreset_reactor.eval_expert_student_fast',
                     *(['--checkpoint',str(paths[0])] if student else ['--teacher-checkpoints',*map(str,paths)]),
                     '--data-root',str(data),'--asset-mam-root',str(assets),'--device','cuda:0',
                     '--output',str(out),'--target-cache',str(target_cache),'--eval-seed','1234']
                if kind=='full': cmd+=['--exact-frd','--metric-workers','12']
                execute(cmd,gpu,out.with_suffix('.log'))
            result=read(args.root/f'eval/{label}_full.json')
            for k in ('clip_id','target_cache_key'):
                if [r[k] for r in result['rows']]!=[r[k] for r in old_eval['rows']]:
                    raise ValueError(f'Matched evaluation failed {k}')
            return result
        stage('full_val1124_teacher_evaluation')
        result=evaluate('teacher',teachers,args.gpus[1])
        summary={'teacher':result['metrics'],'baseline_teacher':old_eval['metrics'],
                 'teacher_strict_gate_passed':quality_gate(result['metrics'],old_eval['metrics']),
                 'matched_teacher_checks':'passed','effective_count':args.effective_count}
        write(args.root/'summary.json',summary)
        if not summary['teacher_strict_gate_passed']:
            summary['student_training']='not_started: strict teacher quality/diversity gate failed'
            summary['wall_seconds']=time.time()-started; write(args.root/'summary.json',summary)
            stage('complete_gate_failed',summary=summary)
            return
        stage('teacher_gate_passed_generating_train_cache')
        cfg=read(base/'frozen_T1/rank_00/config.json')
        execute([sys.executable,'-m','coreset_reactor.teacher_cache','--data-root',str(data),
                 '--teacher-checkpoints',*map(str,teachers),'--output-dir',str(args.root/'train_cache'),
                 '--device','cuda:0','--batch-size','4','--workers','4','--session-allowlist',*cfg['session_allowlist']],
                args.gpus[0],args.root/'cache.log')
        cache=args.root/'train_cache/manifest.json'
        if list(view_rows(cache))!=list(view_rows(base/'new_cache/manifest.json')):
            raise ValueError('Cache TRAIN input mismatch')
        stage('training_matched_contrast_student')
        cmd=read(base/'new_contrast_train.command.json')
        cmd[cmd.index('--cache-manifest')+1]=str(cache)
        cmd[cmd.index('--run-dir')+1]=str(args.root/'student')
        execute(cmd,args.gpus[0],args.root/'student_train.log')
        a=read(args.root/'student/provenance.json'); b=read(base/'students/new_contrast/provenance.json')
        for k in ('initial_student_state_sha256','observed_source_schedule_sha256','initialization_checkpoint_hash'):
            if a[k]!=b[k]: raise ValueError(f'Matched student audit failed {k}')
        stage('full_val1124_student_evaluation')
        student=evaluate('student',[args.root/'student/student.pth'],args.gpus[0],True)
        baseline_student=read(base/'eval/new_contrast_full.json')
        summary.update({'student':student['metrics'],'baseline_student':baseline_student['metrics'],
                        'student_strict_gate_passed':quality_gate(student['metrics'],baseline_student['metrics']),
                        'matched_student_checks':'passed','wall_seconds':time.time()-started})
        write(args.root/'summary.json',summary); stage('complete',summary=summary)
    except Exception as error:
        stage('failed',error=repr(error)); raise


if __name__=='__main__': main()
