"""Two preregistered REACT2025 component controls, without outcome selection.

Uniform-reference teachers use the corrected CCC kernel. A direct-GT student
uses the same cached source views and minimal loss as its KD control. Both
are evaluated on full VAL571; single-seed exploratory results only.
"""
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
from coreset_reactor.run_conditional_slot_pilot import comparison

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,default=str)+'\n');temporary.replace(path)


def replace(command,key,value):
    command=list(command);command[command.index(key)+1]=str(value);return command


def uniform_command(template, directory):
    command=list(template)
    command[command.index('-m')+1]='coreset_reactor.train_latent_reference_teacher'
    i=command.index('--reference-effective-count');del command[i:i+2]
    return replace(command,'--run-dir',directory)


def aligned(a,b):
    if a['metrics']['contexts']!=571 or b['metrics']['contexts']!=571:
        raise ValueError('Full VAL571 is required')
    for key in ('clip_id','target_cache_key'):
        if [r[key] for r in a['rows']]!=[r[key] for r in b['rows']]:
            raise ValueError(f'Evaluation mismatch: {key}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--project-assets',type=Path,default=Path('/public/zhaowenjie/react2025_new'))
    p.add_argument('--gpus',type=int,nargs=3,default=[1,5,6])
    a=p.parse_args()
    if len(set(a.gpus))!=3:raise ValueError('Use three distinct GPUs')
    a.root=a.root.resolve();a.root.mkdir(parents=True,exist_ok=False)
    started=time.time();assets=a.project_assets.resolve();reports=assets/'coreset_reactor/reports'
    mode=reports/'mode_reference_teacher_v2_seed1'
    student=reports/'react2025_student_loss_pruning_seed1/minimal'
    student_bank=reports/'mode_reference_student_seed1'
    target_cache=reports/'fixed_teacher_distillation/val571_postprocessed_gt_cache'
    data=assets/'data';mam=assets/'mam_reactor'
    env=dict(os.environ,MAM_REACTOR_ROOT=str(mam),PYTHONPATH=f'{ROOT/"mam_reactor"}:{ROOT}',
             OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    def stage(name,**kw):
        write(a.root/'status.json',{'stage':name,'pid':os.getpid(),'started_unix':started,
                                   'updated_unix':time.time(),**kw});print(name,flush=True)
    def execute(command,gpu,log):
        log.parent.mkdir(parents=True,exist_ok=True);write(log.with_suffix('.command.json'),command)
        with log.open('x') as f:
            subprocess.run(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),
                           stdout=f,stderr=subprocess.STDOUT,check=True)
    def evaluate(name,gpu,checkpoints,teacher=False):
        output=a.root/f'{name}_val571_full.json'
        cmd=[sys.executable,'-m','coreset_reactor.eval_expert_student_fast',
             '--teacher-checkpoints' if teacher else '--checkpoint',*map(str,checkpoints),
             '--data-root',str(data),'--asset-mam-root',str(mam),'--device','cuda:0',
             '--eval-seed','1234','--target-cache',str(target_cache),'--output',str(output),
             '--exact-frd','--metric-workers','12']
        execute(cmd,gpu,output.with_suffix('.log'));result=read(output)
        aligned(result,read(student/'val571_full.json'))
        print(json.dumps({'evaluated':name,'metrics':result['metrics']}),flush=True)
        return result
    try:
        stage('preflight')
        prior=read(mode/'provenance.json')
        # Only the reference prior is changed in teacher training.
        for name in ('coreset_reactor/train_latent_reference_teacher.py',
                     'mam_reactor/regnn/train_conditional_regnn.py',
                     'mam_reactor/regnn/conditional_model.py','mam_reactor/regnn/conditional_data.py'):
            if sha256_file(ROOT/name)!=prior['source_hashes'][name]:
                raise ValueError(f'Frozen teacher training source changed: {name}')
        cfg=read(student/'config.json');prov=read(student/'provenance.json')
        if cfg['velocity_weight']!=0 or cfg['relation_weight']!=0 or cfg['epochs']!=10:
            raise ValueError('Expected completed minimal KD control')
        for name in ('mam_reactor/regnn/conditional_model.py','mam_reactor/regnn/expert_student.py'):
            if sha256_file(ROOT/name)!=prov['source_hashes'][name]:
                raise ValueError('Student architecture changed')
        if sha256_file(student/'student.pth')!=read(student/'val571_full.json')['checkpoint_sha256']:
            raise ValueError('KD baseline checkpoint changed')
        for name in ('coreset_reactor/eval_expert_student_fast.py','coreset_reactor/fast_frc.py',
                     'coreset_reactor/evaluate.py','coreset_reactor/oracle_select.py'):
            snapshot=student.parent/'source_snapshot'/name
            if sha256_file(ROOT/name)!=sha256_file(snapshot):
                raise ValueError('KD evaluation code changed; rerun baseline before reuse')
        templates=[read(mode/f'rank_{r:02d}/train.command.json') for r in range(1,10)]
        fixed=[reports/f'maxfrdiv_teacher_pilot_600_v3/manifests/T1/rank_{r:02d}.json' for r in range(10)]
        for r,path in enumerate(fixed):
            if sha256_file(path)!=prior['manifest_hashes'][str(r)]:raise ValueError('Fixed GT selection changed')
        q0=reports/'teacher_convergence_2400_seed1/arms/T1/rank_00/conditional-step000600-seed1.pth'
        teachers=[q0]+[mode/f'rank_{r:02d}/conditional-step000600-seed1.pth' for r in range(1,10)]
        cache=read(student_bank/'train_cache/manifest.json')
        if [sha256_file(path) for path in teachers]!=cache['teacher_checkpoint_hashes']:
            raise ValueError('Frozen teacher bank changed')
        hashes={}
        for folder in ('coreset_reactor','mam_reactor/regnn','mam_reactor/framework/metrics'):
            for src in (ROOT/folder).rglob('*.py'):
                rel=src.relative_to(ROOT);dst=a.root/'source_snapshot'/rel
                dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);hashes[str(rel)]=sha256_file(src)
        write(a.root/'experiment.json',{
            'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'source_hashes':hashes,'teacher_checkpoint_hashes':cache['teacher_checkpoint_hashes'],
            'frozen_gt_hashes':[sha256_file(path) for path in fixed],
            'dataset':'REACT2025','seed':1,'gpus':a.gpus,
            'uniform_reference':{'ranks':list(range(1,10)),'updates_each':600,'batch_size':32,
                                 'q0':'unchanged','temperature':.05,'only_change':'mode prior -> uniform prior'},
            'direct_gt':{'updates':8300,'epochs':10,'batch_size':4,'targets':'same Max-FRDiv fixed TRAIN GT',
                         'objective':'MSE + (1-CCC), no velocity or residual','teacher_output_supervision':False,
                         'initialization':'same retained teacher0 backbone as KD, not training from scratch',
                         'target_alignment':'paired native crop; other nine relative-time as current teachers',
                         'control':str(student)},
            'evaluation':'full VAL571, all-session GT, AU rounding, exact FRD; both teacher banks freshly evaluated',
            'limitations':'single seed exploratory VAL, not official sampled-ten TEST or proof of indispensability',
            'selection':'Report both controls regardless of outcome; no weight tuning',
            'compute':'Same per-stage update budgets, not equal end-to-end FLOPs: KD includes teacher training/cache costs'})
        def teacher_queue(gpu,ranks):
            for rank in ranks:
                directory=a.root/f'uniform/rank_{rank:02d}'
                command=uniform_command(templates[rank-1],directory)
                execute(command,gpu,directory/'train.log')
                audit=read(directory/'conditional-step000600-seed1.json')
                for key in ('initial_state_sha256','observed_source_schedule_sha256','global_step','parameter_count'):
                    if audit[key]!=prior['audits'][str(rank)][key]:raise ValueError(f'Teacher mismatch {rank}/{key}')
                write(directory/'matched_audit.json',{'passed':True,'reference':str(mode/f'rank_{rank:02d}')})
                print(f'uniform rank {rank} complete, matched',flush=True)
        def direct_student():
            write(a.root/'direct_status.json',{'stage':'building_real_gt_cache'})
            new_cache=a.root/'real_gt_cache'
            execute([sys.executable,'-m','coreset_reactor.build_matched_real_gt_cache',
                     '--data-root',str(data),'--source-manifest',str(student_bank/'train_cache/manifest.json'),
                     '--fixed-manifests',*map(str,fixed),'--output',str(new_cache)],a.gpus[0],a.root/'real_gt_cache.log')
            write(a.root/'direct_status.json',{'stage':'training'})
            cmd=read(student.parent/'minimal_train.command.json')
            cmd=replace(replace(cmd,'--cache-manifest',new_cache/'manifest.json'),'--run-dir',a.root/'direct_student')
            execute(cmd,a.gpus[0],a.root/'direct_student_train.log')
            direct=read(a.root/'direct_student/provenance.json')
            for key in ('initial_student_state_sha256','observed_source_schedule_sha256','initialization_checkpoint_hash'):
                if direct[key]!=prov[key]:raise ValueError(f'Student mismatch {key}')
            dc=read(a.root/'direct_student/config.json')
            for key in ('seed','epochs','batch_size','lr','max_steps','parameter_count','expert_embedding_dim','velocity_weight','relation_weight'):
                if dc[key]!=cfg[key]:raise ValueError(f'Student config mismatch {key}')
            if read(a.root/'direct_student/train_metrics.json')[-1]['step']!=8300:
                raise ValueError('Student update budget mismatch')
            if direct['target_kind']!='real_gt' or direct['teacher_checkpoint_hashes']:
                raise ValueError('Direct control must not use teacher outputs')
            write(a.root/'direct_student/matched_audit.json',{'passed':True,'only_intervention':'KD targets -> real GT targets'})
            write(a.root/'direct_status.json',{'stage':'full_val_evaluation'})
            result=evaluate('direct_student',a.gpus[0],[a.root/'direct_student/student.pth'])
            write(a.root/'direct_status.json',{'stage':'complete','metrics':result['metrics']})
            return result
        stage('training_uniform_teachers_and_direct_student')
        with ThreadPoolExecutor(max_workers=3) as pool:
            direct_job=pool.submit(direct_student)
            jobs=[pool.submit(teacher_queue,gpu,list(range(i+1,10,2))) for i,gpu in enumerate(a.gpus[1:])]
            for job in jobs:job.result()
            uniform=[q0]+[a.root/f'uniform/rank_{r:02d}/conditional-step000600-seed1.pth' for r in range(1,10)]
            stage('evaluating_teacher_controls_direct_student_continues_if_needed')
            uj=pool.submit(evaluate,'uniform_teacher',a.gpus[1],uniform,True)
            mj=pool.submit(evaluate,'mode_teacher_current_eval',a.gpus[2],teachers,True)
            ur,mr,dr=uj.result(),mj.result(),direct_job.result()
        kd=read(student/'val571_full.json')
        summary={'dataset':'REACT2025','contexts':571,'matched_checks':'passed',
                 'teacher':{'mode':mr['metrics'],'uniform':ur['metrics'],'uniform_minus_mode':comparison(ur,mr)},
                 'student':{'KD_minimal':kd['metrics'],'direct_GT':dr['metrics'],'direct_minus_KD':comparison(dr,kd)},
                 'speed':{'mode':mr['speed'],'uniform':ur['speed'],'direct_GT':dr['speed'],'KD_minimal':kd['speed']},
                 'wall_seconds':time.time()-started}
        write(a.root/'summary.json',summary)
        lines=['# REACT2025 key component ablations','',
               'Full VAL571 / all-session GT / exact FRD. Single seed; matched initialization and source schedules.',
               'Teacher and student comparisons are separate. No outcome-based promotion or discarded negative results.','',
               '| Stage | Arm | FRC | exact FRD | FRDiv | FRVar |','|---|---|---:|---:|---:|---:|']
        for stage_name,name,result in [('Teacher','Mode prior',mr),('Teacher','Uniform prior',ur),
                                      ('Student','Minimal KD',kd),('Student','Direct same fixed GT',dr)]:
            m=result['metrics'];lines.append(f"| {stage_name} | {name} | {m['FRC']:.6f} | {m['exact_FRD']:.6f} | {m['FRDiv']:.6f} | {m['FRVar']:.6f} |")
        lines+=['','Teacher control: corrected CCC in both arms; same fixed targets/q0/600 updates, only prior changes.',
                'Student control: same teacher0 initialization, 8300 updates, MSE+CCC; only supervision targets change.',
                'Direct GT is no output distillation, not a claim of no pretrained teacher initialization.',
                'All 3320 source views checked; 300 targets bitwise checked against production teacher preprocessing.',
                'Session-cluster bootstrap and per-session deltas are in summary.json; not training-seed uncertainty.',
                'Training costs/checkpoints/source hashes are retained. Different supervision has different compute cost.','']
        (a.root/'KEY_COMPONENT_ABLATIONS.md').write_text('\n'.join(lines))
        stage('complete',metrics={'uniform_teacher':ur['metrics'],'direct_student':dr['metrics']})
    except Exception as error:
        stage('failed',error=repr(error));raise


if __name__=='__main__':main()
