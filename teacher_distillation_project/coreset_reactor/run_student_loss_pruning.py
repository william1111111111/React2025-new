"""Matched loss deletion study: minimal KD, +velocity, +signed residual.

The retained full (+velocity,+residual) student is the fourth factorial cell.
The residual coefficient/scales are frozen from its TRAIN-only calibration;
deleting velocity must not also change the residual coefficient.
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

ROOT=Path(__file__).resolve().parents[1]
ARMS={'minimal':(0.,False),'velocity':(.05,False),'residual':(0.,True)}


def read(path): return json.loads(path.read_text())


def study_layout(base, dataset):
    if dataset == 'REACT2024':
        return dict(contexts=1124, train_contexts=6340, updates=15850,
                    full=base/'student', evaluation=base/'val1124_full.json',
                    train_command=base/'train.command.json', eval_command=base/'eval_full.command.json',
                    retained_velocity=None, arms=dict(ARMS), teacher='fixed effective-count4 teacher bank')
    return dict(contexts=571, train_contexts=3320, updates=8300,
                full=base/'contrast', evaluation=base/'contrast/val571_full.json',
                train_command=base/'contrast_command.json', eval_command=base/'students_command.json',
                retained_velocity=base/'basic', arms={k:ARMS[k] for k in ('minimal','residual')},
                teacher='fixed effective-count8 mode-reference teacher bank')


def training_command(template, destination, velocity, frozen=None):
    command=list(template)
    for key,value in (('--run-dir',str(destination)),('--relation-grad-ratio','0'),('--device','cuda:0')):
        command[command.index(key)+1]=value
    command+=['--velocity-weight',str(velocity)]
    if frozen is not None:command+=['--frozen-relation-provenance',str(frozen)]
    return command


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-student-run',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--dataset',choices=('REACT2024','REACT2025'),default='REACT2024')
    p.add_argument('--gpus',type=int,nargs='+',default=[1,5,6])
    args=p.parse_args()
    base=args.baseline_student_run.resolve()
    layout=study_layout(base,args.dataset);arms=layout['arms'];contexts=layout['contexts']
    if len(set(args.gpus))!=len(arms) or len(args.gpus)!=len(arms):
        raise ValueError('Use one distinct GPU per newly trained arm')
    args.root=args.root.resolve();args.root.mkdir(parents=True,exist_ok=False)
    started=time.time()
    def write(path,value):
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,default=str)+'\n');tmp.replace(path)
    def stage(name,**extra):
        write(args.root/'status.json',{'stage':name,'pid':os.getpid(),'started_unix':started,
                                     'updated_unix':time.time(),**extra});print(name,flush=True)
    try:
        stage('preflight')
        reference=read(layout['evaluation'])
        config=read(layout['full']/'config.json')
        provenance=read(layout['full']/'provenance.json')
        cache=Path(config['cache_manifest']);manifest=read(cache)
        if reference['metrics']['contexts']!=contexts or manifest['contexts']!=layout['train_contexts']:
            raise ValueError('Unexpected dataset sizes')
        expected_updates=read(layout['full']/'train_metrics.json')[-1]['step']
        if expected_updates!=layout['updates'] or config['epochs']!=10:
            raise ValueError('Baseline is not the frozen 10-epoch recipe')
        if manifest['cache_manifest_hash']!=provenance['cache_manifest_hash']:
            raise ValueError('Baseline teacher cache mismatch')
        if sha256_file(layout['full']/'student.pth')!=reference['checkpoint_sha256']:
            raise ValueError('Frozen full student checkpoint changed')
        calibration=provenance['relation_calibration']
        if calibration['objective']!='centered_residual' or calibration['weight']!=config['relation_weight']:
            raise ValueError('Unexpected reference residual calibration')
        source_hashes={}
        for folder in ('coreset_reactor','mam_reactor/regnn','mam_reactor/framework/metrics'):
            for src in (ROOT/folder).glob('*.py'):
                rel=src.relative_to(ROOT);dest=args.root/'source_snapshot'/rel
                dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
                source_hashes[str(rel)]=sha256_file(src)
        frozen=args.root/'frozen_full_provenance.json'
        shutil.copy2(layout['full']/'provenance.json',frozen)
        shutil.copy2(layout['full']/'config.json',args.root/'frozen_full_config.json')
        shutil.copy2(layout['evaluation'],args.root/'frozen_full_val.json')
        template=read(layout['train_command'])
        eval_template=read(layout['eval_command'])
        def get(command,key):return command[command.index(key)+1]
        assets=Path(get(eval_template,'--asset-mam-root'))
        target_cache=Path(get(eval_template,'--target-cache'))
        data=Path(config['data_root'])
        env=dict(os.environ,MAM_REACTOR_ROOT=str(assets),PYTHONPATH=f'{ROOT/"mam_reactor"}:{ROOT}',
                 OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        write(args.root/'experiment.json',{
            'dataset':args.dataset,'teacher':layout['teacher']+', no teacher retraining',
            'baseline_student_run':base,'cache_manifest_hash':manifest['cache_manifest_hash'],
            'teacher_hashes':manifest['teacher_checkpoint_hashes'],'source_hashes':source_hashes,
            'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'arms':{name:{'velocity_weight':v,'residual_weight':calibration['weight'] if r else 0}
                    for name,(v,r) in arms.items()},
            'retained_velocity':layout['retained_velocity'],
            'frozen_full':{'velocity_weight':.05,'residual_weight':calibration['weight']},
            'residual_scale':calibration['scale'],'calibration':'Reuse frozen TRAIN calibration; no new weight tuning',
            'architecture':'unchanged shared ExpertConditionalREGNN 3803201 parameters',
            'seed':1,'epochs':10,'updates_per_arm':expected_updates,'gpus':args.gpus,
            'protocol':f'full VAL{contexts} all-session GT, AU rounding, exact DTW-FRD',
            'scope':'Prune objective terms only; no graph/backbone changes, no noise/scaling',
            'limitations':'Single-seed exploratory deletion study; no universal loss-weight or SOTA claim.'})
        def execute(command,gpu,log):
            log.parent.mkdir(parents=True,exist_ok=True);write(log.with_suffix('.command.json'),command)
            with log.open('x') as f:
                subprocess.run(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),
                               stdout=f,stderr=subprocess.STDOUT,check=True)
        def audit_training(d,velocity,residual):
            prov=read(d/'provenance.json')
            for key in ('cache_manifest_hash','initial_student_state_sha256',
                        'initialization_checkpoint_hash','observed_source_schedule_sha256'):
                if prov[key]!=provenance[key]:raise ValueError(f'Matched audit failed {d}/{key}')
            cfg=read(d/'config.json')
            for key in ('seed','epochs','batch_size','lr','max_steps','parameter_count','expert_embedding_dim'):
                if cfg[key]!=config[key]:raise ValueError(f'Config mismatch {d}/{key}')
            if cfg.get('velocity_weight',.05)!=velocity or cfg['relation_weight']!=(calibration['weight'] if residual else 0):
                raise ValueError('Incorrect objective coefficients')
            if read(d/'train_metrics.json')[-1]['step']!=expected_updates:
                raise ValueError('Unequal update budgets')
            return {'passed':True,'comparison':'frozen full student',
                    'velocity_weight':velocity,'relation_weight':cfg['relation_weight']}
        def audit_evaluation(result):
            if result['metrics']['contexts']!=contexts:raise ValueError('Incomplete evaluation')
            for key in ('clip_id','target_cache_key'):
                if [r[key] for r in result['rows']]!=[r[key] for r in reference['rows']]:
                    raise ValueError(f'Matched eval failed {key}')
        if layout['retained_velocity'] is not None:
            retained=layout['retained_velocity']
            audit=audit_training(retained,.05,False)
            evaluated=read(retained/f'val{contexts}_full.json');audit_evaluation(evaluated)
            if sha256_file(retained/'student.pth')!=evaluated['checkpoint_sha256']:
                raise ValueError('Retained velocity checkpoint changed')
            write(args.root/'retained_velocity_audit.json',audit)
            for filename in ('provenance.json','config.json',f'val{contexts}_full.json'):
                shutil.copy2(retained/filename,args.root/f'retained_velocity_{filename}')
        def run_arm(name,gpu):
            velocity,residual=arms[name];d=args.root/name
            command=training_command(template,d,velocity,frozen if residual else None)
            execute(command,gpu,args.root/f'{name}_train.log')
            write(d/'matched_audit.json',audit_training(d,velocity,residual))
            print(f'{name}: training complete, matched audit passed',flush=True)
            for kind in ('fast','full'):
                output=d/f'val{contexts}_{kind}.json'
                cmd=[sys.executable,'-m','coreset_reactor.eval_expert_student_fast',
                     '--checkpoint',str(d/'student.pth'),'--data-root',str(data),'--asset-mam-root',str(assets),
                     '--output',str(output),'--device','cuda:0','--eval-seed','1234','--target-cache',str(target_cache)]
                if kind=='full':cmd+=['--exact-frd','--metric-workers','12']
                execute(cmd,gpu,output.with_suffix('.log'))
                result=read(output)
                audit_evaluation(result)
                print(json.dumps({'arm':name,'eval':kind,'metrics':result['metrics']}),flush=True)
        stage('training_arms_then_full_val',arms=list(arms))
        with ThreadPoolExecutor(max_workers=len(arms)) as pool:
            tasks=[pool.submit(run_arm,name,gpu) for name,gpu in zip(arms,args.gpus)]
            for task in tasks:task.result()
        stage('random_reference_frc100')
        names=['full',*ARMS]
        paths=[layout['evaluation'],*[((layout['retained_velocity'] if a=='velocity' and
                layout['retained_velocity'] is not None else args.root/a)/f'val{contexts}_full.json') for a in ARMS]]
        execute([sys.executable,'-m','coreset_reactor.eval_random_reference_frc',
                 '--evaluations',*map(str,paths),'--names',*names,'--target-cache',str(target_cache),
                 '--output-dir',str(args.root/'random_frc100'),'--draws','100','--expected-contexts',str(contexts)],
                args.gpus[0],args.root/'random_frc100.log')
        results={name:read(path) for name,path in zip(names,paths)}
        baseline=reference['metrics']
        summary={'dataset':args.dataset,'contexts':contexts,'matched_checks':'passed',
                 'metrics':{a:v['metrics'] for a,v in results.items()},
                 'speed_diagnostic':{a:v['speed'] for a,v in results.items()},
                 'strict_no_loss_vs_full':{a:all([
                     v['metrics']['FRC']>=baseline['FRC'],v['metrics']['exact_FRD']<=baseline['exact_FRD'],
                     v['metrics']['FRDiv']>=baseline['FRDiv']]) for a,v in results.items() if a!='full'},
                 'wall_seconds':time.time()-started}
        write(args.root/'summary.json',summary)
        lines=[f'# Matched {args.dataset} student loss deletion','',
               f"{layout['teacher']}. Full VAL{contexts}, all-session GT, exact FRD, AU rounding.",
               f'Single seed. Same initialization, TRAIN cache/views/order, 10 epochs / {expected_updates} updates.',
               'Residual scale/coefficient frozen from the retained full model TRAIN calibration.',
               'No backbone change, teacher change, inference noise or rescaling.', '',
               '| Arm | Velocity | Residual | FRC | exact FRD | FRDiv | FRVar |',
               '|---|---:|---:|---:|---:|---:|---:|']
        for name in names:
            velocity,residual=(.05,True) if name=='full' else ARMS[name]
            m=results[name]['metrics']
            lines.append(f"| {name} | {velocity} | {calibration['weight'] if residual else 0:.6f} | "
                         f"{m['FRC']:.6f} | {m['exact_FRD']:.6f} | {m['FRDiv']:.6f} | {m['FRVar']:.6f} |")
        lines+=['','The full arm is a retained completed run, not retrained in this sweep.',
                'The velocity-only arm is also retained when running REACT2025; matched hashes and evaluation checked.',
                'Deletion candidates are not automatically promoted; report quality/diversity tradeoffs honestly.',
                'No claim that frozen calibration can be removed across datasets; minimal KD needs none.',
                'Velocity is still computed for logging when weighted zero; this is objective deletion, not a runtime optimization.','']
        (args.root/'STUDENT_LOSS_PRUNING.md').write_text('\n'.join(lines))
        stage('complete',summary=summary)
    except Exception as error:
        stage('failed',error=repr(error));raise


if __name__=='__main__':main()
