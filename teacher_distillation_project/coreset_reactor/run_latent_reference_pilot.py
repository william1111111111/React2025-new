"""One preregistered latent-reference teacher bank, strictly matched at 600 steps."""
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
import torch

from coreset_reactor.teacher_cache import sha256_file
from coreset_reactor.run_conditional_slot_pilot import comparison

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','data-root','asset-mam-root','frozen-root','convergence-root','target-cache'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--reference-prior',choices=['uniform','mode'],default='uniform')
    p.add_argument('--reference-effective-count',type=float,default=8)
    p.add_argument('--verify-reference-run',type=Path,
                   help='Compare all final model tensors; reuse prior evaluation only after exact parity')
    args=p.parse_args()
    for name in ('root','data_root','asset_mam_root','frozen_root','convergence_root','target_cache'):
        setattr(args,name,getattr(args,name).resolve())
    if args.verify_reference_run: args.verify_reference_run=args.verify_reference_run.resolve()
    args.root.mkdir(parents=True,exist_ok=False)
    expected=json.loads((args.convergence_root/'arms/T1/rank_00/conditional-step000600-seed1.json').read_text())
    files=['coreset_reactor/train_latent_reference_teacher.py','coreset_reactor/run_latent_reference_pilot.py',
           'coreset_reactor/run_conditional_slot_pilot.py','coreset_reactor/eval_expert_student_fast.py',
           'coreset_reactor/fast_frc.py','coreset_reactor/oracle_select.py','coreset_reactor/evaluate.py',
           'mam_reactor/regnn/train_conditional_regnn.py','mam_reactor/regnn/conditional_model.py',
           'mam_reactor/regnn/conditional_data.py']
    if args.reference_prior=='mode': files+=['coreset_reactor/train_mode_reference_teacher.py']
    for name in files:
        dest=args.root/'source_snapshot'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,dest)
    provenance={'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'source_hashes':{f:sha256_file(ROOT/f) for f in files},
                'manifest_hashes':{str(r):sha256_file(args.frozen_root/f'manifests/T1/rank_{r:02d}.json') for r in range(10)},
                'model':'unchanged ConditionalREGNN graph2 linear head',
                'seed':1,'steps':600,'batch_size':32,'formal_schedule_epochs':50,
                'optimizer':'unchanged AdamW1e-4 betas(.9,.99) wd1e-4 clip1 cosine50epochs',
                'objective':'fixed T1 MSE + 1 - .05*logmeanexp(CCC_all_raw_TRAIN_session/.05); unchanged masked velocity',
                'temperature':.05,'MSE_targets_per_prediction':1,'CCC_targets_per_prediction':'all TRAIN-session opposite-role',
                'training_DTW_calls':0,'new_updates':5400,'updates_including_reused_q0':6000,
                'q0':'reuse exact T1_600 unchanged true paired anchor',
                'inference':'10 frozen teachers, no GT, no post-hoc mixing',
                'numeric_goal':'FRC>=1.10, FRDiv .17-.20, exact_FRD<=T1_600',
                'selection':'one preregistered TRAIN-only configuration, no VAL sweep',
                'speed_caveat':'concurrent jobs; report real throughput, not isolated benchmark'}
    provenance['reference_prior']=args.reference_prior
    provenance['verify_reference_run']=str(args.verify_reference_run) if args.verify_reference_run else None
    if args.reference_prior=='mode':
        provenance['objective']='fixed T1 MSE + 1 - .05*logsumexp(CCC/.05 + log(mode_prior)); unchanged masked velocity'
        provenance['reference_effective_count']=args.reference_effective_count
        provenance['mode_prior']='TRAIN-only mean/std50D kernel around unchanged fixed GT; entropy-matched bandwidth'
        provenance['selection']='exploratory follow-up to uniform-prior collapse; one prespecified count8, no sweep'
    prov=args.root/'provenance.json'
    prov.write_text(json.dumps(provenance,indent=2)+'\n')
    env=dict(os.environ,MAM_REACTOR_ROOT=str(args.asset_mam_root),
             PYTHONPATH=f'{ROOT/"mam_reactor"}:{ROOT}',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')

    def execute(command,gpu,log):
        log.with_suffix('.command.json').write_text(json.dumps(command,indent=2)+'\n')
        with log.open('x') as handle:
            subprocess.run(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),
                           stdout=handle,stderr=subprocess.STDOUT,check=True)

    def queue(gpu,ranks):
        for rank in ranks:
            directory=args.root/f'rank_{rank:02d}'
            directory.mkdir()
            module=('coreset_reactor.train_latent_reference_teacher' if args.reference_prior=='uniform'
                    else 'coreset_reactor.train_mode_reference_teacher')
            cmd=[sys.executable,'-m',module,
                 '--data-dir',str(args.data_root),'--run-dir',str(directory),
                 '--epochs','50','--batch-size','32','--workers','8','--precision','bf16',
                 '--seed','1','--fixed-pair-seed','1','--fixed-pair-rank',str(rank),
                 '--fixed-pair-count','10','--fixed-pair-manifest',str(args.frozen_root/f'manifests/T1/rank_{rank:02d}.json'),
                 '--fixed-pair-alignment-policy','relative_time_masked','--graph-layers','2','--prediction-head','linear',
                 '--max-train-steps','600','--snapshot-steps','600','--save-every','5','--print-every','100',
                 '--reference-temperature','.05']
            if args.reference_prior=='mode': cmd+=['--reference-effective-count',str(args.reference_effective_count)]
            execute(cmd,gpu,directory/'train.log')
            audit=json.loads((directory/'conditional-step000600-seed1.json').read_text())
            for key in ('initial_state_sha256','observed_source_schedule_sha256','global_step','parameter_count'):
                assert audit[key]==expected[key],f'Matched protocol failed {rank}/{key}'
            print(f'Latent-reference q{rank} done; initialization/source/crop matched',flush=True)

    started=time.monotonic()
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(queue,gpu,list(range(i+1,10,3))) for i,gpu in enumerate((1,5,6))]
        for job in jobs: job.result()
    provenance['training_wall_seconds']=time.monotonic()-started
    provenance['matched_checks']='passed'
    provenance['audits']={str(r):json.loads((args.root/f'rank_{r:02d}/conditional-step000600-seed1.json').read_text()) for r in range(1,10)}
    prov.write_text(json.dumps(provenance,indent=2)+'\n')
    ckpts=[args.convergence_root/'arms/T1/rank_00/conditional-step000600-seed1.pth']+[
        args.root/f'rank_{r:02d}/conditional-step000600-seed1.pth' for r in range(1,10)]
    if args.verify_reference_run:
        parity={}
        for rank in range(1,10):
            oldpath=args.verify_reference_run/f'rank_{rank:02d}/conditional-step000600-seed1.pth'
            old=torch.load(oldpath,map_location='cpu')
            new=torch.load(ckpts[rank],map_location='cpu')
            same_keys=set(old['state_dict'])==set(new['state_dict'])
            different=[k for k in old['state_dict'] if not torch.equal(old['state_dict'][k],new['state_dict'][k])] if same_keys else ['different_keys']
            parity[str(rank)]={'model_config_equal':old['model_config']==new['model_config'],
                               'all_tensors_equal':same_keys and not different,'different_keys':different,
                               'old_checkpoint_sha256':sha256_file(oldpath),
                               'new_checkpoint_sha256':sha256_file(ckpts[rank])}
        (args.root/'weight_parity.json').write_text(json.dumps(parity,indent=2)+'\n')
        priorprov=json.loads((args.verify_reference_run/'provenance.json').read_text())
        compatible=all(priorprov['source_hashes'][name]==provenance['source_hashes'][name]
                       for name in ('mam_reactor/regnn/conditional_model.py','mam_reactor/regnn/conditional_data.py',
                                    'coreset_reactor/eval_expert_student_fast.py','coreset_reactor/evaluate.py',
                                    'coreset_reactor/fast_frc.py','coreset_reactor/oracle_select.py'))
        result_path=args.verify_reference_run/'val571_full.json'
        if compatible and all(x['model_config_equal'] and x['all_tensors_equal'] for x in parity.values()) and result_path.exists():
            oldresult=json.loads(result_path.read_text())
            oldsummary=json.loads((args.verify_reference_run/'summary.json').read_text())
            summary={**oldsummary,'verification':'All nine trained model states and configs bitwise identical; paired q0 unchanged',
                     'evaluation_reused_from':str(result_path),'evaluation_result_sha256':sha256_file(result_path),
                     'weight_parity_file':'weight_parity.json',
                     'wall_seconds_this_verification':time.monotonic()-started,
                     'note':'Metrics/latency and original checkpoint identifiers remain attributed to the original evaluated run; no new evaluator run claimed.'}
            (args.root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            print(json.dumps({'verified_bitwise_reproduction':True,'metrics':oldresult['metrics']}),flush=True)
            return
    for stage in ('fast','full'):
        output=args.root/f'val571_{stage}.json'
        cmd=[sys.executable,'-m','coreset_reactor.eval_expert_student_fast',
             '--teacher-checkpoints',*map(str,ckpts),'--data-root',str(args.data_root),
             '--asset-mam-root',str(args.asset_mam_root),'--device','cuda:0','--eval-seed','1234',
             '--output',str(output),'--target-cache',str(args.target_cache)]
        if stage=='full': cmd+=['--exact-frd','--metric-workers','16']
        execute(cmd,5,output.with_suffix('.log'))
        print(json.dumps({'stage':stage,'metrics':json.loads(output.read_text())['metrics']}),flush=True)
    result=json.loads((args.root/'val571_full.json').read_text())
    baselines={a:json.loads((args.convergence_root/f'eval/{a}_s000600_val571_full.json').read_text()) for a in ('T0','T1')}
    summary={'metrics':result['metrics'],'speed':result['speed'],
             'baselines':{a:v['metrics'] for a,v in baselines.items()},
             'vs_T1':comparison(result,baselines['T1']),
             'numeric_goal':result['metrics']['FRC']>=1.10 and .17<=result['metrics']['FRDiv']<=.20 and
                            result['metrics']['exact_FRD']<=baselines['T1']['metrics']['exact_FRD'],
             'wall_seconds':time.monotonic()-started}
    (args.root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
