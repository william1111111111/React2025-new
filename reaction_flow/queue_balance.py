"""Finite two-arm queue on ONE physical GPU; all fixed checkpoints evaluated."""
import json,os,subprocess,time,fcntl
from .balance_common import ROOT,ARMS
from hirp.phase15_audit import sha256_file
from pathlib import Path

def atomic(name,value):
    p=ROOT/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)

def main():
    lock=(ROOT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((ROOT/'resume_regression.json').read_text())['passed']
    assert not (ROOT/'queue_started.json').exists()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUNBUFFERED='1')
    atomic('queue_started.json',dict(time=time.time(),pid=os.getpid(),physical_gpu=1,max_gpus=1,code_hashes={str(p):sha256_file(p) for p in Path('reaction_flow').glob('*.py')}))
    def launch(name,module,args):
        f=(ROOT/f'{name}.txt').open('x');cmd=['.venv/bin/python','-m',module]+args
        proc=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        atomic(f'{name}_command.json',dict(pid=proc.pid,command=cmd,time=time.time()))
        return proc,f
    jobs={arm:launch('train_'+arm,'reaction_flow.train_balance',['--arm',arm]) for arm in ARMS}
    evaluated=[];failures=[];trained=[];evaluation=None;frd=None;frd_done=[];counts={};reports=0
    def report():
        nonlocal reports
        with (ROOT/f'report_{reports:03d}.txt').open('x') as f:
            code=subprocess.call(['.venv/bin/python','-m','reaction_flow.report_balance'],env=env,stdout=f,stderr=subprocess.STDOUT)
        reports+=1
        if code:failures.append('report_'+str(reports-1))
    while True:
        for arm,(p,f) in list(jobs.items()):
            if p.poll() is not None:
                f.close();atomic('train_'+arm+'_exit.json',dict(code=p.returncode,time=time.time()));jobs.pop(arm)
                (failures if p.returncode else trained).append(arm)
        if evaluation:
            key,(p,f)=evaluation
            if p.poll() is not None:
                f.close();atomic('eval_'+key+'_exit.json',dict(code=p.returncode,time=time.time()))
                (failures if p.returncode else evaluated).append('eval_'+key if p.returncode else key)
                evaluation=None;report()
        if evaluation is None:
            for step in (16500,17000,18000):
                for arm in ARMS:
                    key=f'{arm}_{step}';cp=ROOT/arm/'attempt_000/checkpoints'/f'step_{step:06d}.pt'
                    if key not in evaluated and 'eval_'+key not in failures and cp.exists() and cp.with_suffix('.json').exists():
                        evaluation=(key,launch('eval_'+key,'reaction_flow.evaluate_balance',['--checkpoint',str(cp)]));break
                if evaluation:break
        if frd:
            arm,(p,f)=frd
            if p.poll() is not None:
                f.close();atomic(f'frd_{arm}_{counts[arm]}_exit.json',dict(code=p.returncode,time=time.time()));frd=None
                stats=sorted((ROOT/'frd20'/f'seed123_{arm}_step18000').glob('status_*.json'))
                if p.returncode:failures.append('frd_'+arm)
                elif stats and json.loads(stats[-1].read_text())['completed']:frd_done.append(arm)
                elif counts[arm]>=4:failures.append('frd_'+arm)
        if frd is None:
            for arm in ARMS:
                if f'{arm}_18000' in evaluated and arm not in frd_done and 'frd_'+arm not in failures:
                    counts[arm]=counts.get(arm,0)+1;label=f'seed123_{arm}_step18000'
                    frd=(arm,launch(f'frd_{arm}_{counts[arm]}','reaction_flow.frd_balance',['--model',label,'--task',str(ROOT/'task_multitarget'/f'{label}.json')]));break
        progress={};records={}
        for arm in ARMS:
            p=ROOT/arm/'attempt_000/training.jsonl'
            if p.exists():
                rows=[]
                for line in p.read_text().splitlines():
                    try:rows.append(json.loads(line))
                    except json.JSONDecodeError:pass
                if rows:
                    records[arm]=rows;progress[arm]={k:rows[-1][k] for k in ('phase_step','loss','lambda_task','lambda_goal','cap_hit','seconds')}
        atomic('monitor_latest.json',dict(time=time.time(),progress=progress,training_active=list(jobs),evaluation_active=evaluation[0] if evaluation else None,evaluated=evaluated,FRD_active=frd[0] if frd else None,FRD_completed=frd_done,failures=failures))
        if not jobs and not evaluation and not frd:
            if len(records)==2:
                keys=('record_sha256','FM_initial_sha256','task_initial_sha256','tau','target_slots','crop_start','target_ids')
                same=all(all(a[k]==b[k] for k in keys) for a,b in zip(records[ARMS[0]],records[ARMS[1]]))
                atomic('matched_streams.json',dict(matched=same,compared_steps=min(map(len,records.values()))))
                if not same:failures.append('matched_streams')
            report();atomic('queue_finished.json',dict(completed=len(trained)==2 and len(evaluated)==6 and len(frd_done)==2 and not failures,failures=failures,time=time.time()));break
        time.sleep(30)
if __name__=='__main__':main()
