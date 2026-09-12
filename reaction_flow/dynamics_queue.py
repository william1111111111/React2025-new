"""Only two authorized 2000-step arms; no search, no other seeds."""
import json,os,time,subprocess
from pathlib import Path
from .task_dynamics_train import ROOT
from hirp.phase15_audit import sha256_file

def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def main():
    assert json.loads((ROOT/'resume_regression.json').read_text())['max_parameter_error']==0
    assert json.loads((ROOT/'TRAIN_precision_probe.json').read_text())['task_abs_delta']<1e-4
    assert (ROOT/'solver_diagnostic.json').exists() and (ROOT/'condition_probe.json').exists()
    assert not (ROOT/'queue_started.json').exists()
    atomic(ROOT/'queue_started.json',dict(time=time.time(),physical_gpu=1,arms=['T0-task','T1-task-dynamics'],new_steps=2000,code_hashes={str(p):sha256_file(p) for p in Path('reaction_flow').glob('*.py')}))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    def launch(name,args):
        log=(ROOT/f'{name}.txt').open('x');cmd=['.venv/bin/python','-m']+args;proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);atomic(ROOT/f'{name}_command.json',dict(pid=proc.pid,command=cmd,time=time.time()));return proc,log
    jobs={a:launch('train_'+a,['reaction_flow.task_dynamics_train','--arm',a]) for a in ('T0-task','T1-task-dynamics')};evaluated=[];failures=[];evaluation=None;frd=None;frd_done=[];frd_counts={};trained=[]
    while True:
        for arm,(p,f) in list(jobs.items()):
            if p.poll() is not None:
                f.close();atomic(ROOT/f'train_{arm}_exit.json',dict(code=p.returncode,time=time.time()));jobs.pop(arm)
                if p.returncode:failures.append(arm)
                else:trained.append(arm)
        if evaluation:
            arm,(p,f)=evaluation
            if p.poll() is not None:
                f.close();atomic(ROOT/f'eval_{arm}_exit.json',dict(code=p.returncode,time=time.time()))
                if p.returncode:failures.append('eval_'+arm)
                else:evaluated.append(arm)
                evaluation=None
        if evaluation is None:
            for arm in trained:
                if arm not in evaluated and 'eval_'+arm not in failures:
                    cp=ROOT/arm/'attempt_000/checkpoints/step_014000.pt';evaluation=(arm,launch('eval_'+arm,['reaction_flow.evaluate_dynamics','--checkpoint',str(cp)]));break
        if frd:
            arm,(p,f)=frd
            if p.poll() is not None:
                f.close();atomic(ROOT/f'frd_{arm}_{frd_counts[arm]}_exit.json',dict(code=p.returncode,time=time.time()));frd=None
                statuses=sorted((ROOT/'frd20'/f'seed123_{arm}_step14000').glob('status_*.json'))
                if p.returncode:failures.append('frd_'+arm)
                elif statuses and json.loads(statuses[-1].read_text())['completed']:frd_done.append(arm)
                elif frd_counts[arm]>=4:failures.append('frd_'+arm)
        if frd is None:
            for arm in evaluated:
                if arm not in frd_done and 'frd_'+arm not in failures:
                    frd_counts[arm]=frd_counts.get(arm,0)+1;name=f'seed123_{arm}_step14000';frd=(arm,launch(f'frd_{arm}_{frd_counts[arm]}',['reaction_flow.frd_dynamics','--model',name,'--task',str(ROOT/'task_multitarget'/f'{name}.json')]));break
        progress={}
        for arm in ('T0-task','T1-task-dynamics'):
            p=ROOT/arm/'attempt_000/training.jsonl'
            if p.exists():
                try:
                    row=json.loads(p.read_text().splitlines()[-1]);progress[arm]={k:row[k] for k in ('phase_step','FM','task','dynamic','gradient_norm','seconds')}
                except (IndexError,json.JSONDecodeError):pass
        atomic(ROOT/'monitor_latest.json',dict(time=time.time(),progress=progress,training_active=list(jobs),evaluation_active=evaluation[0] if evaluation else None,evaluated=evaluated,FRD_active=frd[0] if frd else None,FRD_completed=frd_done,failures=failures))
        if not jobs and not evaluation and not frd:break
        time.sleep(30)
    atomic(ROOT/'queue_finished.json',dict(completed=len(frd_done)==2 and not failures,failures=failures,time=time.time()))
if __name__=='__main__':main()
