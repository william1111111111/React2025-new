"""One bounded run: three arms, three fixed checkpoints, final FRD20 only."""
import os,json,time,subprocess,hashlib
from pathlib import Path
from mam_staged.common import ROOT as r
ARMS=('S0_quality','S1_joint','S2_staged')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1')
def write(p,x):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(x,indent=2));temp.replace(p)
def launch(name,args):
    f=(r/f'{name}.txt').open('x');cmd=['.venv/bin/python']+args;p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
    write(r/f'{name}_command.json',dict(command=cmd,pid=p.pid,time=time.time(),physical_gpu=1));return dict(name=name,process=p,log=f)
def finish(j):
    code=j['process'].returncode;j['log'].close();write(r/f"{j['name']}_exit.json",dict(code=code,time=time.time()));return code

def main():
    assert json.loads((r/'gpu_regression.json').read_text())['resume_parameter_max_error']==0
    assert json.loads((r/'first_step_regression.json').read_text())['passed']
    assert not (r/'queue_started.json').exists()
    write(r/'queue_started.json',dict(time=time.time(),physical_gpu=1,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ('mam_target','mam_refine','mam_staged') for p in Path(folder).glob('*.py')},additional_steps_per_arm=2000,FRD_max_invocations_per_arm=4,FRD_seconds_per_invocation=7200))
    training={a:launch('train_'+a,['-m','mam_staged.train','--arm',a]) for a in ARMS}
    points=[(a,s) for s in (6500,7000,8000) for a in ARMS];evaluation=None;eval_done=set();frds={};frd_done=set();failed=[];attempts={};iteration=0
    while True:
        for a,j in list(training.items()):
            if j['process'].poll() is not None:
                if finish(j):failed.append(j['name'])
                del training[a]
        if evaluation and evaluation['process'].poll() is not None:
            if finish(evaluation):failed.append(evaluation['name'])
            else:eval_done.add(evaluation['point'])
            evaluation=None
        for pt,j in list(frds.items()):
            if j['process'].poll() is not None:
                if finish(j):failed.append('frd_'+pt[0])
                else:
                    st=sorted((r/'frd20'/f'seed123_{pt[0]}_step8000').glob('status_*.json'))
                    if st and json.loads(st[-1].read_text())['completed']:frd_done.add(pt)
                    elif attempts[pt]>=4:failed.append('frd_'+pt[0]+'_budget_incomplete')
                del frds[pt]
        if not evaluation:
            for a,s in points:
                name=f'eval_{a}_{s}';marker=r/a/'attempt_000/checkpoints'/f'step_{s:06d}.json'
                if marker.exists() and (a,s) not in eval_done and name not in failed:
                    cp=json.loads(marker.read_text());assert hashlib.sha256(Path(cp['path']).read_bytes()).hexdigest()==cp['sha256']
                    evaluation=launch(name,['-m','mam_staged.evaluate','--checkpoint',cp['path']]);evaluation['point']=(a,s);break
        for pt in sorted(eval_done):
            a,s=pt
            if s==8000 and pt not in frd_done and pt not in frds and attempts.get(pt,0)<4 and 'frd_'+a not in failed:
                attempts[pt]=attempts.get(pt,0)+1;label=f'seed123_{a}_step{s}'
                frds[pt]=launch(f'frd_{a}_{s}_part{attempts[pt]}',['-m','mam_staged.frd','--model',label,'--task',str(r/'task_multitarget'/f'{label}.json')])
        progress={}
        for a in ARMS:
            log=r/a/'attempt_000/training.jsonl'
            if log.exists():
                try:
                    row=json.loads(log.read_text().splitlines()[-1]);progress[a]=dict(additional_steps=row['additional_step'],global_step=row['step'],loss=row['loss'])
                except (IndexError,json.JSONDecodeError):pass
        complete=len(frd_done)==3 and len(eval_done)==9 and not failed
        write(r/'monitor_latest.json',dict(time=time.time(),physical_gpu=1,training=progress,training_active=list(training),eval_active=evaluation['name'] if evaluation else None,eval_completed=[list(x) for x in sorted(eval_done)],frd_active=[list(x) for x in frds],frd_completed=[list(x) for x in sorted(frd_done)],failures=failed,completed=complete))
        if iteration%15==0:
            with (r/'report_stdout.txt').open('a') as f:subprocess.run(['.venv/bin/python','-m','mam_staged.report'],env=env,stdout=f,stderr=subprocess.STDOUT)
        if not training and not evaluation and not frds and (complete or failed):break
        iteration+=1;time.sleep(20)
    write(r/'queue_finished.json',dict(time=time.time(),completed=complete,failures=failed))
    subprocess.run(['.venv/bin/python','-m','mam_staged.report'],env=env,check=True)
if __name__=='__main__':main()
