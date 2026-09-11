"""Bounded phase-A training/evaluation on GPU1, no automatic phase-B gate bypass."""
import os,time,json,subprocess,hashlib,argparse
from pathlib import Path
from .config import ROOT as r

def write(p,x):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(x,indent=2));temp.replace(p)

def main(adopt=False):
    assert json.loads((r/'gpu_regression.json').read_text())['resume_max_parameter_error']==0
    assert json.loads((r/'task_rollout_probe.json').read_text())['velocity_gradient_norm']>0
    assert (r/'queue_started.json').exists() if adopt else not (r/'queue_started.json').exists()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1')
    write(r/('queue_adopted.json' if adopt else 'queue_started.json'),dict(time=time.time(),physical_gpu=1,budget=12000,stage_B='not yet launched; inspect A12000 first',code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('reaction_flow').rglob('*.py')}))
    def launch(name,args):
        if adopt:name=name+'_math_v2'
        f=(r/f'{name}.txt').open('x');cmd=['.venv/bin/python']+args;p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT);write(r/f'{name}_command.json',dict(command=cmd,pid=p.pid,time=time.time()));return dict(name=name,p=p,log=f)
    def finish(job):
        code=job['p'].returncode
        if job['log'] is not None:job['log'].close()
        write(r/f"{job['name']}_exit.json",dict(code=code,time=time.time()));return code
    class AdoptedTraining:
        returncode=None
        def poll(self):
            folder=r/'A/attempt_000'
            if (folder/'status.json').exists():self.returncode=0 if json.loads((folder/'status.json').read_text())['completed'] else 1
            elif (folder/'failure.json').exists():self.returncode=1
            return self.returncode
    training=dict(name='adopted_train_A',p=AdoptedTraining(),log=None) if adopt else launch('train_A',['-m','reaction_flow.train']);evaluation=None;done=[];failed=[];frd=None;frd_done=False;frd_parts=0;iteration=0
    while True:
        if training and training['p'].poll() is not None:
            if finish(training):failed.append('train_A')
            training=None
        if evaluation and evaluation['p'].poll() is not None:
            if finish(evaluation):failed.append(evaluation['name'].removesuffix('_math_v2'))
            else:done.append(evaluation['step'])
            evaluation=None
        if not evaluation:
            for s in (2000,6000,12000):
                side=r/'A/attempt_000/checkpoints'/f'step_{s:06d}.json';name=f'eval_A_{s}'
                if side.exists() and s not in done and name not in failed:
                    cp=json.loads(side.read_text());assert hashlib.sha256(Path(cp['path']).read_bytes()).hexdigest()==cp['sha256'];evaluation=launch(name,['-m','reaction_flow.evaluate','--checkpoint',cp['path']]);evaluation['step']=s;break
        if frd and frd['p'].poll() is not None:
            if finish(frd):failed.append('frd_A')
            else:
                statuses=sorted((r/'frd20/seed123_A_step12000').glob('status_*.json'));frd_done=bool(statuses and json.loads(statuses[-1].read_text())['completed'])
                if not frd_done and frd_parts>=4:failed.append('frd_A_budget_incomplete')
            frd=None
        if 12000 in done and not frd and not frd_done and frd_parts<4 and 'frd_A' not in failed:
            frd_parts+=1;frd=launch(f'frd_A_part{frd_parts}',['-m','reaction_flow.frd','--model','seed123_A_step12000','--task',str(r/'task_multitarget/seed123_A_step12000.json')])
        log=r/'A/attempt_000/training.jsonl';row={}
        if log.exists():
            try:row=json.loads(log.read_text().splitlines()[-1])
            except (IndexError,json.JSONDecodeError):pass
        completed=not training and len(done)==3 and frd_done and not failed
        write(r/'monitor_latest.json',dict(time=time.time(),training_step=row.get('phase_step',0),FM=row.get('FM'),training_active=bool(training),eval_completed=done,eval_active=evaluation['name'] if evaluation else None,FRD_active=bool(frd),FRD_completed=frd_done,failures=failed,phase_A_completed=completed,phase_B_started=False))
        if iteration%15==0:
            with (r/'report_stdout.txt').open('a') as f:subprocess.run(['.venv/bin/python','-m','reaction_flow.report'],env=env,stdout=f,stderr=subprocess.STDOUT)
        if not training and not evaluation and not frd and (completed or failed):break
        iteration+=1;time.sleep(20)
    write(r/'phase_A_finished.json',dict(time=time.time(),completed=completed,failures=failed,phase_B_requires='inspect pure-noise quality/activity then fixed F-cont/F-task comparison'))
    subprocess.run(['.venv/bin/python','-m','reaction_flow.report'],env=env,check=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--adopt',action='store_true');a=p.parse_args();main(a.adopt)
