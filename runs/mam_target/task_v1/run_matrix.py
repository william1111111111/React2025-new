"""Bounded two-arm queue; shared GPU0, CPU exact FRD, no other jobs touched."""
import os,sys,time,json,subprocess,hashlib
from pathlib import Path
root=Path('runs/mam_target/task_v1');env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
assert json.loads((root/'resume_regression.json').read_text())['max_parameter_error']==0
assert not (root/'matrix_started.json').exists()
def write(p,x):p.write_text(json.dumps(x,indent=2))
write(root/'matrix_started.json',dict(time=time.time(),budget={'R1':6000,'R2':6000},seed=123,gpu=0,checkpoints=[2000,6000],source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('mam_target').glob('*.py'))}))
def launch(name,args):
 log=open(root/f'{name}.txt','a');cmd=[sys.executable]+args;p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
 write(root/f'{name}_command.json',dict(command=cmd,pid=p.pid,time=time.time()));return dict(process=p,log=log,name=name)
def finish(job):
 job['log'].close();code=job['process'].returncode;write(root/f"{job['name']}_exit.json",dict(code=code,time=time.time()));return code
train={a:launch('train_'+a,['-m','mam_target.train','--arm',a]) for a in ('R1','R2')}
points=[(a,s) for s in (2000,6000) for a in ('R1','R2')];eval_done=set();frd_done=set();eval_job=None;frd_jobs={};failed=[]
while True:
 for arm,job in list(train.items()):
  if job['process'].poll() is not None:
   code=finish(job);del train[arm]
   if code:failed.append('train_'+arm)
 if eval_job and eval_job['process'].poll() is not None:
  code=finish(eval_job);point=eval_job['point']
  if code:failed.append(eval_job['name'])
  else:eval_done.add(point)
  eval_job=None
 for point,job in list(frd_jobs.items()):
  if job['process'].poll() is not None:
   code=finish(job);del frd_jobs[point]
   if code:failed.append(job['name'])
   else:frd_done.add(point)
 if not eval_job:
  for arm,step in points:
   name=f'eval_{arm}_{step}';marker=root/arm/'attempt_000/checkpoints'/f'step_{step:06d}.json'
   if marker.exists() and (arm,step) not in eval_done and name not in failed:
    checkpoint=json.loads(marker.read_text())
    assert hashlib.sha256(Path(checkpoint['path']).read_bytes()).hexdigest()==checkpoint['sha256']
    eval_job=launch(name,['-m','mam_target.evaluate','--checkpoint',checkpoint['path']]);eval_job['point']=(arm,step);break
 for arm,step in eval_done:
  point=(arm,step);name=f'frd_{arm}_{step}';label=f'seed123_{arm}_step{step}'
  if point not in frd_done and point not in frd_jobs and name not in failed:
   frd_jobs[point]=launch(name,['-m','mam_target.frd','--model',label,'--task',str(root/'task_multitarget'/f'{label}.json')])
 progress={}
 for arm in ('R1','R2'):
  log=root/arm/'attempt_000/training.jsonl'
  if log.exists():
   lines=log.read_text().splitlines()
   try:row=json.loads(lines[-1]);progress[arm]={'step':row['step'],'loss':row['loss']}
   except (IndexError,json.JSONDecodeError):pass
 write(root/'monitor_latest.json',dict(time=time.time(),training=progress,training_active=list(train),eval_active=eval_job['name'] if eval_job else None,eval_completed=[list(x) for x in sorted(eval_done)],frd_completed=[list(x) for x in sorted(frd_done)],failures=failed,completed=len(frd_done)==4 and not failed))
 if not train and not eval_job and not frd_jobs:
  if len(frd_done)==4 or failed:break
 time.sleep(20)
write(root/'matrix_finished.json',dict(time=time.time(),completed=len(frd_done)==4 and not failed,failures=failed))
