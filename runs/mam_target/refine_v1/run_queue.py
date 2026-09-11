import os,json,time,subprocess,hashlib
from pathlib import Path
r=Path('runs/mam_target/refine_v1');env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1')
assert json.loads((r/'gpu_regression.json').read_text())['resume_parameter_max_error']==0
assert not (r/'queue_started.json').exists()
def write(p,x):p.write_text(json.dumps(x,indent=2))
write(r/'queue_started.json',dict(time=time.time(),physical_gpu=1,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ('mam_target','mam_refine') for p in Path(folder).glob('*.py')},additional_steps_per_arm=2000))
def launch(name,args):
 f=open(r/f'{name}.txt','x');cmd=['.venv/bin/python']+args;p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT);write(r/f'{name}_command.json',dict(command=cmd,pid=p.pid,time=time.time(),physical_gpu=1));return dict(name=name,process=p,log=f)
def finish(j):
 code=j['process'].returncode;j['log'].close();write(r/f"{j['name']}_exit.json",dict(code=code,time=time.time()));return code
training={arm:launch('train_'+arm,['-m','mam_refine.train','--arm',arm]) for arm in ('R2-cont','R3-cover')};points=[(a,s) for s in (6500,8000) for a in training];evaluation=None;eval_done=set();frds={};frd_done=set();failed=[]
while True:
 for arm,j in list(training.items()):
  if j['process'].poll() is not None:
   if finish(j):failed.append(j['name'])
   del training[arm]
 if evaluation and evaluation['process'].poll() is not None:
  if finish(evaluation):failed.append(evaluation['name'])
  else:eval_done.add(evaluation['point'])
  evaluation=None
 for pt,j in list(frds.items()):
  if j['process'].poll() is not None:
   if finish(j):failed.append(j['name'])
   else:frd_done.add(pt)
   del frds[pt]
 if not evaluation:
  for arm,step in points:
   marker=r/arm/'attempt_000/checkpoints'/f'step_{step:06d}.json';name=f'eval_{arm}_{step}'
   if marker.exists() and (arm,step) not in eval_done and name not in failed:
    cp=json.loads(marker.read_text());assert hashlib.sha256(Path(cp['path']).read_bytes()).hexdigest()==cp['sha256'];evaluation=launch(name,['-m','mam_refine.evaluate','--checkpoint',cp['path']]);evaluation['point']=(arm,step);break
 for pt in eval_done:
  arm,step=pt;name=f'frd_{arm}_{step}'
  if step==8000 and pt not in frd_done and pt not in frds and name not in failed:
   label=f'seed123_{arm}_step{step}';frds[pt]=launch(name,['-m','mam_refine.frd','--model',label,'--task',str(r/'task_multitarget'/f'{label}.json')])
 progress={}
 for arm in ('R2-cont','R3-cover'):
  log=r/arm/'attempt_000/training.jsonl'
  if log.exists():
   try:
    row=json.loads(log.read_text().splitlines()[-1]);progress[arm]=dict(additional_steps=row['additional_step'],global_step=row['step'],loss=row['loss'])
   except (IndexError,json.JSONDecodeError):pass
 write(r/'monitor_latest.json',dict(time=time.time(),physical_gpu=1,training=progress,training_active=list(training),eval_active=evaluation['name'] if evaluation else None,eval_completed=[list(x) for x in sorted(eval_done)],frd_completed=[list(x) for x in sorted(frd_done)],failures=failed,completed=len(frd_done)==2 and not failed))
 if not training and not evaluation and not frds and (len(frd_done)==2 or failed):break
 time.sleep(20)
write(r/'queue_finished.json',dict(time=time.time(),completed=len(frd_done)==2 and not failed,failures=failed))
with open(r/'report_stdout.txt','x') as f:subprocess.run(['.venv/bin/python','-m','mam_refine.report'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
