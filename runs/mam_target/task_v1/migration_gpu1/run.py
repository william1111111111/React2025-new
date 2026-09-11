import os,signal,time,json,subprocess,hashlib
from pathlib import Path
root=Path('runs/mam_target/task_v1');out=root/'migration_gpu1';env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
def write(p,x):p.write_text(json.dumps(x,indent=2))
def launch(name,args):
 cmd=['.venv/bin/python']+args;f=open(out/f'{name}.txt','x');p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
 write(out/f'{name}_command.json',dict(command=cmd,pid=p.pid,physical_gpu=1,time=time.time()));return dict(name=name,process=p,log=f)
def finish(j):
 code=j['process'].returncode;j['log'].close();write(out/f"{j['name']}_exit.json",dict(code=code,time=time.time()));return code
r1=json.loads((root/'R1/attempt_000/status.json').read_text());assert r1['actual_steps']==6000 and r1['completed']
points=[('R1',2000),('R1',6000),('R2',2000),('R2',6000)];eval_done=set();frd_done=set();frds={};evaluation=None;training=None;migrated=False;failed=[]
write(out/'started.json',dict(time=time.time(),budget_unchanged=6000,R2_resume_step=500,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('mam_target').glob('*.py')}))
while True:
 marker=root/'R2/attempt_000/checkpoints/step_000500.json'
 if not migrated and marker.exists():
  cp=json.loads(marker.read_text());assert hashlib.sha256(Path(cp['path']).read_bytes()).hexdigest()==cp['sha256']
  pid=610716;cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode();assert 'mam_target.train --arm R2' in cmd
  os.kill(pid,signal.SIGSTOP);time.sleep(.2)
  rows=(root/'R2/attempt_000/training.jsonl').read_text().splitlines();last=json.loads(rows[-1])['step']
  write(out/'R2_handoff.json',dict(old_pid=pid,resume_checkpoint=cp,last_logged_step=last,resume_step=500,logged_tail_replayed=max(0,last-500),old_attempt_completed=False,old_logs_preserved=True,partial_inflight_step_discarded=True,time=time.time()))
  os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
  for _ in range(100):
   try:
    state=Path(f'/proc/{pid}/stat').read_text().split()[2]
    if state=='Z':break
   except FileNotFoundError:break
   time.sleep(.1)
  else:raise RuntimeError('old R2 did not exit; do not duplicate')
  scheduler=610712;cmd=Path(f'/proc/{scheduler}/cmdline').read_bytes();assert b'run_matrix.py' in cmd
  os.kill(scheduler,signal.SIGTERM);os.kill(scheduler,signal.SIGCONT)
  training=launch('train_R2_gpu1',['-m','mam_target.train','--arm','R2','--resume',cp['path']]);migrated=True
 if training and training['process'].poll() is not None:
  if finish(training):failed.append(training['name'])
  training=None
 if evaluation and evaluation['process'].poll() is not None:
  if finish(evaluation):failed.append(evaluation['name'])
  else:eval_done.add(evaluation['point'])
  evaluation=None
 for pt,j in list(frds.items()):
  if j['process'].poll() is not None:
   if finish(j):failed.append(j['name'])
   else:frd_done.add(pt)
   del frds[pt]
 if evaluation is None:
  for arm,step in points:
   name=f'eval_{arm}_{step}';markers=sorted((root/arm).glob(f'attempt_*/checkpoints/step_{step:06d}.json'))
   if markers and (arm,step) not in eval_done and name not in failed:
    cp=json.loads(markers[-1].read_text());assert hashlib.sha256(Path(cp['path']).read_bytes()).hexdigest()==cp['sha256']
    evaluation=launch(name,['-m','mam_target.evaluate','--checkpoint',cp['path']]);evaluation['point']=(arm,step);break
 for pt in eval_done:
  arm,step=pt;name=f'frd_{arm}_{step}';label=f'seed123_{arm}_step{step}'
  if pt not in frd_done and pt not in frds and name not in failed:
   frds[pt]=launch(name,['-m','mam_target.frd','--model',label,'--task',str(root/'task_multitarget'/f'{label}.json')])
 progress={}
 for arm in ('R1','R2'):
  logs=sorted((root/arm).glob('attempt_*/training.jsonl'))
  if logs:
   try:
    row=json.loads(logs[-1].read_text().splitlines()[-1]);progress[arm]=dict(step=row['step'],loss=row['loss'])
   except (IndexError,json.JSONDecodeError):pass
 write(root/'monitor_latest.json',dict(time=time.time(),physical_gpu=1,R2_migrated=migrated,training=progress,training_active=['R2'] if training or not migrated else [],eval_active=evaluation['name'] if evaluation else None,eval_completed=[list(x) for x in sorted(eval_done)],frd_completed=[list(x) for x in sorted(frd_done)],failures=failed,completed=len(frd_done)==4 and not failed))
 if migrated and not training and not evaluation and not frds and (len(frd_done)==4 or failed):break
 time.sleep(2 if not migrated else 20)
write(root/'matrix_finished.json',dict(time=time.time(),completed=len(frd_done)==4 and not failed,failures=failed,physical_gpu=1))
