"""Fixed two-worker GPU queue plus bounded independent CPU FRD tasks."""
import concurrent.futures as cf,json,subprocess,sys,time,os,threading
from pathlib import Path
sys.path.insert(0,'/home/zhengshiyi/react2025_new')
from hirp.phase15_audit import sha256_file,canonical_hash
root=Path('/home/zhengshiyi/react2025_new');os.chdir(root)
r=root/'runs/phase25/timescale_v1';out=r/'parallel_budget_v1';py=str(root/'.venv/bin/python');lock=threading.Lock()
def save(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2)
def invoke(label,cmd,env=None):
 save(out/(label+'_command.json'),dict(command=cmd,time=time.time()))
 with (out/(label+'.txt')).open('x') as f:
  result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
 save(out/(label+'_exit.json'),dict(returncode=result.returncode,time=time.time()))
 if result.returncode:raise RuntimeError(label+' failed; see log')
def arm_job(seed,arm):
 run=r/f'seed_{seed}';label=f'seed{seed}_{arm}'
 paths=list((run/arm).glob('attempt_*/checkpoints/step_*.pt'))
 cp=max(paths,key=lambda p:int(p.stem.split('_')[1])) if paths else None
 step=int(cp.stem.split('_')[1]) if cp else 0
 if step<6000:
  # Serialized admission avoids two workers both consuming the same free-memory allowance.
  with lock:
   deadline=time.time()+21600
   while True:
    free=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free>=5500:break
    if time.time()>deadline:raise RuntimeError('GPU admission timeout; budget unchanged')
    time.sleep(10)
   cmd=[py,'-m','hirp.train_phase25','--run',str(run),'--device','cuda:0','--arm',arm,'--init-seed',str(seed),'--sampler-seed',str(seed),'--noise-seed',str(seed),'--stop-after','6000']
   if cp:cmd+=['--resume',str(cp)]
   save(out/(label+'_train_command.json'),dict(command=cmd,time=time.time(),admission_free_MiB=free))
   log=(out/(label+'_train.txt')).open('x');proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
   time.sleep(15) # allow CUDA allocation before admitting another worker
  code=proc.wait();log.close();save(out/(label+'_train_exit.json'),dict(returncode=code,time=time.time()))
  if code:raise RuntimeError(label+' training failed')
 for step in (128,500,1000,2000,4000,6000):
  dest=r/'task_multitarget'/f'seed{seed}_{arm}_step{step}.json'
  paths=sorted((run/arm).glob(f'attempt_*/checkpoints/step_{step:06d}.pt'))
  if not paths:raise RuntimeError('missing fixed checkpoint')
  if dest.exists():
   cached=json.loads(dest.read_text());assert canonical_hash(cached['result'])==cached['result_sha256'];assert cached['eval_identity']['checkpoint_sha256']==sha256_file(paths[0])
   save(out/f'{label}_eval{step}_already_completed.json',dict(path=str(dest),sha256=sha256_file(dest),note='preserve prior immutable evaluation identity; no relabelling'))
   continue
  invoke(f'{label}_eval{step}',[py,'-m','hirp.task_phase25','--checkpoint',str(paths[0]),'--device','cuda:0'])
 return label
def frd(arm):
 label=f'seed123_{arm}_step2000';env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
 invoke('frd_'+arm,[py,'-m','hirp.frd_phase25','--model',label,'--task',str(r/'task_multitarget'/f'{label}.json')],env)
 return label
save(out/'protocol.json',dict(gpu=0,max_gpu_jobs=2,min_free_MiB_at_training_admission=5500,training=[dict(seed=s,arm=a,max_steps=6000) for s in (123,42,2026) for a in ('C0','C1','C2')],cpu_frd_jobs=3,source_snapshot={str(p.relative_to(root)):sha256_file(p) for p in sorted((root/'hirp').rglob('*.py'))},budget='unchanged frozen6000, no extra lambda/noise banks',started=time.time()))
errors=[]
with cf.ThreadPoolExecutor(max_workers=2) as gpu,cf.ThreadPoolExecutor(max_workers=3) as cpu:
 futures=[gpu.submit(arm_job,s,a) for s in (123,42,2026) for a in ('C0','C1','C2')]+[cpu.submit(frd,a) for a in ('C0','C1','C2')]
 for f in cf.as_completed(futures):
  try:print('DONE',f.result(),flush=True)
  except Exception as exc:errors.append(str(exc));print('ERROR',str(exc),flush=True)
save(out/'finished.json',dict(completed=not errors,errors=errors,time=time.time()))
