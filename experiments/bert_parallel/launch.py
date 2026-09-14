"""Execution-only parallel supervisor. Frozen experiment code and schedules unchanged."""
import os,sys,time,json,fcntl,subprocess,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from semantic_supervision.bert_experiments.common import OUT,ARMS,read,write,verify
from semantic_supervision.bert_experiments.evaluate import label_for
OLD_PID=2047983
ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='7',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
def run(module,args,tag,cpu=False):
 with (OUT/(tag+'.log')).open('a') as f:subprocess.run([sys.executable,'-m',module,*args],cwd=ROOT,env=dict(ENV,CUDA_VISIBLE_DEVICES='') if cpu else ENV,stdout=f,stderr=subprocess.STDOUT,check=True)
def pipeline(arm):
 def status(stage,step=None):write(OUT/'parallel_status'/(arm+'.json'),dict(arm=arm,stage=stage,step=step,gpu=7,updated_unix=time.time()))
 try:
  if arm=='P0-null':
   status('existing_training',500)
   p=OUT/'training'/arm/'train.lock'
   while True:
    with p.open('a') as f:
     try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);break
     except BlockingIOError:time.sleep(5)
   assert (OUT/'training'/arm/'checkpoints/step_000500.json').exists(),'existing P0 did not complete500'
   # Old supervisor is stopped; remove it only after its original child finishes.
   try:
    cmd=Path(f'/proc/{OLD_PID}/cmdline').read_bytes()
    if b'semantic_supervision.bert_experiments.queue' in cmd:os.kill(OLD_PID,9)
   except FileNotFoundError:pass
  for step in [500,1000]:
   files=list((OUT/'training'/arm/'checkpoints').glob('step_*.json'));latest=max([read(p)['step'] for p in files],default=-1)
   if latest<step:status('training',step);run('semantic_supervision.bert_experiments.train',['--arm',arm,'--stop',str(step)],arm+'_train')
   status('DEV80',step);run('semantic_supervision.bert_experiments.evaluate',['--arm',arm,'--step',str(step)],f'{arm}_eval{step}')
  status('exact_FRD20',1000);labels=tuple(label_for(a,1000) for a in ARMS);label=label_for(arm,1000)
  # Shared FRD protocol creation serialized; subsequent CPU calculations may overlap.
  with (OUT/'parallel_frd.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   code='from reaction_flow.frd_shared import main; main(root='+repr(str(OUT))+',models='+repr(labels)+')'
   with (OUT/(arm+'_frd.log')).open('a') as f:subprocess.run([sys.executable,'-c',code,'--model',label,'--task',str(OUT/'task_multitarget'/(label+'.json')),'--seconds','86400','--max-pairs','2000'],cwd=ROOT,env=dict(ENV,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT,check=True)
  assert read(sorted((OUT/'frd20'/label).glob('status_*.json'))[-1])['completed']
  if arm in ARMS[2:]:
   for kind in ['text','time']:status('intervention_'+kind,1000);run('semantic_supervision.bert_experiments.evaluate',['--arm',arm,'--step','1000','--perturb',kind],f'{arm}_{kind}_diagnostic')
  status('finished',1000)
 except Exception as e:status('failed');write(OUT/'parallel_status'/(arm+'_error.json'),dict(error=str(e)));raise

def main():
 lock=(OUT/'parallel.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
 cmd=Path(f'/proc/{OLD_PID}/cmdline').read_bytes();assert b'semantic_supervision.bert_experiments.queue' in cmd
 os.kill(OLD_PID,19) # SIGSTOP supervisor only; current P0 subprocess continues.
 write(OUT/'queue_status.json',dict(stage='parallel',arms=ARMS,gpus=[7],old_supervisor_paused=OLD_PID,updated_unix=time.time()))
 write(OUT/'PARALLEL_EXECUTION.json',dict(change='serial -> four concurrent arm pipelines on GPU7',model_protocol_changed=False,existing_P0_preserved=True,script=str(Path(__file__)),started_unix=time.time()))
 failures=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  jobs={pool.submit(pipeline,a):a for a in ARMS}
  for f in concurrent.futures.as_completed(jobs):
   try:f.result()
   except Exception as e:failures.append(dict(arm=jobs[f],error=str(e)))
 if failures:write(OUT/'queue_status.json',dict(stage='failed',errors=failures));return
 run('semantic_supervision.bert_experiments.report',[],'final_report',True);assert read(OUT/'completion.json')['complete'];write(OUT/'queue_status.json',dict(stage='finished',arms=ARMS,automatic_extension=False,pushed=False))
if __name__=='__main__':main()
