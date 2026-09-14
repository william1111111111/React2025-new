"""Finite three-arm queue. Quiet status snapshots; no automatic extension or push."""
import json,os,subprocess,sys,time,fcntl
from pathlib import Path
from .controlled import OUT,MODES

def write(obj):
 obj['updated_unix']=time.time();p=OUT/'queue_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)
def run(module,args,tag,gpu='7'):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
 with (OUT/(tag+'.log')).open('a') as log:subprocess.run([sys.executable,'-m',module,*args],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
def main():
 lock=(OUT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  deadline=time.time()+7200
  while len(list((OUT/'dev_source').glob('*/result.json')))<80:
   write({'stage':'waiting_dev_alignment','completed_sources':len(list((OUT/'dev_source').glob('*/result.json'))),'required':80})
   if time.time()>deadline:raise RuntimeError('DEV preparation deadline reached; no training launched')
   time.sleep(10)
  if not (OUT/'DEV_READY.json').exists():
   write({'stage':'dev_lag_and_cache'});run('semantic_supervision.experiments.finalize_dev',[],'dev_finalize')
  ready=json.loads((OUT/'DEV_READY.json').read_text());assert ready['source_records']==80 and ready['non_null_sources']>0 and not ready['listener_used_for_semantics']
  smoke=OUT/'diagnostics/E_event/finished.json';assert smoke.exists() and json.loads(smoke.read_text())['steps']==1
  for arm in MODES:
   write({'stage':'training','arm':arm,'steps':500});run('semantic_supervision.experiments.controlled',['--arm',arm,'--steps','500'],arm+'_train')
   write({'stage':'official_dev80','arm':arm});run('semantic_supervision.experiments.evaluate',['--arm',arm],arm+'_eval')
  write({'stage':'exact_frd20','models':list(MODES)})
  for arm in MODES:
   label=f'seed123_{arm}_step14500'
   # Same unchanged exact-FRD implementation, all 2000 candidate-target pairs.
   code="from reaction_flow.frd_shared import main; main(root="+repr(str(OUT))+",models="+repr(tuple(f'seed123_{a}_step14500' for a in MODES))+")"
   with (OUT/(arm+'_frd.log')).open('a') as log:
    subprocess.run([sys.executable,'-c',code,'--model',label,'--task',str(OUT/'task_multitarget'/(label+'.json')),'--seconds','86400','--max-pairs','2000'],stdout=log,stderr=subprocess.STDOUT,check=True)
   statuses=sorted((OUT/'frd20'/label).glob('status_*.json'))
   if not statuses or not json.loads(statuses[-1].read_text())['completed']:raise RuntimeError('Exact FRD incomplete; no final score claimed')
  write({'stage':'finished','arms':list(MODES),'automatic_extension':False})
 except Exception as e:
  write({'stage':'failed','error_type':type(e).__name__,'message':str(e)[:300]});raise
if __name__=='__main__':main()
