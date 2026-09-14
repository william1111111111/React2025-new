"""Finite one-GPU queue: four matched arms, 500 curves, full1000 and interventions."""
import argparse,fcntl,os,subprocess,sys,time
from .common import *
from .evaluate import label_for

def status(**kw):write(OUT/'queue_status.json',dict(**kw,updated_unix=time.time()))
def run(module,args,tag,gpu):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
 with (OUT/(tag+'.log')).open('a') as f:subprocess.run([sys.executable,'-m',module,*args],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def main(gpu):
 lock=open(OUT/'queue.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify();assert read(OUT/'ACCEPTANCE.json')['passed'];assert (OUT/'CODE_IDENTITY.json').exists()
 for p,h in read(OUT/'CODE_IDENTITY.json').items():assert sha(p)==h
 try:
  for arm in ARMS:
   for step in [500,1000]:
    # Resume completed step500->1000; do not reinterpret stop as extra steps.
    status(stage='training',arm=arm,stop=step,gpu=gpu)
    checkpoints=list((OUT/'training'/arm/'checkpoints').glob('step_*.json'));latest=max([read(p)['step'] for p in checkpoints],default=-1)
    if latest<step:run('semantic_supervision.bert_experiments.train',['--arm',arm,'--stop',str(step)],arm+'_train',gpu)
    status(stage='DEV80',arm=arm,step=step,gpu=gpu);run('semantic_supervision.bert_experiments.evaluate',['--arm',arm,'--step',str(step)],f'{arm}_eval{step}',gpu)
   run('semantic_supervision.bert_experiments.report',[],arm+'_report','')
  labels=tuple(label_for(a,1000) for a in ARMS)
  for arm,label in zip(ARMS,labels):
   status(stage='exact_FRD20',arm=arm)
   code='from reaction_flow.frd_shared import main; main(root='+repr(str(OUT))+',models='+repr(labels)+')'
   with (OUT/(arm+'_frd.log')).open('a') as f:subprocess.run([sys.executable,'-c',code,'--model',label,'--task',str(OUT/'task_multitarget'/(label+'.json')),'--seconds','86400','--max-pairs','2000'],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT,check=True)
   s=read(sorted((OUT/'frd20'/label).glob('status_*.json'))[-1]);assert s['completed'] and s['completed_pairs']==2000
  for arm in ARMS[2:]:
   for kind in ['text','time']:
    status(stage='fixed_intervention',arm=arm,kind=kind,gpu=gpu);run('semantic_supervision.bert_experiments.evaluate',['--arm',arm,'--step','1000','--perturb',kind],f'{arm}_{kind}_diagnostic',gpu)
  run('semantic_supervision.bert_experiments.report',[],'final_report','');assert read(OUT/'completion.json')['complete'];status(stage='finished',arms=ARMS,automatic_extension=False,pushed=False)
 except Exception as ex:
  status(stage='failed',error_type=type(ex).__name__,error=str(ex)[:500]);raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--gpu',default='7');a=p.parse_args();main(a.gpu)
