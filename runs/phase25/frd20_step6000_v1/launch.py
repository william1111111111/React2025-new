import os,sys,json,time,subprocess
from pathlib import Path
out=Path('runs/phase25/frd20_step6000_v1')
env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1')
children=[]
for arm in ['C0','C1','C2']:
 task=Path('runs/phase25/timescale_v1/final_seed123_v1/task_multitarget')/f'seed123_{arm}_step6000.json'
 if not task.exists():task=Path('runs/phase25/timescale_v1/task_multitarget')/task.name
 cmd=[sys.executable,str(out/'compute.py'),'--model',f'seed123_{arm}_step6000','--task',str(task)]
 log=open(out/f'{arm}_stdout.txt','a')
 child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
 (out/f'{arm}_command.json').write_text(json.dumps(dict(command=cmd,pid=child.pid,start=time.time(),threads=1),indent=2))
 children.append((arm,child,log))
for arm,child,log in children:
 code=child.wait();log.close();(out/f'{arm}_exit.json').write_text(json.dumps(dict(exit_code=code,end=time.time())))
