"""One-time scheduler handoff: pause driver only, keep C1 child training."""
import os,signal,subprocess,sys,json,time
from pathlib import Path
root=Path('/home/zhengshiyi/react2025_new');os.chdir(root)
r=Path('runs/phase25/timescale_v1');parent=1838161
cmdline=Path(f'/proc/{parent}/cmdline').read_bytes().replace(b'\0',b' ').decode()
if '-m hirp.run_phase25 --device cuda:0 --seed 123 --through 2000 --evaluate' not in cmdline:raise RuntimeError('driver identity mismatch')
if (r/'seed_123/C2').exists():raise RuntimeError('C2 already started; no parallel launch')
cmd=[str(root/'.venv/bin/python'),'-m','hirp.train_phase25','--run',str(r/'seed_123'),'--device','cuda:0','--arm','C2','--stop-after','2000']
state=dict(parent_pid=parent,parent_command=cmdline,command=cmd,started=time.time(),policy='pause only scheduler; C1 child continues; resume scheduler in finally',completed=False)
with (r/'parallel_c2_launch.json').open('x') as f:json.dump(state,f,indent=2)
os.kill(parent,signal.SIGSTOP)
try:
 with (r/'parallel_c2_training.txt').open('x') as f:
  result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 state['returncode']=result.returncode
 state['completed']=result.returncode==0
finally:
 os.kill(parent,signal.SIGCONT)
 state['ended']=time.time()
 with (r/'parallel_c2_finished.json').open('x') as f:json.dump(state,f,indent=2)
