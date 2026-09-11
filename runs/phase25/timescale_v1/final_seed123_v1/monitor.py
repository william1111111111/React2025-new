import json,time,datetime
from pathlib import Path
r=Path('/home/zhengshiyi/react2025_new/runs/phase25/timescale_v1/final_seed123_v1')
with (r/'monitor_history.jsonl').open('x') as f:
 while True:
  es=len(list((r/'distribution').glob('*_K32.json')));tasks=len(list((r/'task_multitarget').glob('*.json')))
  d=dict(time=datetime.datetime.now().isoformat(),distribution_cases=es,requested_distribution_cases=48,missing_task_cases_done=tasks,requested_missing_tasks=2,alerts=[])
  for p in [r/'es_stdout.txt',r/'C0_task4000.txt',r/'C0_task6000.txt',r/'tasks_stdout.txt']:
   if p.exists() and 'Traceback (most recent call last)' in p.read_text():d['alerts'].append(str(p))
  d['complete']=es==48 and tasks==2
  f.write(json.dumps(d)+'\n');f.flush();p=r/'monitor_latest.tmp';p.write_text(json.dumps(d,indent=2));p.replace(r/'monitor_latest.json')
  if d['complete']:break
  time.sleep(30)
