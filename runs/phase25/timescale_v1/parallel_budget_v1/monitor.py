import json,time,datetime,subprocess
from pathlib import Path
r=Path('/home/zhengshiyi/react2025_new/runs/phase25/timescale_v1');out=r/'parallel_budget_v1'
with (out/'monitor_history.jsonl').open('x') as log:
 while True:
  d=dict(time=datetime.datetime.now().isoformat(),training={},eval_count=len(list((r/'task_multitarget').glob('*.json'))),frd_pairs={},alerts=[])
  for seed in (123,42,2026):
   for arm in ('C0','C1','C2'):
    label=f'{seed}_{arm}';paths=sorted((r/f'seed_{seed}'/arm).glob('attempt_*/training.jsonl'));d['training'][label]=0
    if paths:
     lines=paths[-1].read_text().splitlines()
     for line in reversed(lines):
      try:last=json.loads(line);break
      except json.JSONDecodeError:continue
     else:continue
     d['training'][label]=last['step']
     if last['step']<6000 and time.time()-paths[-1].stat().st_mtime>600 and len(paths)>1:d['alerts'].append(label+' log stale >10min')
    for p in (r/f'seed_{seed}'/arm).glob('attempt_*/failure.json'):d['alerts'].append(str(p))
  for a in ('C0','C1','C2'):
   d['frd_pairs'][a]=len(list((r/'frd20'/f'seed123_{a}_step2000').glob('[0-9]*.json')))
  for p in out.glob('*_exit.json'):
   if json.loads(p.read_text())['returncode']:d['alerts'].append(str(p))
  try:d['gpu0']=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader'],text=True,timeout=10).strip()
  except Exception as exc:d['gpu_error']=str(exc)
  log.write(json.dumps(d)+'\n');log.flush();p=out/'monitor_latest.tmp';p.write_text(json.dumps(d,indent=2));p.replace(out/'monitor_latest.json')
  if (out/'finished.json').exists():break
  time.sleep(30)
