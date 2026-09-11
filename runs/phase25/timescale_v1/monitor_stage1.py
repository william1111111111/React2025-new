"""Read-only 30-second monitoring; no restart or budget decisions."""
import datetime,json,math,subprocess,time
from pathlib import Path
r=Path('/home/zhengshiyi/react2025_new/runs/phase25/timescale_v1')
out=r/'monitor_stage1';out.mkdir(exist_ok=False)
def finite(x):
 if isinstance(x,float):return math.isfinite(x)
 if isinstance(x,dict):return all(finite(v) for v in x.values())
 if isinstance(x,list):return all(finite(v) for v in x)
 return True
started=time.time()
with (out/'history.jsonl').open('x') as log:
 while True:
  now=time.time();state=dict(time=datetime.datetime.now().isoformat(),arms={},alerts=[],evaluations=[])
  for arm in ('C0','C1','C2'):
   paths=sorted((r/'seed_123'/arm).glob('attempt_*/training.jsonl'));item=dict(actual_steps=0)
   if paths:
    p=paths[-1];rows=[]
    for line in p.read_text().splitlines():
     try:rows.append(json.loads(line))
     except json.JSONDecodeError:continue # writer may be in its final partial line
    if rows:
     last=rows[-1];item.update(actual_steps=last['step'],rows=len(rows),conditional=last['conditional']['loss'],total=last['loss'],log_age_seconds=now-p.stat().st_mtime)
     if not all(finite(x) for x in rows):state['alerts'].append(arm+': nonfinite training record')
     if item['actual_steps']!=len(rows):state['alerts'].append(arm+': step/row mismatch')
     if item['actual_steps']<2000 and item['log_age_seconds']>300:state['alerts'].append(arm+': no training log update for 5min')
   for p in (r/'seed_123'/arm).glob('attempt_*/failure.json'):state['alerts'].append(str(p))
   state['arms'][arm]=item
  expected=[r/'task_multitarget'/f'seed123_{a}_step{s}.json' for a in ('C0','C1','C2') for s in (128,500,1000,2000)]
  for p in expected:
   if p.exists():
    try:
     v=json.loads(p.read_text())['result'];state['evaluations'].append(dict(file=p.name,FRC=v['multi_target']['FRC']))
     if not finite(v):state['alerts'].append(p.name+': nonfinite evaluation')
    except (json.JSONDecodeError,KeyError):pass
  for p in [r/'gpu0_stage1_tmux.txt',r/'parallel_c2_controller.txt']:
   if p.exists() and 'Traceback (most recent call last)' in p.read_text():state['alerts'].append(str(p)+': traceback')
  try:state['gpu0']=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True,timeout=10).strip()
  except Exception as exc:state['gpu_query_error']=str(exc)
  state['stage_complete']=all(state['arms'][a]['actual_steps']==2000 for a in state['arms']) and len(state['evaluations'])==12
  log.write(json.dumps(state,allow_nan=False)+'\n');log.flush()
  temp=out/'latest.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(out/'latest.json')
  if state['stage_complete'] or now-started>21600:
   (out/'finished.json').write_text(json.dumps(state,indent=2));break
  time.sleep(30)
