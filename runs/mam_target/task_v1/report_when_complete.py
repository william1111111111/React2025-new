import time,json,csv,hashlib
from pathlib import Path
root=Path('runs/mam_target/task_v1')
while not (root/'matrix_finished.json').exists():time.sleep(30)
end=json.loads((root/'matrix_finished.json').read_text());rows=[]
for arm in ('R1','R2'):
 for step in (2000,6000):
  label=f'seed123_{arm}_step{step}';task=root/'task_multitarget'/f'{label}.json'
  if not task.exists():continue
  result=json.loads(task.read_text())['result'];m=result['multi_target'];status=sorted((root/'frd20'/label).glob('status_*.json'))
  frd=json.loads(status[-1].read_text()) if status else {}
  rows.append(dict(arm=arm,step=step,seed=123,policy='native continuous AU',FRC=m['FRC'],FRD20=frd.get('FRD'),FRD_pairs=frd.get('completed_pairs',0),S_MSE=m['smse'],FRVar=m['FRVar']))
if rows:
 with open(root/'task_comparison.csv','w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lines=['# MAM-target first batch actual status','',f'Queue completed: {end["completed"]}; failures: {end["failures"]}. Single seed123 Development80; FRD20 is subset only.','', '| arm | step | FRC ↑ | exact FRD20 ↓ | S-MSE | FRVar |','|---|---:|---:|---:|---:|---:|']
for r in rows:lines.append(f'| {r["arm"]} | {r["step"]} | {r["FRC"]:.6f} | {r["FRD20"]} | {r["S_MSE"]:.6f} | {r["FRVar"]:.6f} |')
lines+=['| MAM native archive | archived | 0.810962 | 172.576423 | 0.157204 | 0.058912 |','', 'Do not mix best FRC and best FRD from different checkpoints. MAM pretraining/rounding/budget differ. Diversity requires grouped trajectory checks stored with exports, not maximizing variance. No flow/codec or additional seeds automatically started. Raw commands, checkpoints, identities and task matrices retained.']
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
