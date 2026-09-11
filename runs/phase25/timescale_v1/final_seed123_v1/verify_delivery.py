import json,subprocess,sys
from pathlib import Path
sys.path.insert(0,'/home/zhengshiyi/react2025_new')
from hirp.phase15_audit import sha256_file,canonical_hash
r=Path('runs/phase25/timescale_v1');final=r/'final_seed123_v1'
old=json.load(open('runs/phase24/evaluation_v1/artifact_fingerprints.json'))['files'];bad=[x['path'] for x in old if sha256_file(x['path'])!=x['sha256']];assert not bad
checks=[]
for directory in (r/'task_multitarget',final/'task_multitarget',final/'distribution'):
 for p in directory.glob('*.json'):
  v=json.load(open(p))
  if 'eval_identity' not in v:continue
  assert canonical_hash(v['eval_identity'])==v['eval_identity_sha256'];assert canonical_hash(v['result'])==v['result_sha256']
  checks.append(str(p))
assert len(checks)==66
src=[dict(path=str(p),sha256=sha256_file(p)) for p in sorted(Path('hirp').rglob('*.py'))]
patch=subprocess.run(['git','diff','--','.gitignore'],capture_output=True,text=True).stdout
names=['hirp/data_phase25.py','hirp/diagnose_phase25.py','hirp/frd_phase25.py','hirp/run_phase25.py','hirp/task_phase25.py','hirp/train_phase25.py','hirp/write_report_phase25.py','hirp/task_phase25_final.py','hirp/evaluate_phase25_final.py','hirp/tests/test_phase25.py','hirp/tests/test_phase25_handoff.py','hirp/tests/test_phase25_final.py']
for name in names:patch+=subprocess.run(['git','diff','--no-index','/dev/null',name],capture_output=True,text=True).stdout
with (final/'code_diff.patch').open('x') as f:f.write(patch)
with (final/'source_snapshot.json').open('x') as f:json.dump(src,f,indent=2)
with (final/'verify_delivery.json').open('x') as f:json.dump(dict(historical_phase24_checked=len(old),historical_mismatches=bad,result_identity_and_content_checks=len(checks),baseline_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),new_code_files=names,new_code_added_lines=sum(len(Path(n).read_text().splitlines()) for n in names),no_new_training_in_finish=True,no_new_banks=True),f,indent=2)
files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha256_file(p),local_binary=p.suffix in ('.npy','.pt','.pth','.npz')) for p in sorted(final.rglob('*')) if p.is_file() and p.name not in ('artifact_fingerprints.json','verify_stdout.txt')]
with (final/'artifact_fingerprints.json').open('x') as f:json.dump(dict(files=files,scope='final evaluation/report artifacts, checkpoint fingerprints in analysis/checkpoint_fingerprints.json; local binary arrays not for git'),f,indent=2)
print('VERIFIED',len(old),'historical files',len(checks),'result identities',len(files),'delivery artifacts')
