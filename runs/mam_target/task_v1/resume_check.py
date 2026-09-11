import subprocess,os,json
from pathlib import Path
import torch
root=Path('runs/mam_target/task_v1');env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1')
commands=[['--arm','R2','--stop','6','--label','resume_full'],['--arm','R2','--stop','3','--label','resume_split'],['--arm','R2','--stop','6','--label','resume_split','--resume',str(root/'resume_split/attempt_000/checkpoints/step_000003.pt')]]
for i,c in enumerate(commands):
 with open(root/f'resume_{i}.txt','w') as f:subprocess.run(['.venv/bin/python','-m','mam_target.train']+c,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
a=torch.load(root/'resume_full/attempt_000/checkpoints/step_000006.pt',weights_only=True);b=torch.load(root/'resume_split/attempt_001/checkpoints/step_000006.pt',weights_only=True)
error=max(float((a['model'][k]-b['model'][k]).abs().max()) for k in a['model'])
assert error==0 and a['rows']==b['rows']
(root/'resume_regression.json').write_text(json.dumps(dict(max_parameter_error=error,training_rows_equal=True,actual_steps=6,split='3+3',device='GPU0'),indent=2))
