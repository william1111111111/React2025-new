import subprocess,os,json
from pathlib import Path
import torch
r=Path('runs/mam_target/refine_v1');env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1')
commands=[['--arm','R2-cont','--stop','2','--label','smoke_A_full'],['--arm','R2-cont','--stop','1','--label','smoke_A_split'],['--arm','R2-cont','--stop','2','--label','smoke_A_split','--resume',str(r/'smoke_A_split/attempt_000/checkpoints/step_006001.pt')],['--arm','R3-cover','--stop','2','--label','smoke_B']]
for i,c in enumerate(commands):
 with open(r/f'gpu_smoke_{i}.txt','x') as f:subprocess.run(['.venv/bin/python','-m','mam_refine.train']+c,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
a=torch.load(r/'smoke_A_full/attempt_000/checkpoints/step_006002.pt',map_location='cpu',weights_only=True);b=torch.load(r/'smoke_A_split/attempt_001/checkpoints/step_006002.pt',map_location='cpu',weights_only=True)
error=max(float((a['model'][k]-b['model'][k]).abs().max()) for k in a['model']);assert error==0 and a['new_rows']==b['new_rows']
i=json.loads((r/'smoke_A_full/attempt_000/initial.json').read_text());j=json.loads((r/'smoke_B/attempt_000/initial.json').read_text());assert i['initial_model_hash']==j['initial_model_hash'] and i['initial_optimizer_hash']==j['initial_optimizer_hash']
(r/'gpu_regression.json').write_text(json.dumps(dict(resume_parameter_max_error=error,new_rows_equal=True,A_B_initial_model_hash=i['initial_model_hash'],A_B_initial_optimizer_hash=i['initial_optimizer_hash'],new_schedule_global_steps=[6001,6002],physical_gpu=1),indent=2))
