from pathlib import Path
import json,torch,numpy as np
from mam_refine.audit import ROOT,write
from hirp.train_phase25 import TrainConfig,schedule,stream_hashes
from hirp.session_data import SessionPopulation
from hirp.config import HiRPConfig
from hirp.phase15_audit import sha256_file
old=Path('runs/phase25/timescale_v1/seed_123');parent=Path('runs/mam_target/task_v1/R2/attempt_001/checkpoints/step_006000.pt')
results=json.loads((ROOT/'RESULTS_LATEST.json').read_text());row=next(x for x in results['models'] if x['arm']=='R2' and x['step']==6000)
assert row['FRD20']<172.57642347297096 and row['FRC80']>.810962318916204
cfg=TrainConfig(max_steps=8000);records,noise,hashes=schedule(SessionPopulation('data','train',750),cfg,HiRPConfig().latent_dim)
original=json.loads((old/'schedule.json').read_text())['records'];assert records[:6000]==original
assert torch.equal(noise[:6000],torch.from_numpy(np.load(old/'noise.npy')))
saved=torch.load(parent,map_location='cpu',weights_only=True);assert saved['step']==6000 and saved['arm']=='R2' and saved['completed']
assert stream_hashes(records[:6000],noise[:6000])==saved['consumed_hashes']
records=records[6000:];noise=noise[6000:];write(ROOT/'schedule.json',dict(records=records,hashes=stream_hashes(records,noise)))
with (ROOT/'noise.npy').open('xb') as f:np.save(f,noise.numpy())
write(ROOT/'manifest.json',dict(parent_checkpoint=str(parent),parent_sha256=sha256_file(parent),parent_global_step=6000,arms={'R2-cont':'softmin','R3-cover':'one_to_one'},additional_steps=2000,global_end_step=8000,eval_additional_steps=[500,2000],FRD_additional_steps=[2000],seed=123,physical_gpu=1,T=750,B=4,K=4,lr=1e-4,weight_decay=.01,optimizer='restore exact parent AdamW state in both; no reset',weights=saved['weights'],data_manifest=str(old/'manifest.json'),data_manifest_sha256=sha256_file(old/'manifest.json'),split_hash=saved['split_hash'],scales=saved['scales'],schedule_hashes=stream_hashes(records,noise),noise_sha256=sha256_file(ROOT/'noise.npy'),schedule_rule='regenerate seeded schedule through8000; verify first6000 exactly, consume only6001..8000. alternative seed uses global step; same crop/noise/targets across arms',inherited_valid_frames=sum(sum(r['source_lengths']) for r in saved['rows']),inherited_parameter_count=sum(v.numel() for v in saved['model'].values()),parent_selection='user fixed R2 step6000 after completed exact FRD20 gate; no per-input selection',baseline_results_sha256=sha256_file(ROOT/'RESULTS_LATEST.json')))
print('PREPARED',len(records),'new steps; exact prefix verified; parent',sha256_file(parent))
