import json,torch
from pathlib import Path
from .config import FlowConfig,ROOT as OLD,DATA_MANIFEST
from .data import FlowData
from .train import write
from .shared_noise import prior_metadata
from hirp.train_phase25 import schedule,TrainConfig
from hirp.phase15_audit import sha256_file,canonical_hash
ROOT=Path('runs/reaction_flow/shared_noise_v1')
PARENT=Path('runs/reaction_flow/task_dynamics_v1/T0-task/attempt_000/checkpoints/step_014000.pt')
PARENT_SHA='e6199183cdc3526e1fc1390c89057e40514af8d2f2101d69a5e1dabb7f43f4d6'
def main():
    cfg=FlowConfig();assert sha256_file(PARENT)==PARENT_SHA;parent=torch.load(PARENT,map_location='cpu',weights_only=True)
    assert parent['phase']=='T0-task' and parent['global_step']==14000 and parent['completed']
    weight=parent['rows'][-1]['lambda_task'];assert parent['rows'][-1]['lambda_dyn']==0
    provenance=json.loads(DATA_MANIFEST.read_text())
    for row in provenance['data_files']+provenance['normalization']:assert sha256_file(row['path'])==row['sha256']
    data=FlowData(cfg);records,_,_=schedule(data.inner.pool,TrainConfig(max_steps=16000,T=750),32)
    generator=torch.Generator().manual_seed(cfg.target_seed)
    for r in records:r['target_slots']=torch.randint(4,(cfg.B,),generator=generator).tolist()
    old=json.loads((OLD/'schedule.json').read_text())['records'];assert records[:14000]==old
    new=records[14000:];assert len(new)==2000 and new[0]['step']==14001 and new[-1]['step']==16000
    write(ROOT/'schedule.json',dict(records=new,verified_old_prefix_sha256=canonical_hash(old),rule='same source/ref/crop/slot streams continued; not repeated prefix'))
    write(ROOT/'PROTOCOL.json',dict(parent=str(PARENT),parent_sha256=PARENT_SHA,arms={'G0-local':prior_metadata(0.),'G1-shared':prior_metadata(.05)},seed=123,budget=2000,global_steps=[14001,16000],checkpoints=[500,1000,2000],lambda_task=weight,warmup=False,optimizer='inherit T0 AdamW, LR2e-5 wd.01',condition='frozen',T=750,B=4,K_train=4,K_eval=10,solver='Euler16',precision='FP32 training, FP64 production then FP32 unchanged legacy metrics',schedule_sha256=sha256_file(ROOT/'schedule.json'),coordinate_stats_sha256=sha256_file(OLD/'coordinate_stats.json'),data_manifest_sha256=sha256_file(DATA_MANIFEST),target_law='unchanged TRAIN paired+3 full-source resized alternatives; uniform FM slot',stage='no dynamics, no speaker projection, no AU threshold change'))
    print('PREPARED',len(new),'fixed task weight',weight,flush=True)
if __name__=='__main__':main()
