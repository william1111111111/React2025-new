"""Freeze future 16001..18000 TRAIN schedule, then calibrate on its first16."""
import json,time,statistics,shutil
import torch
from .balance_common import ROOT,PARENT,PARENT_SHA,AROOT,DATA_MANIFEST,resources,gradients
from .balance_controller import LAMBDA_OLD
from .config import FlowConfig
from .data import FlowData
from .shared_model import load_checkpoint
from .train_shared import losses
from .task_dynamics_train import get_batch
from .train import write,configure_flow
from hirp.train_phase25 import schedule,TrainConfig,restore_rng
from hirp.phase15_audit import sha256_file,canonical_hash
from mam_refine.train import tree_hash

def main():
    configure_flow();ROOT.mkdir(exist_ok=True);assert not (ROOT/'calibration.json').exists()
    free=shutil.disk_usage(ROOT).free;assert free>100*1024**3
    assert sha256_file(PARENT)==PARENT_SHA
    cfg=FlowConfig();data=FlowData(cfg)
    records,_,_=schedule(data.inner.pool,TrainConfig(max_steps=18000,T=750),32)
    generator=torch.Generator().manual_seed(cfg.target_seed)
    for r in records:r['target_slots']=torch.randint(4,(cfg.B,),generator=generator).tolist()
    old=json.loads((AROOT/'schedule.json').read_text())['records']
    prior=json.loads(open('runs/reaction_flow/shared_noise_v1/schedule.json').read())['records']
    assert records[:14000]==old and records[14000:16000]==prior
    new=records[16000:];write(ROOT/'schedule.json',dict(records=new,verified_prefix_sha256=canonical_hash(records[:16000]),rule='continued original TRAIN RNG streams to16001..18000'))
    inputs={str(p):sha256_file(p) for p in [ROOT/'schedule.json',DATA_MANIFEST,AROOT/'coordinate_stats.json',AROOT/'manifest.json']}
    write(ROOT/'PROTOCOL.json',dict(version='prior-task-balance-v1',parent=str(PARENT),parent_sha256=PARENT_SHA,arms=['B0-fixed','B1-ratio'],rho=.05,seed=123,lambda_old=LAMBDA_OLD,r_target=1.,ema_decay=.9,measurement_interval=20,warmup=200,lambda_bounds=[.25*LAMBDA_OLD,4*LAMBDA_OLD],lambda_timing='step t uses previous goal with ramp min(t/200,1); measurement at t%20==0 updates goal for t+1',calibration='medians of component norms on frozen parent, first16 TRAIN records; extra compute, reused as formal schedule beginning',checkpoints=[500,1000,2000],global_steps=[16001,18000],budget=2000,inputs=inputs,physical_gpu=1,max_gpus=1,precision='FP32 training; FP64 Euler16 production, native continuous AU',free_disk_bytes=free))
    del data
    cfg,records,data,protocol=resources();model,saved=load_checkpoint(PARENT,'cuda:0');model.train();restore_rng(saved['rng'])
    params=[p for p in model.velocity.parameters() if p.requires_grad];before=tree_hash(model.state_dict());rows=[];started=time.time()
    for rec in records[:16]:
        batch,y,n,ids,slots=get_batch(data,rec)
        fm,task,_,pred,tau=losses(model,batch,y,n,rec,cfg)
        a,b,cos=gradients(fm,task,params)
        assert all(p.grad is None for p in model.parameters())
        rows.append(dict(global_step=rec['step'],FM=float(fm),task=float(task),a=a,b=b,ratio=a/b,cosine=cos,FM_noise=batch['_FM_initial_hash'],task_noise=batch['_task_initial_hash'],record_sha256=canonical_hash(rec)))
        print('CAL',len(rows),rows[-1],flush=True);del fm,task,pred
    assert tree_hash(model.state_dict())==before and sha256_file(PARENT)==PARENT_SHA
    write(ROOT/'calibration.json',dict(parent_sha256=PARENT_SHA,A=statistics.median(r['a'] for r in rows),B=statistics.median(r['b'] for r in rows),rows=rows,steps=0,model_unchanged=True,seconds=time.time()-started))
if __name__=='__main__':main()
