"""Bounded real T750 validation, including EMA node20 and split resume."""
import gc,json
import torch
from .balance_common import ROOT,PARENT,PARENT_SHA,initialize,resources
from .shared_model import load_checkpoint
from .shared_noise import recording_bases
from .train_balance import train
from .train import configure_flow,write
from hirp.paired_data import paired_model_inputs
from mam_refine.train import tree_hash

def main():
    configure_flow();cfg,records,data,_=resources();batch=data.batch(records[0])[0]
    inputs={k:v[:1] for k,v in paired_model_inputs(batch).items()}
    z,g=recording_bases(cfg,'balance-init-fixed-source',750,2);z=z[None].cuda();g=g[None].cuda()
    parent,saved=load_checkpoint(PARENT,'cuda:0');expected=parent.sample(**inputs,sample_count=2,noise=z,global_noise=g)
    parent_hash=tree_hash(parent.state_dict());opt_hash=tree_hash(saved['optimizer']);del parent,saved
    checks=[]
    for arm in ['B0-fixed','B1-ratio']:
        m,opt,params,control,saved=initialize(arm)
        assert tree_hash(m.state_dict())==parent_hash and tree_hash(opt.state_dict())==opt_hash
        aux=m.sample(**inputs,sample_count=2,noise=z,global_noise=g,return_aux=True)
        assert torch.equal(aux['predictions'],expected)
        replay=m.sample(**inputs,sample_count=2,noise=aux['local_noise'],global_noise=aux['global_noise'])
        assert torch.equal(replay,expected)
        assert all(p.grad is None for p in m.condition.parameters())
        checks.append(dict(arm=arm,model_hash=parent_hash,optimizer_hash=opt_hash,rho=m.rho,parent_output_error=0.,aux_replay_error=0.))
        del m,opt,params,saved,aux,replay
    del data,batch,inputs,expected,z,g;gc.collect();torch.cuda.empty_cache()
    write(ROOT/'initialization_checks.json',dict(parent_sha256=PARENT_SHA,arms=checks))
    train('B0-fixed',1,label='smoke_B0')
    train('B1-ratio',21,label='smoke_full',extra_checkpoints=(19,))
    cp=ROOT/'smoke_full/attempt_000/checkpoints/step_016019.pt'
    train('B1-ratio',20,resume=cp,label='smoke_split')
    cp2=ROOT/'smoke_split/attempt_000/checkpoints/step_016020.pt'
    train('B1-ratio',21,resume=cp2,label='smoke_resume')
    a=torch.load(ROOT/'smoke_full/attempt_000/checkpoints/step_016021.pt',map_location='cpu',weights_only=True)
    b=torch.load(ROOT/'smoke_resume/attempt_000/checkpoints/step_016021.pt',map_location='cpu',weights_only=True)
    assert tree_hash(a['model'])==tree_hash(b['model']) and tree_hash(a['optimizer'])==tree_hash(b['optimizer'])
    for k in ['ratio_controller','schedule_position','schedule_sha256','consumed_prefix_sha256']:
        assert a[k]==b[k],k
    fixed=torch.load(ROOT/'smoke_B0/attempt_000/checkpoints/step_016001.pt',map_location='cpu',weights_only=True)
    keys=['source_indices','target_slots','crop_start','target_ids','tau','record_sha256','FM_initial_sha256','task_initial_sha256']
    for k in keys:assert a['rows'][0][k]==fixed['rows'][0][k],k
    assert a['rows'][19]['EMA_A']!=a['rows'][18]['EMA_A']
    write(ROOT/'resume_regression.json',dict(passed=True,model_equal=True,optimizer_equal=True,controller_equal=True,schedule_equal=True,crosses_measurement_step=20,compared_steps=[19,20,21],shape='T750 B4 K4 Euler16 FP32',parent_output_error=0.,noise_and_data_keys_equal=True,short_validation_updates=24,calibration_batches=16,production_updates=0))
    print('SMOKE PASSED',flush=True)
if __name__=='__main__':main()
