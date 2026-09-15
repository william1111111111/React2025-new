"""Real TRAIN contract checks, independent of DEV scores."""
import json
import torch
from reaction_flow.train import configure_flow
from hirp.paired_data import paired_model_inputs
from .prepare import OUT,write
from .train import Experiment

def main():
    configure_flow();torch.set_num_threads(4);torch.manual_seed(123)
    e=Experiment('M1-mode');rec=json.loads((OUT/'schedule.json').read_text())['calibration'];x=e.data.batch(rec)
    h,m=e.base.condition(**paired_model_inputs(x['batch']));s=e.sample(x,rec,2)
    assert not s.requires_grad
    assert torch.equal(e.inject(h,(x['plan']-e.mean)/e.std,x['mask'],x['block_index']),h)
    again=e.sample(x,rec,2);prefix=e.sample(x,rec,1)
    repeat=float((again-s).abs().max());prefix_error=float((prefix-s[:,:1]).abs().max())
    assert repeat==0 and prefix_error<1e-5
    for i,index in enumerate(rec['source_indices']):
        slot=rec['target_slots'][i];y=e.data.aligned(index,x['ids'][i][slot],slot==0);start=rec['block_indices'][i]*750;n=int(x['lengths'][i]);assert torch.equal(y[start:start+n].cuda(),x['y'][i,:n])
        a,mask=e.data.plan(index,y);assert torch.equal(a,x['plan'][i,:len(a)])
    saved=torch.load(OUT/'smoke/M1-mode/latest.pt',map_location='cpu',weights_only=True);e.inject.load_state_dict(saved['inject']);e.base.load_state_dict(saved['base']);e.prior.load_state_dict(saved['prior'])
    a=e.inject(h,s[:,0],x['prior_mask'],x['block_index']);b=e.inject(h,s[:,1],x['prior_mask'],x['block_index']);sensitivity=float((a-b).abs().max());assert sensitivity>0
    e.opt.zero_grad(set_to_none=True);a.square().mean().backward();grad=float(e.inject.project[0].weight.grad.norm());assert grad>0
    assert all(p.grad is None for p in e.prior.parameters()) and all(p.grad is None for p in e.base.condition.parameters())
    m0=json.loads((OUT/'smoke/M0-control/training.jsonl').read_text().splitlines()[0]);m1=json.loads((OUT/'smoke/M1-mode/training.jsonl').read_text().splitlines()[0])
    for k in ('source_indices','block_indices','target_slots','target_ids','valid_frames'):assert m0[k]==m1[k]
    write(OUT/'acceptance.json',dict(real_TRAIN_batch_size=4,zero_init_condition_exact=True,plan_target_match=True,same_input_same_noise_error=repeat,K_prefix_error=prefix_error,trained_plan_sensitivity=sensitivity,projection_gradient=grad,prior_decoder_gradients_none=True,frozen_encoder_gradients_none=True,shared_exposure_verified=True,smoke_decoder_steps_per_arm=2,formal_training_started=False))
    print('real acceptance passed',sensitivity,grad,flush=True)
if __name__=='__main__':main()
