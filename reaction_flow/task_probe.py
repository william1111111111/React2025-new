"""One real TRAIN pure-noise Euler16 backward, no task calibration or optimization."""
import time,json
import torch
from .train import configure_flow,write,fm_batch
from .config import ROOT,FlowConfig
from .sampler import load_checkpoint
from .data import resources,draw
from .finetune import task_loss
from hirp.paired_data import paired_model_inputs

def main():
    configure_flow();cfg=FlowConfig();_,records,data=resources(cfg,'cuda:0');rec=records[12000]
    model,_=load_checkpoint(ROOT/'smoke_continuous/attempt_000/checkpoints/step_000002.pt','cuda:0')
    batch,targets,lengths,ids,y,ns,slots=data.batch(rec);h,sm=model.condition(**paired_model_inputs(batch))
    noise=draw((4,4,750,24),cfg.rollout_seed,rec['step'],'task_rollout').cuda().requires_grad_();torch.cuda.reset_peak_memory_stats();start=time.time()
    p=model.rollout(h,sm,noise,16,batch['crop_start'],gradient_checkpointing=True);loss=task_loss(p,batch,targets,lengths);loss.backward();torch.cuda.synchronize()
    norm=float(sum(x.grad.square().sum() for x in model.velocity.parameters() if x.grad is not None).sqrt())
    assert norm>0 and torch.isfinite(loss) and torch.isfinite(noise.grad).all() and float(noise.grad.abs().sum())>0
    assert all(x.grad is None for x in model.condition.parameters())
    write(ROOT/'task_rollout_probe.json',dict(TRAIN_step=rec['step'],K=4,R=4,T=750,Euler_NFE=16,task_surrogate=float(loss),velocity_gradient_norm=norm,noise_gradient_norm=float(noise.grad.norm()),condition_no_gradient=True,peak_memory_bytes=torch.cuda.max_memory_allocated(),seconds=time.time()-start,training_updates=0,note='untrained technical probe only; lambda calibration awaits A12000, no DEV score'))
    print('TASK_BACKWARD_OK',norm,time.time()-start,flush=True)
if __name__=='__main__':main()
