import copy,random,tempfile
from pathlib import Path
import numpy as np
import torch
from generator_quality_guard.train import Run
from generator_quality_guard.common import OUT,write,digest
from reaction_flow.train import configure_flow
from hirp.train_phase25 import restore_rng
configure_flow();torch.set_num_threads(2);torch.manual_seed(123);np.random.seed(123);random.seed(123)
r=Run('preflight-rollback-only')
p=next(r.m.student.parameters());p.grad=torch.ones_like(p);r.opt.step();r.opt.zero_grad(set_to_none=True)
r.save(r.dest/'rollback_check.pt');s=torch.load(r.dest/'rollback_check.pt',weights_only=True)
def draws():return (random.random(),np.random.rand(),torch.rand(4),torch.rand(4,device='cuda'))
def step():
 z=draws();r.opt.zero_grad(set_to_none=True);p.grad=torch.full_like(p,z[0]+z[1]+float(z[2].sum())+float(z[3].sum()));r.opt.step();return z,digest(r.m.student)
a,h=step()
with torch.no_grad():next(r.m.old.parameters()).add_(.1)
r.step=7;r.rejected=100;r.consecutive_rejections=1
r.rollback(r.dest/'rollback_check.pt');b,h2=step()
assert h==h2 and all(torch.equal(x,y) if torch.is_tensor(x) else x==y for x,y in zip(a,b))
assert digest(r.m.old)==r.m.ref_hash
assert r.step==7 and r.rejected==100 and r.consecutive_rejections==1
write(OUT/'ROLLBACK_INTEGRATION.json',dict(passed=True,actual_Run_rollback=True,old_mutated_then_restored=True,optimizer_nonempty=True,python_numpy_CPU_CUDA_RNG_restored=True,next_random_update_exact=True,proposal_and_rejection_evidence_retained=True))
(r.dest/'rollback_check.pt').unlink()
print('Actual Run.rollback passed, old + optimizer + all RNG + next randomized update')
