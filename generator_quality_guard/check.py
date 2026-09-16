"""Real TRAIN equality, active-path gradient, pure-noise graph and rollback checks."""
import sys,copy,time
import numpy as np
import torch
from .common import *
from .quality import scores,rollout_student_with_grad,ccc,active_dtw,hinges
from generator_reward_nft.model import Models
from generator_reward_nft.objective import nft
from reaction_flow.train import configure_flow
from reaction_flow.data import draw
from reaction_flow.flow_path import flow_path
from hirp.train_phase25 import rng_state,restore_rng
from hirp.phase24 import LEGACY

def norm(loss,params):
 gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True);return sum(float(g.square().sum()) for g in gs if g is not None)**.5

def main():
 configure_flow();torch.set_num_threads(4);torch.manual_seed(123);sys.path.insert(0,str(LEGACY))
 from framework.metrics.FRC import concordance_correlation_coefficient as official_ccc
 from tslearn.metrics import dtw
 m=Models();refhash=digest(m.env.generator.model);oldhash=digest(m.old);records=read(OUT/'SCHEDULE.json')['quality'];params=list(m.student.parameters());opt=torch.optim.AdamW(params,lr=2e-5,weight_decay=.01);rows=[];scales=read(OLD/'scales.json')
 for rec in records[:2]:
  e=m.env.episode(rec);noise=m.env.noise(e,4,'action');clean,parent=m.generate(e,noise,m.ref);pred=rollout_student_with_grad(m,e,noise)
  assert torch.equal(torch.stack([p.detach().cpu() for p in pred]),parent)
  c,d,C,D=scores(pred,e['targets'])
  with torch.no_grad():c0,d0,_,_=scores(list(parent),e['targets'])
  hc,hd=hinges(c,d,c.new_tensor(float(c0)),d.new_tensor(float(d0)),scales);assert float(hc)==0 and float(hd)<1e-20
  # Verify true native-frame candidate/target distances and CCC on every pair.
  errors=[];ccc_errors=[]
  for i,p in enumerate(parent):
   for j,y in enumerate(e['targets']):
    n=min(len(p),len(y));distance=sum(w*float(dtw(p[:n,a:b].numpy(),y[:n,a:b].numpy())) for a,b,w in [(0,15,1/15),(15,17,1),(17,25,1/8)])
    errors.append(abs(float(D[i,j])-distance));ccc_errors.append(abs(float(C[i,j])-official_ccc(y[:n].numpy(),p[:n].numpy())[0]))
  assert max(errors)<1e-8 and max(ccc_errors)<1e-6
  gc=norm(c,params);gd=norm(d,params);assert gc>0 and gd>0
  rows.append(dict(source=e['clip_id'],frames=e['n'],pure_noise_output_exact=True,DTW_max_error=max(errors),CCC_max_error=max(ccc_errors),CCC_score_gradient=gc,distance_score_gradient=gd,parent_h_C=float(hc),parent_h_D=float(hd)))
  del pred,c,d,C,D,hc,hd
 # A detached real sampled endpoint still supplies NFT parameter gradients.
 e=m.env.episode(records[0]);noise=m.env.noise(e,2,'action');clean,_=m.generate(e,noise,m.old);u0=clean[:1].cuda().detach();eps=draw(u0.shape,123,0,'guard_check_epsilon').cuda().double();tau=u0.new_tensor([.5]);u,target=flow_path(u0,eps,tau,e['mask']);offset=torch.tensor([e['start']],device='cuda')
 with torch.no_grad():old=m.old(u,tau,e['h'],e['mask'],e['mask'],offset)
 new=m.student(u,tau,e['h'],e['mask'],e['mask'],offset,gradient_checkpointing=True);loss=nft(new,old,u,u0,tau,u0.new_tensor([.8]),e['mask']);opt.zero_grad();loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(params,1.));assert grad>0;opt.step()
 # Save a nonempty optimizer and RNG; reject a trial update, restore, reproduce the next update.
 state=copy.deepcopy(m.student.state_dict());ostate=copy.deepcopy(opt.state_dict());rng=rng_state();savedhash=digest(m.student)
 def one_step():
  opt.zero_grad();v=m.student(u.detach(),tau,e['h'],e['mask'],e['mask'],offset,gradient_checkpointing=True);objective=v.square().mean();objective.backward();torch.nn.utils.clip_grad_norm_(params,1.);opt.step();return digest(m.student)
 h1=one_step();m.student.load_state_dict(state);opt.load_state_dict(ostate);restore_rng(rng);assert digest(m.student)==savedhash;h2=one_step();assert h1==h2
 m.student.load_state_dict(state);opt.load_state_dict(ostate);restore_rng(rng)
 # Diagnostic-only known changed model demonstrates actual fixed-parent hinge activation.
 bad=torch.load(NFT/'training/N1-quality-coverage/checkpoints/step_004000.pt',map_location='cpu',weights_only=True);m.student.load_state_dict(bad['student']);e=m.env.episode(records[0]);noise=m.env.noise(e,4,'action');_,parent=m.generate(e,noise,m.ref)
 with torch.no_grad():c0,d0,_,_=scores(list(parent),e['targets'])
 pred=rollout_student_with_grad(m,e,noise);c,d,_,_=scores(pred,e['targets']);hc,hd=hinges(c,d,c.new_tensor(float(c0)),d.new_tensor(float(d0)),scales);gc=norm(hc,params);gd=norm(hd,params);assert gc>0 or gd>0
 assert digest(m.env.generator.model)==refhash and digest(m.old)==oldhash and all(p.grad is None for p in m.ref.parameters())
 write(OUT/'ACCEPTANCE.json',dict(passed=True,real_cases=rows,NFT_gradient=grad,rollback_optimizer_RNG_next_update_exact=True,ref_and_old_unchanged=True,diagnostic_bad_checkpoint_only=str(NFT/'training/N1-quality-coverage/checkpoints/step_004000.pt'),bad_model=dict(C=float(c),D=float(d),C0=float(c0),D0=float(d0),h_C=float(hc),h_D=float(hd),grad_C=gc,grad_D=gd),formal_initialization='fresh original P2, never diagnostic N1',numerical_tolerances=dict(CCC_forward=1e-6,DTW_forward=1e-8,monitor_acceptance=0.),quality_rollouts_for_checks=24,student_quality_rollouts=12,reference_quality_rollouts=12,old_endpoint_rollouts=2,paired_noise_identity=True))
 print('REAL ACCEPTANCE passed',rows,flush=True)
if __name__=='__main__':main()
