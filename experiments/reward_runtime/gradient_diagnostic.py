"""Read-only TRAIN gradients at fixed policy500; no optimizer step or protocol edit."""
import time
import numpy as np
import torch
from reaction_flow.train import configure_flow
from reaction_flow.data import draw
from reward_policy.train import Trainer
from reward_policy.common import OUT,read,write
from reward_policy.policy import replace_low_frequency,normal_log_prob
from reward_policy.pilot import reference,valid
from reward_policy.rewards import utility,difference_advantages

def main():
    configure_flow();torch.set_num_threads(4);torch.manual_seed(123);episodes=read(OUT/'episodes.json')['training'];result=[]
    for arm,batches in [('R-quality',[0]),('R-distance',[0]),('R-coverage',[0,1,2])]:
        tr=Trainer(arm);tr.load(OUT/'training'/arm/'checkpoints/step_000500.pt');params=list(tr.policy.parameters())
        for batch in batches:
            terms=[];bonus_terms=[];contexts=[];masks=[]
            for rec in episodes[2*batch:2*batch+2]:
                ref=reference(tr.env,rec,'reference_table');e=tr.env.episode(rec);noise=tr.env.noise(e,10,'action');adjusted,actions,mask=replace_low_frequency(noise,e['basis'],tr.policy,e['context']);s=tr.env.scores(e,tr.env.generate(e,adjusted));good=valid(s,tr.scales)
                u=utility(s['ccc'],s['dtw'],good,s['phi'],s['target_phi'],tr.scales['phi'],float(np.quantile(ref['ccc'].max(1),.1)),float(np.quantile(ref['dtw'].min(1),.9)))
                args=(arm,s['ccc'].max(1),s['dtw'].min(1),~good,s['phi'],u,tr.scales)
                adv=difference_advantages(*args,tr.dual);bonus=difference_advantages(*args,np.zeros(3));ctx=e['context'].expand(10,-1).detach();logp=tr.policy.log_prob(actions.detach(),ctx,mask)
                terms.append(-(torch.tensor(adv,device='cuda')*logp).sum());bonus_terms.append(-(torch.tensor(bonus,device='cuda')*logp).sum());contexts.append(ctx);masks.append(mask)
            actor=torch.stack(terms).mean();bonus=torch.stack(bonus_terms).mean();mask=torch.cat(masks);ctx=torch.cat(contexts);z0=draw(mask.shape,9400123,500+batch,'KL_independent').cuda().double();z,lq=tr.policy.rsample_and_log_prob(z0,ctx,mask);kl=(lq-normal_log_prob(z,mask)).mean()
            def grad(loss):
                gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
                return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g,p in zip(gs,params)])
            ga=grad(actor);gk=grad(.01*kl);gb=grad(bonus);total=ga+gk
            result.append(dict(arm=arm,checkpoint_step=500,batch_index=batch,actor_grad_norm=float(ga.norm()),weighted_KL_grad_norm=float(gk.norm()),KL_to_actor_norm_ratio=float(gk.norm()/ga.norm()),cosine=float(torch.nn.functional.cosine_similarity(ga,gk,dim=0)),bonus_grad_norm=float(gb.norm()),quality_grad_norm=float((ga-gb).norm()),gradient_clip_multiplier=min(1.,1./float(total.norm())),KL_sample_estimate=float(kl),optimizer_updates=0))
            print(result[-1],flush=True)
        del tr
    write(OUT/('KL_gradient_diagnostic_'+str(int(time.time()))+'.json'),dict(scope='fixed final checkpoint, TRAIN diagnostic only; no beta ablation and no updates',rows=result))
if __name__=='__main__':main()
