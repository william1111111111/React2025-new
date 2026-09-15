"""Exactly 100 updates, identity restart, independent fixed TRAIN probes."""
import argparse,json,hashlib,time,random
import numpy as np
import torch
from reaction_flow.train import configure_flow
from reaction_flow.data import draw
from reward_policy.environment import TrainEnvironment
from reward_policy.policy import Policy,replace_low_frequency,normal_log_prob
from reward_policy.pilot import reference,valid
from reward_policy.common import read,write,sha
from hirp.train_phase25 import rng_state,restore_rng
from .audit import OUT,OLD,quant
from .rewards import affinity,quality_credits,coverage_gains
ARMS=('R0-baselined','R1-calibrated')

def vector(policy):return torch.cat([p.detach().reshape(-1) for p in policy.parameters()])
def digest(policy):return hashlib.sha256(vector(policy).cpu().numpy().tobytes()).hexdigest()

class Run:
    def __init__(self,arm):
        self.arm=arm;self.env=TrainEnvironment();self.policy=Policy().cuda().double();self.opt=torch.optim.AdamW(self.policy.parameters(),lr=1e-5,weight_decay=0.);self.dual=np.ones(3);self.ema=np.zeros(3);self.step=0;self.rows=[];self.old=read(OLD/'scales.json');self.metric=read(OUT/'GEOMETRY_STRATIFIED.json');self.scales=self.old if arm==ARMS[0] else self.metric['scales'];self.h2=1. if arm==ARMS[0] else self.metric['h2'];self.initial=vector(self.policy).clone();self.probe_base={}
    def episode(self,rec,reference_folder):
        ref=reference(self.env,rec,reference_folder);e=self.env.episode(rec);noise=self.env.noise(e,10,'action');adjusted,actions,mask=replace_low_frequency(noise,e['basis'],self.policy,e['context']);pred=self.env.generate(e,adjusted);s=self.env.scores(e,pred);n=e['n'];ns=[len(y) for y in e['targets']];pm=np.stack([np.arange(96)//24<min(4,n,nn) for nn in ns]);sim,d2=affinity(s['phi'],s['target_phi'],np.array(self.scales['phi']),pm,self.h2)
        cg=s['ccc']>=np.quantile(ref['ccc'].max(1),.1);dg=s['dtw']<=np.quantile(ref['dtw'].min(1),.9);good=valid(s,self.old);rg=valid(ref,self.old);eligible=cg&dg&good[:,None];u=sim*eligible;c=s['ccc'].max(1);d=s['dtw'].min(1);q=self.dual[0]*c/self.old['c']-self.dual[1]*d/self.old['d']-self.dual[2]*(~good);aq=quality_credits(q);ac=coverage_gains(u);ctx=e['context'].expand(10,-1).detach();lp=self.policy.log_prob(actions.detach(),ctx,mask)
        violation=np.array([(ref['ccc'].max(1).mean()-c.mean())/self.old['c'],(d.mean()-ref['dtw'].min(1).mean())/self.old['d'],float((~good).mean()-(~rg).mean())])
        row=dict(record_step=rec['step'],eligible_pair_rate=float(eligible.mean()),CCC_pass=float(cg.mean()),DTW_pass=float(dg.mean()),motion_pass=float(good.mean()),affinity_quantiles=quant(sim[eligible]),coverage=float(u.max(0).mean()),coverage_gain_quantiles=quant(ac),nontrivial_gain_fraction=float((ac>1e-6).mean()),quality_credit_quantiles=quant(aq),q_within_variance=float(q.var()),q_source_mean=float(q.mean()),mean_CCC=float(c.mean()),mean_DTW=float(d.mean()),reference_CCC=float(ref['ccc'].max(1).mean()),reference_DTW=float(ref['dtw'].min(1).mean()),metric_seconds=s['metric_seconds'])
        return dict(quality=-(lp*torch.tensor(aq,device='cuda')).sum(),bonus=-(lp*torch.tensor(ac,device='cuda')).sum(),ctx=ctx,mask=mask,violation=violation,row=row,env=e,noise=noise,adjusted=adjusted,actions=actions,pred=pred,scores=s)
    def losses(self,records,folder='reference_table'):
        xs=[self.episode(r,folder) for r in records];quality=torch.stack([x['quality'] for x in xs]).mean();bonus=torch.stack([x['bonus'] for x in xs]).mean();mask=torch.cat([x['mask'] for x in xs]);ctx=torch.cat([x['ctx'] for x in xs]);z0=draw(mask.shape,9400123,self.step,'KL_independent').cuda().double();z,lq=self.policy.rsample_and_log_prob(z0,ctx,mask);kl=(lq-normal_log_prob(z,mask)).mean();return xs,quality,bonus,kl
    def grad(self,loss):
        gs=torch.autograd.grad(loss,list(self.policy.parameters()),retain_graph=True,allow_unused=True)
        return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for p,g in zip(self.policy.parameters(),gs)])
    def gradients(self,q,b,k):
        gq,gb,gk=self.grad(q),self.grad(b),self.grad(.01*k)
        cos=lambda x,y:float(torch.nn.functional.cosine_similarity(x,y,dim=0))
        return dict(quality_norm=float(gq.norm()),bonus_norm=float(gb.norm()),weighted_KL_norm=float(gk.norm()),quality_bonus_cosine=cos(gq,gb),actor_KL_cosine=cos(gq+gb,gk))
    def update(self,records):
        tick=time.time();xs,q,b,k=self.losses(records);stats=self.gradients(q,b,k);loss=q+b+.01*k
        if not torch.isfinite(loss) or float(k)>20:raise RuntimeError('fixed finite/KL guard')
        self.opt.zero_grad(set_to_none=True);before=vector(self.policy).clone();loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(self.policy.parameters(),1.));self.opt.step();delta=vector(self.policy)-before
        assert all(p.grad is None and not p.requires_grad for p in self.env.generator.model.parameters())
        self.ema=.95*self.ema+.05*np.mean([x['violation'] for x in xs],0);self.dual=np.clip(self.dual+.01*self.ema,0,20);self.step+=1
        row=dict(step=self.step,quality_actor=float(q),bonus_actor=float(b),KL_estimate=float(k),gradients=stats,preclip_norm=norm,clip_multiplier=min(1,1/norm),optimizer_delta_L2=float(delta.norm()),optimizer_relative_delta=float(delta.norm()/before.norm()),dual=self.dual.tolist(),ema=self.ema.tolist(),episodes=[x['row'] for x in xs],seconds=time.time()-tick,action_rollouts=20,fresh_reference_rollouts=0,reference_draws_reused=20)
        self.rows.append(row);return row
    def probe(self,records,dest):
        xs,q,b,k=self.losses(records,'calibration');stats=self.gradients(q,b,k);rows=[]
        for x in xs:
            e=x['env'];key=e['rec']['step'];z=x['actions'];base_z=torch.einsum('nr,knd->krd',e['basis'],x['noise']).flatten(1);active=x['mask'][0];aa=z[:,active];zz=base_z[:,active];cov=lambda v:torch.cov(v.T).detach().cpu().tolist()
            current=vector(self.policy);delta=current-self.initial
            if key not in self.probe_base:self.probe_base[key]=x['pred'].clone()
            channels={}
            for name,lo,hi in [('AU',0,15),('VA',15,17),('expression',17,25)]:
                real=torch.cat([y[:,lo:hi].diff(dim=0).abs().flatten() for y in e['targets']]);acc=torch.cat([y[:,lo:hi].diff(dim=0).diff(dim=0).abs().flatten() for y in e['targets'] if len(y)>2]);channels[name]=dict(real_speed=quant(real.numpy()),real_acceleration=quant(acc.numpy()),prediction_mean_abs_change=float((x['pred'][...,lo:hi]-self.probe_base[key][...,lo:hi]).abs().mean()))
            logscales=[];shifts=[];v=base_z
            with torch.no_grad():
                for layer in self.policy.layers:
                    ls,shift=layer.parameters_at(v,x['ctx'],x['mask']);logscales.extend(ls[x['mask']].cpu().numpy());shifts.extend(shift[x['mask']].cpu().numpy());v,_=layer(v,x['ctx'],x['mask'])
            rows.append(dict(**x['row'],action_delta_RMS=float((aa-zz).square().mean().sqrt()),action_mean=float(aa.mean()),action_std=float(aa.std()),action_covariance=cov(aa),base_action_covariance=cov(zz),coupling_logscale=quant(logscales),coupling_shift=quant(shifts),scale_saturation=float(np.mean(np.abs(logscales)>.25*.95)),full_noise_delta_RMS=float((x['adjusted']-x['noise']).square().mean().sqrt()),rewarded_block_noise_delta_RMS=float((x['adjusted'][:,e['start']:e['start']+e['n']]-x['noise'][:,e['start']:e['start']+e['n']]).square().mean().sqrt()),prediction_delta_RMS=float((x['pred']-self.probe_base[key]).square().mean().sqrt()),native_groups=channels))
        write(dest/f'probe_{self.step:03d}.json',dict(step=self.step,no_update=True,policy_sha256=digest(self.policy),parameter_change_L2=float(delta.norm()),parameter_relative_change=float(delta.norm()/self.initial.norm()),gradients=stats,KL_estimate=float(k),episodes=rows,extra_action_rollouts=20,reference_rollouts_reused=20))
    def save(self,path):
        path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(dict(policy=self.policy.state_dict(),optimizer=self.opt.state_dict(),dual=self.dual.tolist(),ema=self.ema.tolist(),step=self.step,rows=self.rows,rng=rng_state(),arm=self.arm,initial=self.initial,probe_base=self.probe_base,protocol_sha256=sha(OUT/'PROTOCOL.json')),tmp);tmp.replace(path)

def main(arm):
    configure_flow();torch.set_num_threads(4);torch.manual_seed(123);np.random.seed(123);random.seed(123);protocol=read(OUT/'PROTOCOL.json')
    for p,h in protocol['inputs'].items():assert sha(p)==h
    for p,h in read(OUT/'CODE_IDENTITY.json').items():assert sha(p)==h
    run=Run(arm);dest=OUT/'training'/arm;dest.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=open(dest/'train.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    latest=dest/'latest.pt'
    if latest.exists():
        s=torch.load(latest,map_location='cpu',weights_only=True);assert s['protocol_sha256']==sha(OUT/'PROTOCOL.json');run.policy.load_state_dict(s['policy']);run.opt.load_state_dict(s['optimizer']);run.dual=np.array(s['dual']);run.ema=np.array(s['ema']);run.step=s['step'];run.rows=s['rows'];run.initial=s['initial'].cuda();run.probe_base=s['probe_base'];restore_rng(s['rng'])
    records=read(OLD/'episodes.json')['training'][:200];probes=[read(OLD/'episodes.json')['calibration'][i] for i in protocol['probe_indices']]
    if run.step==0:run.probe(probes,dest);run.save(dest/'checkpoints/step_000.pt')
    (dest/'training.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in run.rows))
    while run.step<100:
        row=run.update(records[2*run.step:2*run.step+2]);write(dest/'status.json',dict(stage='training',actual=run.step,requested=100,latest=row))
        with (dest/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(arm,run.step,row['seconds'],row['gradients'],flush=True)
        if run.step in (25,50,100):run.probe(probes,dest);run.save(dest/'checkpoints'/f'step_{run.step:03d}.pt')
        if run.step%10==0 or run.step in (25,50):run.save(latest)
    write(dest/'completion.json',dict(training_complete=True,actual_updates=100,formal_action_rollouts=2000,probe_action_rollouts=80,fresh_reference_rollouts=0,reference_generation_cost='existing old calibration/reference tables reused; historical cost retained in old manifests',official_metrics_run=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);a=p.parse_args();main(a.arm)
