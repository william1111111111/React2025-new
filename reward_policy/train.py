"""One on-policy score-function update per fresh group; independent pathwise KL."""
import argparse,json,time,random,shutil
import numpy as np
import torch
from reaction_flow.train import configure_flow
from reaction_flow.data import draw
from hirp.train_phase25 import rng_state,restore_rng
from .common import OUT,ARMS,read,write,sha
from .policy import Policy,replace_low_frequency,normal_log_prob
from .environment import TrainEnvironment
from .pilot import reference,valid
from .rewards import utility,difference_advantages,coverage

class Trainer:
    def __init__(self,arm):
        self.arm=arm;self.env=TrainEnvironment();self.policy=Policy().cuda().double();self.opt=torch.optim.AdamW(self.policy.parameters(),lr=1e-5,weight_decay=0.)
        self.scales=read(OUT/'scales.json');self.scales['phi']=np.array(self.scales['phi']);self.dual=np.ones(3);self.ema=np.zeros(3);self.step=0;self.rows=[]
    def update(self,records):
        logs=[];advantages=[];contexts=[];masks=[];constraints=[];metrics=[];begin=time.time()
        for rec in records:
            ref=reference(self.env,rec,'reference_table');e=self.env.episode(rec);noise=self.env.noise(e,10,'action')
            adjusted,actions,mask=replace_low_frequency(noise,e['basis'],self.policy,e['context']);pred=self.env.generate(e,adjusted);s=self.env.scores(e,pred)
            cf=float(np.quantile(ref['ccc'].max(1),.1));dc=float(np.quantile(ref['dtw'].min(1),.9));good=valid(s,self.scales);refgood=valid(ref,self.scales)
            u=utility(s['ccc'],s['dtw'],good,s['phi'],s['target_phi'],self.scales['phi'],cf,dc)
            c=s['ccc'].max(1);d=s['dtw'].min(1);adv=difference_advantages(self.arm,c,d,~good,s['phi'],u,self.scales,self.dual)
            ctx=e['context'].expand(10,-1).detach();logs.append(self.policy.log_prob(actions.detach(),ctx,mask));advantages.append(torch.tensor(adv,device='cuda'));contexts.append(ctx);masks.append(mask)
            constraints.append([(ref['ccc'].max(1).mean()-c.mean())/self.scales['c'],(d.mean()-ref['dtw'].min(1).mean())/self.scales['d'],float((~good).mean()-(~refgood).mean())])
            metrics.append(dict(record_step=rec['step'],mean_CCC=float(c.mean()),mean_DTW=float(d.mean()),coverage=coverage(u),bad_rate=float((~good).mean()),advantage_min=float(adv.min()),advantage_max=float(adv.max()),metric_seconds=s['metric_seconds'],reference_CCC=float(ref['ccc'].max(1).mean()),reference_DTW=float(ref['dtw'].min(1).mean())))
        actor=-(torch.stack(logs)*torch.stack(advantages).detach()).sum(1).mean()
        mask=torch.cat(masks);ctx=torch.cat(contexts);z0=draw(mask.shape,9400123,self.step,'KL_independent').cuda().double();z,lq=self.policy.rsample_and_log_prob(z0,ctx,mask)
        kl=(lq-normal_log_prob(z,mask)).mean();loss=actor+.01*kl
        if not torch.isfinite(loss) or float(kl)>20:raise RuntimeError(f'fixed policy safety stop: loss={float(loss)},total_KL={float(kl)}')
        self.opt.zero_grad(set_to_none=True);loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(self.policy.parameters(),1.));assert all(p.grad is None or torch.isfinite(p.grad).all() for p in self.policy.parameters())
        assert all(p.grad is None and not p.requires_grad for p in self.env.generator.model.parameters())
        self.opt.step();violation=np.mean(constraints,0);self.ema=.95*self.ema+.05*violation;self.dual=np.clip(self.dual+.01*self.ema,0,20);self.step+=1
        row=dict(step=self.step,actor=float(actor),KL_total=float(kl),KL_per_dimension=float(kl)/float(mask.sum(-1).double().mean()),gradient_norm=gn,dual=self.dual.tolist(),constraint_ema=self.ema.tolist(),episodes=metrics,seconds=time.time()-begin,generator_frozen=True,reward_actions_detached=True,candidate_rollouts=20,peak_gpu_bytes=torch.cuda.max_memory_allocated())
        self.rows.append(row);return row
    def save(self,path):
        path.parent.mkdir(exist_ok=True,parents=True);tmp=path.with_suffix('.tmp');torch.save(dict(policy=self.policy.state_dict(),optimizer=self.opt.state_dict(),dual=self.dual.tolist(),ema=self.ema.tolist(),step=self.step,rows=self.rows,rng=rng_state(),arm=self.arm,protocol_sha256=sha(OUT/'PROTOCOL.json'),scales_sha256=sha(OUT/'scales.json')),tmp);tmp.replace(path);write(path.with_suffix('.json'),dict(step=self.step,sha256=sha(path)))
    def load(self,path):
        s=torch.load(path,map_location='cpu',weights_only=True);assert s['arm']==self.arm and s['protocol_sha256']==sha(OUT/'PROTOCOL.json') and s['scales_sha256']==sha(OUT/'scales.json')
        self.policy.load_state_dict(s['policy']);self.opt.load_state_dict(s['optimizer']);self.dual=np.array(s['dual']);self.ema=np.array(s['ema']);self.step=s['step'];self.rows=s['rows'];restore_rng(s['rng'])

def main(arm,stop=500,diagnostic=None):
    configure_flow();torch.set_num_threads(4);torch.manual_seed(123);np.random.seed(123);random.seed(123)
    if diagnostic is None:
        assert read(OUT/'REACHABILITY.json')['gate_passed'];assert read(OUT/'TRAIN_ACCEPTANCE.json')['passed']
        for p,h in read(OUT/'CODE_IDENTITY.json').items():assert sha(p)==h
    assert 0<stop<=500
    dest=OUT/('diagnostics/'+diagnostic if diagnostic else 'training')/arm;dest.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=open(dest/'train.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    trainer=Trainer(arm);latest=dest/'latest.pt'
    if latest.exists():
        trainer.load(latest);lp=dest/'training.jsonl'
        if lp.exists() and len(lp.read_text().splitlines())>trainer.step:shutil.copy2(lp,dest/f'interrupted_{time.time_ns()}.jsonl')
        lp.write_text(''.join(json.dumps(r)+'\n' for r in trainer.rows))
    else:trainer.save(latest)
    episodes=read(OUT/'episodes.json')['training']
    try:
        while trainer.step<stop:
            j=trainer.step;row=trainer.update(episodes[j*2:j*2+2]);write(dest/'status.json',dict(stage='training',actual=trainer.step,requested=500,latest=row))
            with (dest/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(arm,trainer.step,row['seconds'],row['KL_total'],flush=True)
            if trainer.step%10==0 or trainer.step==stop:trainer.save(latest)
            if trainer.step in (100,500):trainer.save(dest/'checkpoints'/f'step_{trainer.step:06d}.pt')
    except BaseException as exc:
        trainer.save(latest);write(dest/'failure.json',dict(step=trainer.step,error=repr(exc),automatic_budget_extension=False));raise
    write(dest/'completion.json',dict(training_complete=trainer.step==500,evaluation_complete=False,actual_steps=trainer.step,candidate_rollouts=sum(r['candidate_rollouts'] for r in trainer.rows)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--stop',type=int,default=500);p.add_argument('--diagnostic');a=p.parse_args();main(a.arm,a.stop,a.diagnostic)
