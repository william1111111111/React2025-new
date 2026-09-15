"""Two independent optimizers: observed-plan FM prior and mode-aware T0 decoder."""
import argparse,json,random,time
import numpy as np
import torch
from reaction_flow.sampler import load_checkpoint,metadata
from reaction_flow.train import configure_flow
from reaction_flow.data import draw
from reaction_flow.flow_path import flow_path,flow_loss
from .task_mask import masked_task_parts
from hirp.paired_data import paired_model_inputs
from hirp.train_phase25 import rng_state,restore_rng
from .models import PlanPrior,PlanInjection
from .plans import summarize,mode_loss
from .data import ModeData
from .prepare import OUT,PARENT,PARENT_SHA,sha,write

class Experiment:
    def __init__(self,arm):
        self.arm=arm;self.base,self.parent=load_checkpoint(PARENT,'cuda:0');self.base.train()
        self.prior=PlanPrior().cuda();self.inject=PlanInjection().cuda()
        self.opt=torch.optim.AdamW(self.base.velocity.parameters(),lr=2e-5,weight_decay=.01)
        self.opt.load_state_dict(self.parent['optimizer'])
        for g in self.opt.param_groups:g['lr']=2e-5
        if arm=='M1-mode':self.opt.add_param_group(dict(params=list(self.inject.parameters()),lr=1e-4,weight_decay=.01))
        self.prior_opt=torch.optim.AdamW(self.prior.parameters(),lr=1e-4,weight_decay=.01)
        stats=json.loads((OUT/'plan_stats.json').read_text());self.mean=torch.tensor(stats['mean'],device='cuda');self.std=torch.tensor(stats['std'],device='cuda')
        self.data=ModeData(self.base);self.lambda_mode=None

    def prior_loss(self,x,rec):
        target=(x['plan']-self.mean)/self.std;mask=x['mask'];z=draw(target.shape,8200123,rec['step'],'prior_FM').cuda();tau=draw((len(z),),8300123,rec['step'],'prior_tau',False).cuda()
        state=(tau[:,None,None]*target+(1-tau[:,None,None])*z)*mask
        v=self.prior(state,tau,x['context'],x['positions'],mask)
        return ((v-(target-z)).square()*mask).sum()/mask.sum().clamp_min(1)

    def sample(self,x,rec,k):
        b,blocks,_=x['plan'].shape
        # Candidate-specific keyed draws preserve K-prefix as K changes.
        z=torch.stack([draw((b,blocks,96),8400123,rec['step'],f'plan_{j}').cuda() for j in range(k)],1)
        rep=lambda v:v[:,None].expand(-1,k,*v.shape[1:]).reshape(b*k,*v.shape[1:])
        s=self.prior.sample(z.reshape(b*k,blocks,96),rep(x['context']),rep(x['positions']),rep(x['prior_mask']))
        return s.reshape(b,k,blocks,96).detach()

    def decoder_losses(self,x,rec):
        b=x['batch'];cfg=self.base.config;h,sm=self.base.condition(**paired_model_inputs(b));valid=torch.arange(750,device='cuda')[None]<x['lengths'][:,None]
        target=self.base.transform(x['y']).detach();z=draw(target.shape,cfg.noise_seed,rec['step'],'FM_noise').cuda();tau=draw((len(z),),cfg.tau_seed,rec['step'],'FM_tau',False).cuda();state,velocity=flow_path(target,z,tau,valid)
        hp=self.inject(h,(x['plan']-self.mean)/self.std,x['mask'],x['block_index']) if self.arm=='M1-mode' else h
        fm=flow_loss(self.base.velocity(state,tau,hp,valid,sm,b['crop_start'],gradient_checkpointing=True),velocity,valid)
        local=draw((len(z),4,750,24),cfg.rollout_seed,rec['step'],'task_rollout').cuda()
        if self.arm=='M1-mode':
            plans=self.sample(x,rec,4);B,K,L,D=plans.shape
            rep=lambda v:v[:,None].expand(-1,K,*v.shape[1:]).reshape(B*K,*v.shape[1:])
            hh=self.inject(rep(h),plans.reshape(B*K,L,D),rep(x['prior_mask']),rep(x['block_index']))
            pred=self.base.rollout(hh,rep(sm),local.reshape(B*K,1,750,24),16,rep(b['crop_start']),gradient_checkpointing=True).reshape(B,K,750,25)
            actual=[];desired=[];masks=[]
            for i,pts in enumerate(x['pts']):
                a,m=summarize(self.base.transform(pred[i,:,:len(pts)]),pts)
                actual.append(a.flatten(-2));desired.append(plans[i,:,x['block_index'][i]]*self.std+self.mean);masks.append(m.flatten().expand(K,-1))
            mode=mode_loss(torch.stack(actual),torch.stack(desired).detach(),torch.stack(masks),self.std)
        else:
            pred=self.base.rollout(h,sm,local,16,b['crop_start'],gradient_checkpointing=True);mode=fm.new_zeros(())
        task,parts=masked_task_parts(pred,b)
        return fm,task,mode,parts

    def parameters(self):return [p for g in self.opt.param_groups for p in g['params']]

    def calibrate(self,rec,destination):
        x=self.data.batch(rec);fm,task,mode,_=self.decoder_losses(x,rec);params=self.parameters()
        def norm(loss):
            gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
            return float(sum(g.square().sum() for g in gs if g is not None).sqrt())
        nt,nm=norm(task),norm(mode)
        if nm<=1e-12 or nt<=1e-12:raise RuntimeError('zero calibration gradient; no silent default lambda')
        self.lambda_mode=.1*self.parent['calibration']['lambda_ref']*nt/nm
        write(destination,dict(lambda_mode=self.lambda_mode,task_gradient=nt,mode_gradient=nm,lambda_task=self.parent['calibration']['lambda_ref'],source='independent TRAIN batch, once before decoder training',record=rec))

    def save(self,path,step,warmup,rows):
        obj=dict(format_version='observable-mode-v1',arm=self.arm,base_metadata=metadata(self.base),base=self.base.state_dict(),prior=self.prior.state_dict(),inject=self.inject.state_dict(),optimizer=self.opt.state_dict(),prior_optimizer=self.prior_opt.state_dict(),step=step,warmup=warmup,rows=rows,rng=rng_state(),lambda_mode=self.lambda_mode,parent_sha256=PARENT_SHA,protocol_sha256=sha(OUT/'PROTOCOL.json'),stats_sha256=sha(OUT/'plan_stats.json'))
        path.parent.mkdir(exist_ok=True,parents=True);tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path);write(path.with_suffix('.json'),dict(sha256=sha(path),step=step,warmup=warmup))

def run(arm,stop=6000,diagnostic=False,warmup_stop=2000,diagnostic_label="smoke"):
    configure_flow();torch.set_num_threads(4);random.seed(123);np.random.seed(123);torch.manual_seed(123)
    assert sha(PARENT)==PARENT_SHA and 0<stop<=6000 and 0<=warmup_stop<=2000
    assert diagnostic or warmup_stop==2000
    if not diagnostic:
        for p,h in json.loads((OUT/'CODE_IDENTITY.json').read_text()).items():assert sha(p)==h
        for p,h in json.loads((OUT/'CACHE_IDENTITY.json').read_text()).items():assert sha(p)==h
    e=Experiment(arm);schedule=json.loads((OUT/'schedule.json').read_text());dest=OUT/(diagnostic_label if diagnostic else 'training')/arm;dest.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=open(dest/'train.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    step=warmup=0;rows=[];path=dest/'latest.pt'
    if path.exists():
        s=torch.load(path,map_location='cpu',weights_only=True)
        assert s['arm']==arm and s['protocol_sha256']==sha(OUT/'PROTOCOL.json') and s['stats_sha256']==sha(OUT/'plan_stats.json')
        e.base.load_state_dict(s['base']);e.prior.load_state_dict(s['prior']);e.inject.load_state_dict(s['inject']);e.opt.load_state_dict(s['optimizer']);e.prior_opt.load_state_dict(s['prior_optimizer']);restore_rng(s['rng']);step=s['step'];warmup=s['warmup'];rows=s['rows'];e.lambda_mode=s['lambda_mode']
    tick=time.time()
    if path.exists():
        lp=dest/'training.jsonl'
        if lp.exists() and len(lp.read_text().splitlines())>step:lp.rename(dest/f'interrupted_log_{time.time_ns()}.jsonl')
        lp.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    if arm=='M1-mode':
        for j in range(warmup,warmup_stop):
            prior_start=time.time();rec=schedule['prior_warmup'][j];x=e.data.batch(rec);e.prior_opt.zero_grad(set_to_none=True);loss=e.prior_loss(x,rec);loss.backward();torch.nn.utils.clip_grad_norm_(e.prior.parameters(),1.);e.prior_opt.step();warmup=j+1
            with (dest/'prior_warmup.jsonl').open('a') as f:f.write(json.dumps(dict(step=warmup,loss=float(loss),seconds=time.time()-prior_start,peak_gpu_bytes=torch.cuda.max_memory_allocated()))+'\n')
            if warmup%25==0 or warmup==1:
                write(dest/'status.json',dict(stage='prior_warmup',actual=warmup,requested=warmup_stop,loss=float(loss),seconds=time.time()-tick));print(arm,'prior',warmup,float(loss),flush=True)
            if warmup%250==0 or warmup==warmup_stop:e.save(path,step,warmup,rows)
        if e.lambda_mode is None:
            if not diagnostic and (OUT/'calibration.json').exists():e.lambda_mode=json.loads((OUT/'calibration.json').read_text())['lambda_mode']
            else:e.calibrate(schedule['calibration'],dest/'calibration.json' if diagnostic else OUT/'calibration.json')
    for j in range(step,stop):
        start=time.time();rec=schedule['decoder'][j];x=e.data.batch(rec);e.prior_opt.zero_grad(set_to_none=True)
        prior_loss=None
        if arm=='M1-mode':
            pl=e.prior_loss(x,rec);pl.backward();torch.nn.utils.clip_grad_norm_(e.prior.parameters(),1.);e.prior_opt.step();prior_loss=float(pl);e.prior_opt.zero_grad(set_to_none=True)
        e.opt.zero_grad(set_to_none=True);fm,task,mode,parts=e.decoder_losses(x,rec);lm=(e.lambda_mode or 0)*min((j+1)/200,1);loss=fm+e.parent['calibration']['lambda_ref']*task+lm*mode;loss.backward()
        assert all(p.grad is None for p in e.prior.parameters()),'decoder gradient leaked into prior'
        assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in e.parameters())
        gn=float(torch.nn.utils.clip_grad_norm_(e.parameters(),1.));e.opt.step();step=j+1;torch.cuda.synchronize()
        row=dict(step=step,prior_steps=warmup+step if arm=='M1-mode' else 0,FM=float(fm),task=float(task),mode=float(mode),prior_FM=prior_loss,lambda_mode=lm,loss=float(loss),gradient_norm=gn,prior_decoder_gradients_none=True,source_indices=rec['source_indices'],block_indices=rec['block_indices'],target_slots=rec['target_slots'],target_ids=x['ids'],valid_frames=int(x['batch']['source_lengths'].sum()),seconds=time.time()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),task_components=parts)
        rows.append(row)
        with (dest/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        write(dest/'status.json',dict(stage='decoder',actual=step,requested=6000,latest=row));print(arm,step,float(loss),row['seconds'],flush=True)
        if step%100==0 or step==stop:e.save(path,step,warmup,rows)
        if step in (1000,3000,6000):e.save(dest/'checkpoints'/f'step_{step:06d}.pt',step,warmup,rows)
    write(dest/'completion.json',dict(decoder_steps=step,prior_steps=warmup+step if arm=='M1-mode' else 0,training_complete=step==6000,evaluation_complete=False,diagnostic=diagnostic,seconds=time.time()-tick))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['M0-control','M1-mode'],required=True);p.add_argument('--stop',type=int,default=6000);p.add_argument('--diagnostic',action='store_true');p.add_argument('--warmup-stop',type=int,default=2000);p.add_argument('--diagnostic-label',default='smoke');a=p.parse_args();run(a.arm,a.stop,a.diagnostic,a.warmup_stop,a.diagnostic_label)
