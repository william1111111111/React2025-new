"""Full velocity NFT: 200 rounds x 20 actual updates. Frozen source path and ref."""
import argparse,random,time,hashlib,json
import numpy as np
import torch
import torch.nn.functional as F
from reaction_flow.train import configure_flow
from reaction_flow.data import draw
from reaction_flow.flow_path import flow_path,flow_loss
from hirp.train_phase25 import rng_state,restore_rng
from reward_policy.pilot import arrays,serial
from .common import *
from .model import Models
from .objective import nft,mean
from .rewards import components,weights

def tensor_hash(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

class Run:
    def __init__(self,arm):
        self.arm=arm;self.m=Models();self.opt=torch.optim.AdamW(self.m.student.parameters(),lr=2e-5,weight_decay=.01)
        self.dual=np.ones(3);self.ema=np.zeros(3);self.step=0;self.round=0;self.index=0;self.buffer=[];self.rows=[]
        self.cal=read(OUT/'CALIBRATION.json');self.schedule=read(OUT/'SCHEDULE.json');self.dest=OUT/'training'/arm;self.dest.mkdir(parents=True,exist_ok=True)
    def reference(self,e):
        p=OUT/'reference_bank'/(str(e['rec']['step'])+'.json')
        # Queue serializes N0 round sampling before N1 if necessary using a per-record file lock.
        import fcntl
        p.parent.mkdir(exist_ok=True)
        with p.with_suffix('.lock').open('w') as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            if p.exists():
                row=read(p);assert row['record']==e['rec'] and row['parent_sha256']==sha(PARENT)
                return arrays(row['scores']),False
            noise=self.m.env.noise(e,10,'reference');_,pred=self.m.generate(e,noise,self.m.ref);s=self.m.env.scores(e,pred)
            write(p,dict(record=e['rec'],parent_sha256=sha(PARENT),noise_sha256=tensor_hash(noise),scores=serial(s),new_reference_rollouts=10))
            return s,True
    def sample_round(self):
        oldhash=digest(self.m.old);self.buffer=[]
        for j,rec in enumerate(self.schedule['training'][self.round*16:(self.round+1)*16]):
            p=self.dest/'buffers'/f'round_{self.round:03d}_group_{j:02d}.pt'
            if p.exists():
                x=torch.load(p,map_location='cpu',weights_only=True);assert x['behavior_sha256']==oldhash and x['record']==rec
            else:
                e=self.m.env.episode(rec);noise=self.m.env.noise(e,10,'action');clean,pred=self.m.generate(e,noise,self.m.old)
                assert not clean.requires_grad and not pred.requires_grad
                s=self.m.env.scores(e,pred);ref,new=self.reference(e)
                q,b,support,violation,stats=components(s,ref,e['n'],[len(y) for y in e['targets']],self.dual)
                r=weights(q,b,support,self.arm,self.cal)
                x=dict(record=rec,clip_id=e['clip_id'],start=e['start'],n=e['n'],target_ids=e['target_ids'],h=e['h'].cpu(),mask=e['mask'].cpu(),clean=clean,pred=pred,r=torch.from_numpy(r),q=q.tolist(),b=b.tolist(),supported=support.tolist(),violation=violation.tolist(),stats=stats,behavior_sha256=oldhash,noise_sha256=tensor_hash(noise),condition_sha256=tensor_hash(e['h']),target_sha256=[tensor_hash(y) for y in e['targets']],frame_pts_sha256=hashlib.sha256(np.asarray(e['pts']).tobytes()).hexdigest(),noise_key=[9200123,rec['step'],'action'],reference_new=new,reference_bank=str(OUT/'reference_bank'/(str(rec['step'])+'.json')),scores=serial(s))
                save(p,x)
                write(p.with_suffix('.json'),{k:v for k,v in x.items() if k not in ('h','mask','clean','pred','r','scores')}|dict(r=r.tolist(),buffer_sha256=sha(p),new_action_rollouts=10))
            self.buffer.append(x);write(self.dest/'status.json',dict(stage='sampling',round=self.round,groups=j+1,total=16,actual_updates=self.step,requested=4000));print(self.arm,'round',self.round,'sample',j+1,flush=True)
        self.index=0;self.save(self.dest/'latest.pt')
    def forward(self,velocity,u,tau,h,mask,start,grad=False):return velocity(u,tau,h,mask,mask,torch.tensor([start],device='cuda'),gradient_checkpointing=grad)
    def update(self):
        tick=time.time();self.opt.zero_grad(set_to_none=True)
        before=[p.detach().clone() for p in self.m.student.parameters()]
        lr=2e-5*min(1,(self.step+1)/100)
        for g in self.opt.param_groups:g['lr']=lr
        losses=[];regs=[];replays=[]
        # Eight endpoints, each visited once per round, deterministic shuffled ordering.
        order=np.random.default_rng(123+self.round).permutation(160)
        for slot in order[self.index*8:(self.index+1)*8]:
            group,k=divmod(int(slot),10);x=self.buffer[group];U=x['clean'][k:k+1].cuda().detach();h=x['h'].cuda();mask=x['mask'].cuda();r=x['r'][k:k+1].cuda().detach()
            eps=draw(U.shape,123,self.step,f'nft_epsilon_{slot}').cuda().double();tau=(.02+.96*draw((1,),123,self.step,f'nft_tau_{slot}',False)).cuda().double();u,_=flow_path(U,eps,tau,mask)
            with torch.no_grad():old=self.forward(self.m.old,u,tau,h,mask,x['start']);ref=self.forward(self.m.ref,u,tau,h,mask,x['start'])
            new=self.forward(self.m.student,u,tau,h,mask,x['start'],True)
            loss=nft(new,old,u,U,tau,r,mask);reg=mean((new-ref).square(),mask).mean()
            ((loss+.01*reg)/8).backward();losses.append(float(loss));regs.append(float(reg))
        # Four independent actual TRAIN reactions, slots chosen independently of rewards.
        for i,rec in enumerate(self.schedule['replay'][4*self.step:4*self.step+4]):
            e=self.m.env.episode(rec);y=e['targets'][rec['target_slots'][0]];n=len(y);mask=torch.arange(750,device='cuda')[None]<n
            with torch.no_grad():U=self.m.env.generator.model.base.transform(F.pad(y,(0,0,0,750-n))[None].cuda().double())
            eps=draw(U.shape,123,self.step,f'replay_epsilon_{i}').cuda().double();tau=draw((1,),123,self.step,f'replay_tau_{i}',False).cuda().double();u,v=flow_path(U,eps,tau,mask)
            new=self.m.student(u,tau,e['h'],mask,e['mask'],torch.tensor([e['start']],device='cuda'),gradient_checkpointing=True)
            loss=flow_loss(new,v,mask);(.25*loss/4).backward();replays.append(float(loss))
        norm=float(torch.nn.utils.clip_grad_norm_(self.m.student.parameters(),1.));assert np.isfinite(norm)
        self.opt.step();delta=sum(float((p.detach()-b).square().sum()) for p,b in zip(self.m.student.parameters(),before))**.5
        assert all(p.grad is None and not p.requires_grad for p in self.m.old.parameters())
        assert all(p.grad is None and not p.requires_grad for p in self.m.env.generator.model.parameters())
        self.step+=1;self.index+=1
        row=dict(step=self.step,round=self.round,buffer_updates=self.index,nft=float(np.mean(losses)),FM_real=float(np.mean(replays)),ref_vector_field_MSE=float(np.mean(regs)),gradient_norm=norm,actual_parameter_delta=delta,lr=lr,seconds=time.time()-tick,dual=self.dual.tolist(),peak_GPU_GiB=torch.cuda.max_memory_allocated()/2**30)
        self.rows.append(row);write(self.dest/'status.json',dict(stage='training',actual_updates=self.step,requested=4000,latest=row));print(self.arm,row,flush=True)
        with (self.dest/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    def save(self,path):
        save(path,dict(student=cpu_state(self.m.student),old=cpu_state(self.m.old),ref_sha256=self.m.ref_hash,parent_sha256=sha(PARENT),optimizer=self.opt.state_dict(),dual=self.dual.tolist(),ema=self.ema.tolist(),step=self.step,round=self.round,buffer_index=self.index,buffer=self.buffer,rng=rng_state(),arm=self.arm,rows=self.rows,protocol_sha256=sha(OUT/'PROTOCOL.json')))
    def resume(self):
        p=self.dest/'latest.pt'
        if not p.exists():return
        s=torch.load(p,map_location='cpu',weights_only=True);assert s['arm']==self.arm and s['ref_sha256']==self.m.ref_hash and s['protocol_sha256']==sha(OUT/'PROTOCOL.json')
        self.m.student.load_state_dict(s['student']);self.m.old.load_state_dict(s['old']);self.opt.load_state_dict(s['optimizer']);self.dual=np.array(s['dual']);self.ema=np.array(s['ema']);self.step=s['step'];self.round=s['round'];self.index=s['buffer_index'];self.buffer=s['buffer'];self.rows=s['rows'];restore_rng(s['rng'])
        (self.dest/'training.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))

def main(arm):
    configure_flow();torch.set_num_threads(4);torch.manual_seed(123);np.random.seed(123);random.seed(123)
    protocol=read(OUT/'PROTOCOL.json')
    for p,h in protocol['inputs'].items():assert sha(p)==h,p
    for p,h in read(OUT/'CODE_IDENTITY.json').items():assert sha(p)==h,p
    run=Run(arm)
    import fcntl
    with (run.dest/'train.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run.resume()
        if run.step==0 and not (run.dest/'checkpoints/step_000000.pt').exists():run.save(run.dest/'checkpoints/step_000000.pt')
        while run.round<200:
            if not run.buffer:run.sample_round()
            while run.index<20:
                run.update();run.save(run.dest/'latest.pt')
                if run.step in (500,1000,2000,4000):run.save(run.dest/'checkpoints'/f'step_{run.step:06d}.pt')
            assert digest(run.m.ref)==run.m.ref_hash
            run.ema=.95*run.ema+.05*np.mean([x['violation'] for x in run.buffer],0);run.dual=np.clip(run.dual+.01*run.ema,0,20);run.m.synchronize();run.round+=1;run.index=0;run.buffer=[];run.save(run.dest/'latest.pt')
        write(run.dest/'completion.json',dict(actual_updates=run.step,rounds=run.round,action_rollouts=32000,real_replay_visits=16000,unique_replay_sources=len({r['source_indices'][0] for r in run.schedule['replay']}),training_complete=True))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);main(p.parse_args().arm)
