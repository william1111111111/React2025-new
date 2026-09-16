"""Full velocity NFT: at most 50 rounds x 20 proposals, with transactional acceptance."""
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
from generator_reward_nft.model import Models
from generator_reward_nft.objective import nft,mean
from .rewards import components,weights
from .quality import rollout_student_with_grad,scores,hinges
from .monitor import evaluate as monitor_evaluate
import copy

def tensor_hash(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

class Run:
    def __init__(self,arm):
        self.arm=arm;self.m=Models();self.opt=torch.optim.AdamW(self.m.student.parameters(),lr=2e-5,weight_decay=.01)
        self.dual=np.ones(3);self.ema=np.zeros(3);self.step=0;self.round=0;self.index=0;self.buffer=[];self.rows=[];self.accepted=0;self.rejected=0;self.consecutive_rejections=0
        self.cal=read(OUT/'CALIBRATION.json');self.schedule=read(OUT/'SCHEDULE.json');self.dest=OUT/'training'/arm;self.dest.mkdir(parents=True,exist_ok=True)
    def reference(self,e):
        p=OUT/'reference_bank'/(str(e['rec']['step'])+'.json')
        prior=NFT/'reference_bank'/(str(e['rec']['step'])+'.json')
        if prior.exists():
            row=read(prior);assert row['record']==e['rec'] and row['parent_sha256']==sha(PARENT);return arrays(row['scores']),False
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
                r=weights(q,b,support,self.arm,self.cal,stats['coverage_group_feasible'])
                all_below=(s['ccc'].max(1)<stats['C0'])&(s['dtw'].min(1)>stats['D0'])
                stats.update(all_below_reference_positive_rate=float(((r>.5)&all_below).sum()/max(1,all_below.sum())),positive_weight_rate=float((r>.5).mean()),coverage_enabled=bool(stats['coverage_group_feasible'] and self.arm=='Q1-coverage-guarded'))
                if not stats['coverage_group_feasible']:
                    assert not ((r>.5)&all_below).any()
                x=dict(record=rec,clip_id=e['clip_id'],start=e['start'],n=e['n'],target_ids=e['target_ids'],h=e['h'].cpu(),mask=e['mask'].cpu(),clean=clean,pred=pred,r=torch.from_numpy(r),q=q.tolist(),b=b.tolist(),supported=support.tolist(),violation=violation.tolist(),stats=stats,behavior_sha256=oldhash,noise_sha256=tensor_hash(noise),condition_sha256=tensor_hash(e['h']),target_sha256=[tensor_hash(y) for y in e['targets']],frame_pts_sha256=hashlib.sha256(np.asarray(e['pts']).tobytes()).hexdigest(),noise_key=[9200123,rec['step'],'action'],reference_new=new,reference_bank=str(OUT/'reference_bank'/(str(rec['step'])+'.json')),scores=serial(s))
                save(p,x)
                write(p.with_suffix('.json'),{k:v for k,v in x.items() if k not in ('h','mask','clean','pred','r','scores')}|dict(r=r.tolist(),buffer_sha256=sha(p),new_action_rollouts=10))
            self.buffer.append(x);write(self.dest/'status.json',dict(stage='sampling',round=self.round,groups=j+1,total=16,proposed_updates=self.step,accepted_updates=self.accepted,rejected_updates=self.rejected,max_proposed=1000));print(self.arm,'round',self.round,'sample',j+1,flush=True)
        self.index=0;self.save(self.dest/'latest.pt')
    def forward(self,velocity,u,tau,h,mask,start,grad=False):return velocity(u,tau,h,mask,mask,torch.tensor([start],device='cuda'),gradient_checkpointing=grad)
    def update(self):
        tick=time.time();self.opt.zero_grad(set_to_none=True)
        before=[p.detach().clone() for p in self.m.student.parameters()]
        lr=2e-5*min(1,(self.step+1)/100)
        for g in self.opt.param_groups:g['lr']=lr
        losses=[];regs=[];replays=[];diagnostic=(self.step==0 or (self.step+1)%20==0);parts={k:torch.zeros(sum(p.numel() for p in self.m.student.parameters()),device='cuda',dtype=torch.double) for k in ['NFT','ref_field','real_FM']} if diagnostic else {}
        # Eight endpoints, each visited once per round, deterministic shuffled ordering.
        order=np.random.default_rng(123+self.round).permutation(160)
        for slot in order[self.index*8:(self.index+1)*8]:
            group,k=divmod(int(slot),10);x=self.buffer[group];U=x['clean'][k:k+1].cuda().detach();h=x['h'].cuda();mask=x['mask'].cuda();r=x['r'][k:k+1].cuda().detach()
            eps=draw(U.shape,123,self.step,f'nft_epsilon_{slot}').cuda().double();tau=(.02+.96*draw((1,),123,self.step,f'nft_tau_{slot}',False)).cuda().double();u,_=flow_path(U,eps,tau,mask)
            with torch.no_grad():old=self.forward(self.m.old,u,tau,h,mask,x['start']);ref=self.forward(self.m.ref,u,tau,h,mask,x['start'])
            new=self.forward(self.m.student,u,tau,h,mask,x['start'],True)
            loss=nft(new,old,u,U,tau,r,mask);reg=mean((new-ref).square(),mask).mean()
            if diagnostic:
                parts['NFT']+=self.grad_vector(loss/8);parts['ref_field']+=self.grad_vector(.01*reg/8)
            ((loss+.01*reg)/8).backward();losses.append(float(loss));regs.append(float(reg))
        # Four independent actual TRAIN reactions, slots chosen independently of rewards.
        for i,rec in enumerate(self.schedule['replay'][4*self.step:4*self.step+4]):
            e=self.m.env.episode(rec);y=e['targets'][rec['target_slots'][0]];n=len(y);mask=torch.arange(750,device='cuda')[None]<n
            with torch.no_grad():U=self.m.env.generator.model.base.transform(F.pad(y,(0,0,0,750-n))[None].cuda().double())
            eps=draw(U.shape,123,self.step,f'replay_epsilon_{i}').cuda().double();tau=draw((1,),123,self.step,f'replay_tau_{i}',False).cuda().double();u,v=flow_path(U,eps,tau,mask)
            new=self.m.student(u,tau,e['h'],mask,e['mask'],torch.tensor([e['start']],device='cuda'),gradient_checkpointing=True)
            loss=flow_loss(new,v,mask)
            if diagnostic:parts['real_FM']+=self.grad_vector(.25*loss/4)
            (.25*loss/4).backward();replays.append(float(loss))
        quality=self.quality_update(diagnostic)
        component_gradients={k:float(v.norm()) for k,v in parts.items()} if diagnostic else None
        norm=float(torch.nn.utils.clip_grad_norm_(self.m.student.parameters(),1.));assert np.isfinite(norm)
        self.opt.step();delta=sum(float((p.detach()-b).square().sum()) for p,b in zip(self.m.student.parameters(),before))**.5
        assert all(p.grad is None and not p.requires_grad for p in self.m.old.parameters())
        assert all(p.grad is None and not p.requires_grad for p in self.m.env.generator.model.parameters())
        self.step+=1;self.index+=1
        row=dict(step=self.step,quality=quality,component_gradient_norms=component_gradients,gradient_diagnostic=diagnostic,round=self.round,buffer_updates=self.index,nft=float(np.mean(losses)),FM_real=float(np.mean(replays)),ref_vector_field_MSE=float(np.mean(regs)),gradient_norm=norm,actual_parameter_delta=delta,lr=lr,seconds=time.time()-tick,dual=self.dual.tolist(),peak_GPU_GiB=torch.cuda.max_memory_allocated()/2**30)
        self.rows.append(row);write(self.dest/'status.json',dict(stage='training',proposed_updates=self.step,accepted_updates=self.accepted,rejected_updates=self.rejected,max_proposed=1000,latest=row));print(self.arm,row,flush=True)
        with (self.dest/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    def grad_vector(self,loss):
        gs=torch.autograd.grad(loss,list(self.m.student.parameters()),retain_graph=True,allow_unused=True)
        return torch.cat([(g if g is not None else torch.zeros_like(p)).detach().reshape(-1) for p,g in zip(self.m.student.parameters(),gs)])
    def quality_update(self,diagnostic):
        rec=self.schedule['quality'][self.step];e=self.m.env.episode(rec);noise=self.m.env.noise(e,4,'action')
        path=OUT/'quality_reference'/f"{rec['step']}.json";path.parent.mkdir(exist_ok=True)
        import fcntl
        with path.with_suffix('.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if path.exists():ref=read(path);assert ref['record']==rec and ref['parent_sha256']==sha(PARENT);new_ref=False
            else:
                _,pred=self.m.generate(e,noise,self.m.ref)
                with torch.no_grad():c,d,_,_=scores(list(pred),e['targets'])
                ref=dict(record=rec,parent_sha256=sha(PARENT),C=float(c),D=float(d),noise_sha256=tensor_hash(noise));write(path,ref);new_ref=True
        pred=rollout_student_with_grad(self.m,e,noise);c,d,C,D=scores(pred,e['targets']);scales=read(OLD/'scales.json');hc,hd=hinges(c,d,c.new_tensor(ref['C']),d.new_tensor(ref['D']),scales)
        grads={}
        if diagnostic:grads=dict(CCC_protection=float(self.grad_vector(hc).norm()),distance_protection=float(self.grad_vector(hd).norm()))
        (hc+hd).backward()
        return dict(source=e['clip_id'],record=rec,target_ids=e['target_ids'],noise_sha256=tensor_hash(noise),C=float(c),D=float(d),C0=ref['C'],D0=ref['D'],violation_C=ref['C']-float(c),violation_D=float(d)-ref['D'],h_C=float(hc),h_D=float(hd),gradient_norms=grads,student_rollouts=4,reference_new_rollouts=4 if new_ref else 0,reference_reused_rollouts=0 if new_ref else 4,independent_quality_schedule=True)
    def rollback(self,path):
        s=torch.load(path,map_location='cpu',weights_only=True);assert s['ref_sha256']==self.m.ref_hash
        self.m.student.load_state_dict(s['student']);self.m.old.load_state_dict(s['old']);self.opt.load_state_dict(s['optimizer']);restore_rng(s['rng']);self.buffer=[];self.index=0
        # Proposed count, rejection evidence, fixed controllers and next schedule IDs are not rewound.
        assert digest(self.m.student)==digest_state(s['student']) and digest(self.m.old)==digest_state(s['old'])
    def save(self,path):
        save(path,dict(student=cpu_state(self.m.student),old=cpu_state(self.m.old),ref_sha256=self.m.ref_hash,parent_sha256=sha(PARENT),optimizer=self.opt.state_dict(),accepted_steps=self.accepted,rejected_steps=self.rejected,consecutive_rejections=self.consecutive_rejections,control_state=dict(mu_C=1.,mu_D=1.,lr_policy='fixed proposal-index warmup; no relaxation'),dual=self.dual.tolist(),ema=self.ema.tolist(),step=self.step,round=self.round,buffer_index=self.index,buffer=self.buffer,rng=rng_state(),arm=self.arm,rows=self.rows,protocol_sha256=sha(OUT/'PROTOCOL.json')))
    def resume(self):
        p=self.dest/'latest.pt'
        if not p.exists():return
        s=torch.load(p,map_location='cpu',weights_only=True)
        ap=self.dest/'accepted_latest.pt'
        if ap.exists():
            accepted_state=torch.load(ap,map_location='cpu',weights_only=True)
            if accepted_state['accepted_steps']>s['accepted_steps']:
                s=accepted_state
        assert s['arm']==self.arm and s['ref_sha256']==self.m.ref_hash and s['protocol_sha256']==sha(OUT/'PROTOCOL.json')
        self.m.student.load_state_dict(s['student']);self.m.old.load_state_dict(s['old']);self.opt.load_state_dict(s['optimizer']);self.dual=np.array(s['dual']);self.ema=np.array(s['ema']);self.step=s['step'];self.round=s['round'];self.index=s['buffer_index'];self.buffer=s['buffer'];self.rows=s['rows'];self.accepted=s['accepted_steps'];self.rejected=s['rejected_steps'];self.consecutive_rejections=s['consecutive_rejections'];restore_rng(s['rng'])
        (self.dest/'training.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))

def digest_state(state):
    h=hashlib.sha256()
    for k,v in state.items():h.update(k.encode());h.update(v.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def main(arm):
    configure_flow();torch.set_num_threads(4);torch.manual_seed(123);np.random.seed(123);random.seed(123)
    for p,h in read(OUT/'PROTOCOL.json')['inputs'].items():assert sha(p)==h,p
    for p,h in read(OUT/'CODE_IDENTITY.json').items():assert sha(p)==h,p
    run=Run(arm);import fcntl
    with (run.dest/'train.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run.resume();accepted=run.dest/'accepted_latest.pt'
        if not accepted.exists():run.save(accepted);run.save(run.dest/'checkpoints/accepted_000000.pt')
        reference=read(OUT/'monitor/P2_reference/summary.json')
        while run.step<1000 and run.consecutive_rejections<3:
            # A saved 100-step boundary is monitored before any subsequent proposal.
            if run.step and run.step%100==0 and run.index==20:
                proposal=run.dest/'checkpoints'/f'proposal_{run.step:06d}.pt'
                if not proposal.exists():run.save(proposal)
                measured=monitor_evaluate(run.m,run.m.student,f'{arm}_proposal_{run.step:06d}')
                passC=measured['FRC']>=reference['FRC'];passD=measured['FRD']<=reference['FRD'];ok=passC and passD
                row=dict(proposed=run.step,accepted=ok,FRC_pass=passC,FRD_pass=passD,reference_FRC=reference['FRC'],reference_FRD=reference['FRD'],FRC=measured['FRC'],FRD=measured['FRD'],S_MSE=measured['S_MSE'],source_deltas=[dict(index=i,FRC=x['FRC']-y['FRC'],FRD=x['FRD']-y['FRD']) for i,(x,y) in enumerate(zip(measured['rows'],reference['rows']))],tolerances=dict(FRC=0.,FRD=0.))
                if ok:
                    run.accepted+=100;run.consecutive_rejections=0;run.m.synchronize();run.round=run.step//20;run.index=0;run.buffer=[];run.save(accepted);run.save(run.dest/'checkpoints'/f'accepted_{run.step:06d}.pt')
                else:
                    run.rejected+=100;run.consecutive_rejections+=1;run.rollback(accepted);run.round=run.step//20
                row.update(accepted_steps=run.accepted,rejected_steps=run.rejected,consecutive_rejections=run.consecutive_rejections)
                write(run.dest/f'transaction_{run.step:06d}.json',row);run.save(run.dest/'latest.pt');print('TRANSACTION',arm,row,flush=True)
                if run.step>=1000 or run.consecutive_rejections>=3:break
            if not run.buffer:run.sample_round()
            while run.index<20:
                run.update();run.save(run.dest/'latest.pt')
            assert digest(run.m.ref)==run.m.ref_hash
            if run.step%100!=0:
                run.m.synchronize();run.round=run.step//20;run.index=0;run.buffer=[];run.save(run.dest/'latest.pt')
        # Handle endpoint boundary even when exactly at max proposal budget.
        if run.step==1000 and run.index==20:
            proposal=run.dest/'checkpoints'/f'proposal_{run.step:06d}.pt';run.save(proposal)
            measured=monitor_evaluate(run.m,run.m.student,f'{arm}_proposal_{run.step:06d}')
            passC=measured['FRC']>=reference['FRC'];passD=measured['FRD']<=reference['FRD'];ok=passC and passD
            if ok:
                run.accepted+=100;run.consecutive_rejections=0;run.m.synchronize();run.buffer=[];run.index=0;run.round=50;run.save(accepted);run.save(run.dest/'checkpoints/accepted_001000.pt')
            else:run.rejected+=100;run.consecutive_rejections+=1;run.rollback(accepted)
            write(run.dest/'transaction_001000.json',dict(proposed=1000,accepted=ok,FRC_pass=passC,FRD_pass=passD,FRC=measured['FRC'],FRD=measured['FRD'],reference_FRC=reference['FRC'],reference_FRD=reference['FRD'],accepted_steps=run.accepted,rejected_steps=run.rejected));run.save(run.dest/'latest.pt')
        write(run.dest/'completion.json',dict(proposed_steps=run.step,accepted_steps=run.accepted,rejected_steps=run.rejected,consecutive_rejections=run.consecutive_rejections,stop_reason='three consecutive rejected transactions' if run.consecutive_rejections>=3 else '1000 proposed budget',last_proposal=str(run.dest/'checkpoints'/f'proposal_{run.step:06d}.pt'),last_accepted=str(accepted),fixed_ref_sha256=run.m.ref_hash))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);main(p.parse_args().arm)
