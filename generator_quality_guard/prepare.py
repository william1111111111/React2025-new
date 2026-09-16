import random,numpy as np
from .common import *
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'DESIGN.json').exists():return
 old=read(NFT/'SCHEDULE.json');rows=read('runs/reaction_flow/mode_supervision_v1/cohort.json');rng=random.Random(8400123);quality=[]
 for j in range(1000):
  idx=rng.choice(old['population_indices']);r=rows[idx];bs=[b for b in range((r['source_length']+749)//750) if min(r['source_length'],r['paired_length'])-750*b>=2];quality.append(dict(step=8400000+j,session_id=r['clip_id'].split('/')[1],source_indices=[idx],block_indices=[rng.choice(bs)],target_slots=[0]))
 write(OUT/'SCHEDULE.json',dict(training=old['training'][:800],replay=old['replay'][:4000],quality=quality,population=old['population'],population_indices=old['population_indices']))
 cal=[read(p) for p in sorted((NFT/'calibration').glob('*.json'))];assert len(cal)==64;q=np.array([r['q'] for r in cal]);b=np.array([r['b'] for r in cal]);assert (b>1e-6).any()
 write(OUT/'CALIBRATION.json',dict(Z_Q=float(max(q.std(),1e-3)),Z_B=float(max(b.std(),1e-3)),scope='64 fixed TRAIN calibration groups from same P2; q_abs not centered in weights',centered=False,inputs={str(p):sha(p) for p in sorted((NFT/'calibration').glob('*.json'))}))
 write(OUT/'DESIGN.json',dict(arms=ARMS,parent_sha256=sha(PARENT),max_proposed=1000,transaction_steps=100,consecutive_rejection_limit=3,seed=123,lr=2e-5,warmup_proposed_steps=100,weight_decay=.01,clip=1.,NFT=dict(beta=1.,K=10,groups_per_round=16,batch=8,round_updates=20,old_EMA=.5),quality_rollout=dict(K=4,T_max=750,Euler=16,mu_C=1.,mu_D=1.,fixed=True,delta_C=0.,delta_D=0.),monitor=dict(TRAIN_recordings=16,K=10,target_K=10,delta_C=0.,delta_D=0.,tolerance='0 initially; any numerical tolerance must be fixed by preflight equivalence before launch'),rollback='student/optimizer/old/RNG restored to last acceptance; buffers discarded; proposed counter and next schedule/noise IDs always advance; fixed mu, no LR adaptation',no_DEV_TEST_control=True,no_auto_push=True))
if __name__=='__main__':main()
