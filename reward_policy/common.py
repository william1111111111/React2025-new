import hashlib,json,random
from pathlib import Path
from mode_supervision.prepare import write,sha
OUT=Path('runs/reaction_flow/reward_policy_v1')
BERT=Path('runs/reaction_flow/bert_semantic_v1')
PARENT=BERT/'training/P2-bert/checkpoints/step_001000.pt'
ARMS=('R-quality','R-distance','R-coverage')

def read(p):return json.loads(Path(p).read_text())

def prepare():
    assert not (OUT/'PROTOCOL.json').exists()
    side=read(PARENT.with_suffix('.json'));assert sha(PARENT)==side['sha256']
    task=read(BERT/'task_multitarget/seed123_P2-bert_step15000.json');frd=read(BERT/'frd20/seed123_P2-bert_step15000/status_000.json');assert frd['completed_pairs']==2000
    assert task['eval_identity']['checkpoint_sha256']==side['sha256']
    rows=read('runs/reaction_flow/mode_supervision_v1/cohort.json');sessions={}
    for r in rows:
        blocks=[b for b in range((r['source_length']+749)//750) if min(r['source_length'],r['paired_length'])-b*750>=2]
        if blocks:sessions.setdefault(r['clip_id'].split('/')[1],[]).append((r['index'],blocks))
    rng=random.Random(123);names=sorted(sessions)
    def episode(j,session):
        idx,blocks=rng.choice(sessions[session]);return dict(step=200000+j,session_id=session,source_indices=[idx],block_indices=[rng.choice(blocks)],target_slots=[0])
    calibration=[episode(j,names[j%len(names)]) for j in range(64)]
    reachability=[episode(1000+j,names[j%len(names)]) for j in range(16)]
    schedule=[episode(2000+j,rng.choices(names,weights=[len(sessions[s]) for s in names])[0]) for j in range(1000)]
    write(OUT/'episodes.json',dict(calibration=calibration,reachability=reachability,training=schedule))
    frozen=[PARENT,PARENT.with_suffix('.json'),BERT/'task_multitarget/seed123_P2-bert_step15000.json',BERT/'frd20/seed123_P2-bert_step15000/status_000.json',BERT/'cohort.json',BERT/'dev_sources.json',BERT/'content/index.json',BERT/'content/vectors.npy',Path('runs/reaction_flow/mode_supervision_v1/cohort.json'),OUT/'episodes.json']
    write(OUT/'PROTOCOL.json',dict(version='quality-constrained-noise-policy-v1',parent_sha256=side['sha256'],parent_metrics={k:task['result']['metrics'][k] for k in ('FRC','smse','FRVar')}|{'exact_FRD20':frd['FRD']},arms=ARMS,seed=123,updates=500,B=2,K=10,checkpoints=[100,500],rollouts_per_arm=10000,calibration_rollouts=640,reachability_rollouts=512,independent_reference_rollouts=10000,lr=1e-5,weight_decay=0.,clip=1.,couplings=4,width=128,log_scale_bound=.25,beta_KL=.01,KL_stop_total=20.,dual=dict(initial=[1.,1.,1.],lr=.01,range=[0.,20.],EMA=.95),thresholds=dict(CCC_quantile=.1,DTW_quantile=.9,motion_quantile=.99,motion_multiplier=2.,source='TRAIN independent parent episode groups'),scales='TRAIN pooled IQR floor1e-3; Phi per-coordinate IQR floor1e-3',reward='exact unsmoothed DTW over valid actual block frames; pair-supported gated coverage; original K normalization of leave-one-out quality',reachability_gate='at least 4/16 sources with at least two quality-valid candidates whose normalized Phi distance >0.25 and positive coverage; otherwise report pilot and do not start policy training',quality_tolerance=dict(delta_C=0.,delta_D=0.,FRC_min_parent_ratio=1.,FRD_max_parent_ratio=1.,SMSE_min_parent_ratio=1.5),visibility='TRAIN rewards only; speaker-only inference; existing legal BERT events where available, otherwise NULL',gpu=7,paid_API_calls=0,frozen_inputs={str(p):sha(p) for p in frozen},no_automatic_push=True))
    print('locked parent',side['sha256'],flush=True)
if __name__=='__main__':prepare()
