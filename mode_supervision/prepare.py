"""Audit all TRAIN source feature clocks; freeze matched block/target schedule."""
import concurrent.futures, hashlib, json, random, subprocess, time
from pathlib import Path
import numpy as np
from hirp.paired_data import PairedReactionDataset
from hirp.session_data import SessionPopulation
OUT=Path('runs/reaction_flow/mode_supervision_v1')
PARENT=Path('runs/reaction_flow/task_dynamics_v1/T0-task/attempt_000/checkpoints/step_014000.pt')
PARENT_SHA='e6199183cdc3526e1fc1390c89057e40514af8d2f2101d69a5e1dabb7f43f4d6'

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def inspect(arg):
    index,path=arg;rel=path.relative_to(Path('data/train/facial-attributes'))
    key=rel.with_suffix('').as_posix();dest=OUT/'clocks'/f'{index:04d}.json'
    if dest.exists():return json.loads(dest.read_text())
    video=Path('data/train/video-face-crop')/rel.with_suffix('.mp4')
    rows=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','json',str(video)],timeout=180))['frames']
    pts=np.array([float(x['best_effort_timestamp_time']) for x in rows])
    if not len(pts) or not np.isfinite(pts).all() or not (np.diff(pts)>0).all():raise ValueError(f'invalid PTS: {key}')
    files={};lengths={}
    for folder,width in [('audio-features',768),('facial-attributes',25),('coefficients',58)]:
        p=Path('data/train')/folder/rel;a=np.load(p,mmap_mode='r');a=a[:,0] if a.ndim==3 else a
        if a.ndim!=2 or a.shape[1]!=width or not np.isfinite(a).all():raise ValueError(f'invalid source {p}')
        files[str(p)]=sha(p);lengths[folder]=len(a)
    n=min(lengths.values())
    if len(pts)!=n:raise ValueError(f'PTS/source mismatch: {key} {len(pts)} {lengths}')
    target=Path('data/train/facial-attributes/listener')/rel.relative_to('speaker')
    a=np.load(target,mmap_mode='r');paired_length=len(a)
    if a.ndim!=2 or a.shape[1]!=25 or not np.isfinite(a).all() or paired_length<1:
        raise ValueError(f'paired trajectory does not cover source: {key}')
    row=dict(index=index,clip_id=key,source_length=n,paired_length=paired_length,frame_pts=pts.tolist(),source_hashes=files,video=str(video),video_sha256=sha(video),target_path=str(target),target_sha256=sha(target),timeline_evidence='actual speaker video PTS; exact feature frame-count correspondence; no independently verified sensor synchronization')
    write(dest,row);return row

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    assert sha(PARENT)==PARENT_SHA
    pool=SessionPopulation('data','train',750);start=time.time();rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for row in ex.map(inspect,enumerate(pool.dataset.records)):
            rows.append(row)
            if len(rows)%50==0:write(OUT/'prepare_status.json',dict(stage='PTS audit',completed=len(rows),total=len(pool.dataset),seconds=time.time()-start));print('audited',len(rows),flush=True)
    write(OUT/'cohort.json',rows)
    rng=random.Random(123);sessions=sorted(pool.sources);weights=[len(pool.sources[s]) for s in sessions];records=[]
    for step in range(8001):
        session=rng.choices(sessions,weights=weights,k=1)[0];indices=rng.choices(pool.sources[session],k=4)
        records.append(dict(step=14001+step,session_id=session,source_indices=indices,block_indices=[rng.randrange((rows[i]['source_length']+749)//750) for i in indices],target_slots=[rng.randrange(4) for _ in indices]))
    write(OUT/'schedule.json',dict(calibration=records[0],prior_warmup=records[1:2001],decoder=records[2001:]))
    write(OUT/'PROTOCOL.json',dict(parent=str(PARENT),parent_sha256=PARENT_SHA,arms=['M0-control','M1-mode'],decoder_steps=6000,prior_warmup_steps=2000,checkpoints=[1000,3000,6000],seed=123,gpu=7,B=4,T=750,K_train=4,K_eval=10,euler_steps=16,source_population=len(rows),schedule_sha256=sha(OUT/'schedule.json'),cohort_sha256=sha(OUT/'cohort.json'),lambda_mode_calibration='once on held-out TRAIN schedule batch; 0.1 times inherited weighted-task gradient norm; 200 step ramp',quality_tolerance=dict(FRC_min_ratio=1.,FRD_max_ratio=1.,SMSE_min_ratio=1.5),prior_task_gradient=False,semantics=False,paid_API_calls=0,target_policy='uniform slot 0 paired + 3 deterministic same-session alternatives; paired original timeline; alternatives original full-length linear interpolate align_corners=True',sampling_policy='source-proportional session; uniform source within session; uniform generation block',normalization='TRAIN-only plan statistics; masked coefficient standard deviations floor 0.001; equal transformed AU/VA/expression groups',no_automatic_push=True,no_budget_extension=True))
    write(OUT/'prepare_status.json',dict(stage='complete',completed=len(rows),seconds=time.time()-start))
    print('complete',len(rows),flush=True)
if __name__=='__main__':main()
