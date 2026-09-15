"""Actual TRAIN calibration and finite reachability; no DEV reward access."""
import time
import numpy as np
import torch
from reaction_flow.train import configure_flow
from .environment import TrainEnvironment
from .policy import Policy,replace_low_frequency
from .common import OUT,read,write
from .rewards import utility,coverage

def serial(scores):return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in scores.items()}
def arrays(scores):return {k:np.array(v) if isinstance(v,list) else v for k,v in scores.items()}

def valid(scores,scales):return scores['domain_valid']&(scores['speed']<=scales['speed_ceiling'])&(scores['accel']<=scales['accel_ceiling'])

def reference(env,rec,folder,k=10,kind='reference'):
    path=OUT/folder/(str(rec['step'])+'.json')
    if path.exists():return arrays(read(path)['scores'])
    tick=time.time();e=env.episode(rec);noise=env.noise(e,k,kind);pred=env.generate(e,noise);scores=env.scores(e,pred)
    write(path,dict(record=rec,scope='TRAIN block exact unsmoothed DTW; not official full recording FRD',candidate_count=k,noise_bank=kind,scores=serial(scores),seconds=time.time()-tick))
    return scores

def main():
    configure_flow();torch.set_num_threads(4);torch.manual_seed(123);env=TrainEnvironment();episodes=read(OUT/'episodes.json');tick=time.time()
    # Real generator identity, beyond algebra-only tests.
    e=env.episode(episodes['reachability'][0]);noise=env.noise(e,2,'reachability');policy=Policy().cuda().double();adjusted,_,_=replace_low_frequency(noise,e['basis'],policy,e['context']);assert torch.equal(noise,adjusted)
    original=env.generate(e,noise);changed=env.generate(e,adjusted);assert torch.equal(original,changed)
    write(OUT/'identity_acceptance.json',dict(real_source=e['clip_id'],noise_exact=True,prediction_exact=True,frames=e['n'],generator_frozen=all(not p.requires_grad for p in env.generator.model.parameters())))
    scores=[]
    for i,rec in enumerate(episodes['calibration']):
        scores.append(reference(env,rec,'calibration'));write(OUT/'pilot_status.json',dict(stage='calibration',completed=i+1,total=64,seconds=time.time()-tick));print('calibration',i+1,flush=True)
    def robust(x):return np.maximum(np.quantile(x,.75,axis=0)-np.quantile(x,.25,axis=0),1e-3)
    scales=dict(c=float(robust(np.concatenate([s['ccc'].max(1) for s in scores]))),d=float(robust(np.concatenate([s['dtw'].min(1) for s in scores]))),phi=robust(np.concatenate([s['phi'] for s in scores])).tolist(),speed_ceiling=float(np.quantile(np.concatenate([s['speed'] for s in scores]),.99)*2),accel_ceiling=float(np.quantile(np.concatenate([s['accel'] for s in scores]),.99)*2))
    if (OUT/'scales.json').exists():assert read(OUT/'scales.json')==scales
    else:write(OUT/'scales.json',scales)
    rows=[]
    for i,rec in enumerate(episodes['reachability']):
        ref=reference(env,rec,'reachability_reference');s=reference(env,rec,'reachability',32,'reachability');cf=float(np.quantile(ref['ccc'].max(1),.1));dc=float(np.quantile(ref['dtw'].min(1),.9));u=utility(s['ccc'],s['dtw'],valid(s,scales),s['phi'],s['target_phi'],np.array(scales['phi']),cf,dc)
        ix=np.flatnonzero(u.max(1)>0);phi=s['phi'][ix];dist=np.mean(((phi[:,None]-phi[None])/np.array(scales['phi']))**2,-1) if len(ix)>1 else np.zeros((1,1));supported=len(ix)>1 and float(dist.max())>.25
        rows.append(dict(record=rec,qualified_candidates=len(ix),max_supported_phi_distance=float(dist.max()),coverage32=coverage(u),diverse_support=bool(supported),diagnostic_only=True))
        write(OUT/'pilot_status.json',dict(stage='reachability',completed=i+1,total=16));print('reachability',i+1,rows[-1]['qualified_candidates'],supported,flush=True)
    count=sum(r['diverse_support'] for r in rows);write(OUT/'REACHABILITY.json',dict(rows=rows,sources_with_diverse_support=count,gate_passed=count>=4,seconds=time.time()-tick,scope='finite TRAIN diagnostic; does not establish capacity upper bound',actual_candidate_rollouts=64*10+16*(32+10)+4))
    write(OUT/'pilot_status.json',dict(stage='complete',gate_passed=count>=4));print('reachability gate',count,'/16',flush=True)
if __name__=='__main__':main()
