"""Fixed eight independent banks, unchanged per-bank estimators."""
import argparse,json,time
import numpy as np
import torch
from torch.utils.data import default_collate
from .phase24 import *
from .phase23_cache import eval_identity
from .evaluate_phase22 import evaluate_model
from .evaluate_phase2 import reference_cache_and_baseline
from .group_features import DescriptorScaler
from .session_data import SessionPopulation
from .train_phase21 import configure


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:6');a=p.parse_args();configure();plan,snapshot=runtime()
    protocol=json.loads((ROOT/'mc_protocol.json').read_text());assert sha256_file(PLAN)==protocol['plan_sha256']
    pool=SessionPopulation(Path('data'),'val',128);batch=default_collate([pool.dataset[i] for i in plan['indices']]);assert batch['clip_id']==plan['clip_ids']
    crops=[dict(clip_id=batch['clip_id'][i],source_length=int(batch['source_lengths'][i]),pair_length=int(batch['pair_lengths'][i]),crop_start=int(batch['crop_start'][i])) for i in range(80)]
    sp=Path('runs/phase2/descriptor_stats.json');scaler=DescriptorScaler.load(sp);sg=DescriptorScaler.load(sp).to(a.device);refs,_=reference_cache_and_baseline(pool,scaler)
    out=ROOT/'mc';out.mkdir(exist_ok=True);completed=[];start=time.perf_counter()
    for point in json.loads((ROOT/'candidate_manifest.json').read_text()):
        model,meta=load_candidate(point,a.device)
        if not (ROOT/'unchanged_metric_regression.json').exists():
            bank=torch.from_numpy(np.load(plan['banks'][0]['path']))[:,:32]
            result=evaluate_model(model,batch,refs,scaler,sg,bank,torch.tensor(plan['permutations'][0]['donors']),a.device,32);result.pop('all_predictions')
            old=json.loads((OLD/f"evaluation_v1/seed_{point['seed']}/{point['arm']}_lambda{point['lambda_group']:g}_bank0_K32.json").read_text())['result']
            from .verify_phase23 import numbers
            aa=numbers(result['aggregate']);bb=numbers(old['aggregate']);err=max(abs(aa[k]-bb[k]) for k in aa);assert err<3e-6
            write(ROOT/'unchanged_metric_regression.json',dict(max_aggregate_error=err,checkpoint=point['checkpoint'],old_protocol_K=32,old_bank=0,scope='same fixed input/noise, unchanged historical metric function'))
        for bank_info in protocol['banks']:
            normalization(plan);verify_files([bank_info])
            identity=eval_identity(point['checkpoint'],meta,PLAN,plan,bank_info,64,sp,crops,dict(schema='phase24',dependencies_sha256=canonical_hash(snapshot),mc_protocol_sha256=sha256_file(ROOT/'mc_protocol.json'),torch=torch.__version__))
            identity['candidate_manifest_sha256']=sha256_file(ROOT/'candidate_manifest.json')
            path=out/f"seed{point['seed']}_{point['arm']}_lambda{point['lambda_group']:g}_bank{bank_info['seed']}.json"
            previous=guarded(plan,lambda:cached(path,identity))
            if previous is None:
                bank=torch.from_numpy(np.load(bank_info['path']));r=guarded(plan,lambda:evaluate_model(model,batch,refs,scaler,sg,bank,torch.tensor(plan['permutations'][0]['donors']),a.device,64));r.pop('all_predictions')
                r.update(seed=point['seed'],arm=point['arm'],lambda_group=point['lambda_group'],bank=bank_info['seed'],K=64);save_cached(path,identity,r)
                print('MC',path.name,r['aggregate']['conditional']['loss'],r['aggregate']['marginal']['loss'],flush=True)
            completed.append(dict(path=str(path),sha256=sha256_file(path)))
        del model,meta;torch.cuda.empty_cache()
    complete(out/'completed.json',completed)
    if not (out/'resources.json').exists():write(out/'resources.json',dict(wall_seconds=time.perf_counter()-start,device=a.device,cases=len(completed),new_training_steps=0))
    print('COMPLETE',len(completed),flush=True)


if __name__=='__main__':main()
