"""Bank0-only checkpoint learning curves, separate from two-bank final summaries."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate
from .evaluate_phase23 import points,OLD
from .evaluate_phase22 import evaluate_model
from .evaluate_phase2 import reference_cache_and_baseline
from .phase23_cache import eval_identity,cached,save_cached,complete,implementation_identity
from .phase22 import load_checkpoint
from .session_data import SessionPopulation
from .group_features import DescriptorScaler
from .phase15_audit import sha256_file,canonical_hash
from .train_phase21 import configure


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('runs/phase23/tradeoff_v1'));p.add_argument('--seed',type=int,required=True);p.add_argument('--device',default='cuda:7');a=p.parse_args();configure()
    plan_path=OLD/'evaluation_plan/manifest.json';plan=json.loads(plan_path.read_text());pool=SessionPopulation(Path('data'),'val',128)
    batch=default_collate([pool.dataset[i] for i in plan['indices']]);assert batch['clip_id']==plan['clip_ids']
    crops=[dict(clip_id=batch['clip_id'][i],source_length=int(batch['source_lengths'][i]),pair_length=int(batch['pair_lengths'][i]),crop_start=int(batch['crop_start'][i])) for i in range(80)]
    scaler_path=Path('runs/phase2/descriptor_stats.json');scaler=DescriptorScaler.load(scaler_path);sg=DescriptorScaler.load(scaler_path).to(a.device);refs,_=reference_cache_and_baseline(pool,scaler)
    bank_info=plan['banks'][0];bank=torch.from_numpy(np.load(bank_info['path']))[:,:32];versions=dict(source=implementation_identity(),curve=sha256_file(__file__),torch=torch.__version__,precision='FP32, TF32 disabled',aggregation='bank0 only')
    out=a.root/'learning_curves_bank0'/f'seed_{a.seed}';out.mkdir(parents=True,exist_ok=True);rows=[]
    selected=points(a.root,a.seed);assert len(selected)==7
    for point in selected:
        summary=json.loads((Path(point['run'])/point['arm']/'attempt_000/summary.json').read_text())
        for cp in summary['checkpoints']:
            assert sha256_file(cp['path'])==cp['sha256'];model,meta=load_checkpoint(cp['path'],a.device)
            identity=eval_identity(cp['path'],meta,plan_path,plan,bank_info,32,scaler_path,crops,versions)
            path=out/f"{point['arm']}_lambda{point['lambda_group']:g}_step{meta['global_step']:06d}.json"
            if cached(path,identity) is None:
                result=evaluate_model(model,batch,refs,scaler,sg,bank,torch.tensor(plan['permutations'][0]['donors']),a.device,32);result.pop('all_predictions')
                result.update(seed=a.seed,arm=point['arm'],lambda_group=point['lambda_group'],step=meta['global_step'],bank=0,K=32,aggregation='bank0 only',checkpoint_origin=point['origin']);save_cached(path,identity,result)
                print('CURVE',path.name,result['aggregate']['conditional']['loss'],result['aggregate']['marginal']['loss'],flush=True)
            rows.append(dict(path=str(path),sha256=sha256_file(path)));del model,meta;torch.cuda.empty_cache()
    print('COMPLETE',complete(out/'completed.json',rows),flush=True)


if __name__=='__main__':main()
