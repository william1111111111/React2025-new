"""The 12-point finite extension; unchanged Phase22 model, objective and trainer."""
import argparse,json,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from .train_phase22 import TrainConfig,prepare,train,RealData,write
from .train_phase21 import configure
from .phase15_audit import canonical_hash


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('runs/phase23/tradeoff_v1'));p.add_argument('--seed',type=int,required=True);p.add_argument('--device',default='cuda:4');a=p.parse_args();configure()
    protocol=json.loads((a.root/'protocol.json').read_text())
    if a.seed not in protocol['seeds']:raise ValueError('seed outside locked matrix')
    old=Path('runs/phase22/replicated_v1')/f'seed_{a.seed}'
    old_manifest=json.loads((old/'manifest.json').read_text());initial=json.loads((old/'C0/attempt_000/summary.json').read_text())['initialization_hash']
    for weight in protocol['new_lambdas']:
        run=a.root/f'lambda_{weight:g}'/f'seed_{a.seed}'
        cfg=TrainConfig(init_seed=a.seed,sampler_seed=a.seed,noise_seed=a.seed,lambda_group=weight)
        manifest=prepare(run,cfg)
        expected=dict(old_manifest['config']);expected['lambda_group']=weight
        assert canonical_hash(asdict(cfg))==canonical_hash(expected)
        for key in ('schedule','split_hash','model_config','scales','output_config','prior_mode'):
            assert canonical_hash(manifest[key])==canonical_hash(old_manifest[key]),key
        records=json.loads((run/'schedule.json').read_text())['records'];noise=torch.from_numpy(np.load(run/'noise.npy'))
        assert np.array_equal(noise.numpy(),np.load(old/'noise.npy'))
        for arm in ('C1','C2'):
            data=RealData(manifest,cfg,arm,a.device)
            result,model,opt=train(run/arm,cfg,arm,records,noise,data,manifest,a.device)
            assert result['initialization_hash']==initial
            assert result['completed'] and result['actual_optimizer_steps']==result['requested_steps']==result['training_rows']==2000
            print('COMPLETE',a.seed,weight,arm,json.dumps(result),flush=True)
            del data,model,opt;torch.cuda.empty_cache()
    write(a.root/f'training_seed{a.seed}_completed.json',dict(seed=a.seed,new_runs=4,actual_optimizer_steps=8000,all_matched_checks=True))


if __name__=='__main__':main()
