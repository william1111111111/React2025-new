"""Versioned Development-80 evaluator and complete cache identity validation."""
import argparse,itertools,json,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate
from .phase23_cache import eval_identity,cached,save_cached,complete,implementation_identity
from .phase23_aux import auxiliary
from .evaluate_phase22 import evaluate_model,extra_shuffles
from .evaluate_phase2 import reference_cache_and_baseline
from .phase22 import load_checkpoint
from .train_phase22 import write
from .train_phase21 import configure
from .group_features import DescriptorScaler
from .session_data import SessionPopulation
from .train_phase2 import description
from .group_scores import session_mixture_score
from .phase15_audit import sha256_file,canonical_hash

OLD=Path('runs/phase22/replicated_v1')


def points(root,seed):
    result=[]
    for row in json.loads((root/'reused_checkpoints.json').read_text()):
        if row['seed']==seed:result.append(dict(**row,origin='phase22_reused'))
    for weight in (.03,.3):
        run=root/f'lambda_{weight:g}'/f'seed_{seed}'
        for arm in ('C1','C2'):
            path=run/arm/'attempt_000/summary.json'
            if path.exists():
                s=json.loads(path.read_text())
                if s['completed'] and s['actual_optimizer_steps']==s['requested_steps']==s['training_rows']==2000:
                    result.append(dict(seed=seed,arm=arm,lambda_group=weight,checkpoint=s['checkpoints'][-1],run=str(run),origin='phase23_new'))
    return result


def auxiliary_metrics(pred,batch,pool):
    generated=auxiliary(pred,batch['source_lengths']);paired=auxiliary(batch['paired_target'][:,None],batch['pair_lengths'])
    refs={}
    for session in sorted(set(batch['session_id'])):
        refs[session]=[]
        for rid in pool.references[session]:
            x=pool.load_reference(rid);refs[session].extend(auxiliary(x['reaction'][None,None],x['length'][None]))
    rows=[]
    for session in sorted(refs):
        ids=[i for i,s in enumerate(batch['session_id']) if s==session];r={}
        for key in ('covariance','autocorrelation'):
            g=np.mean([generated[i][key] for i in ids],axis=0);q=np.mean([x[key] for x in refs[session]],axis=0)
            r[key+'_population_RMS']=float(np.sqrt(np.mean((g-q)**2)))
            r[key+'_paired_RMS']=float(np.mean([np.sqrt(np.mean((np.array(generated[i][key])-np.array(paired[i][key]))**2)) for i in ids]))
        rows.append(dict(session_id=session,**r))
    return dict(per_session=rows,aggregate={key:float(np.mean([r[key] for r in rows])) for key in rows[0] if key!='session_id'},
                per_input_generated=generated,per_input_paired=paired)


def subset_scores(pred,batch,scaler,refs):
    with torch.no_grad():phi,valid=description(pred,batch['source_lengths'],scaler)
    sessions=sorted(set(batch['session_id']));rows=[]
    for rank,subset in enumerate(itertools.combinations(range(4),2)):
        values=[]
        for session in sessions:
            ids=[i for i,s in enumerate(batch['session_id']) if s==session];chosen=[ids[j] for j in subset]
            d=session_mixture_score(phi[chosen],valid[chosen],*refs[session]);values.append(dict(session=session,loss=float(d['loss'])))
        rows.append(dict(subset_rank=rank,within_session_positions=subset,macro_ES=float(np.mean([v['loss'] for v in values])),per_session=values))
    return rows


def evaluate(root,seed,device,name='evaluation_v1',only_reused=False):
    if '/' in name or name in ('.','..'):raise ValueError('evaluation-name must be one new directory name')
    plan_path=OLD/'evaluation_plan/manifest.json';plan=json.loads(plan_path.read_text())
    selected=points(root,seed)
    if only_reused:selected=[p for p in selected if p['origin']=='phase22_reused']
    elif len(selected)!=7:raise ValueError('full frontier requires all 7 pre-fixed points for this seed')
    pool=SessionPopulation(Path('data'),'val',128);batch=default_collate([pool.dataset[i] for i in plan['indices']])
    assert batch['clip_id']==plan['clip_ids']
    crops=[dict(clip_id=batch['clip_id'][i],source_length=int(batch['source_lengths'][i]),pair_length=int(batch['pair_lengths'][i]),crop_start=int(batch['crop_start'][i])) for i in range(80)]
    out=root/name/f'seed_{seed}';out.mkdir(parents=True,exist_ok=True)
    scaler_path=Path('runs/phase2/descriptor_stats.json');scaler=DescriptorScaler.load(scaler_path);sg=DescriptorScaler.load(scaler_path).to(device)
    refs,real=reference_cache_and_baseline(pool,scaler)
    versions=dict(source=implementation_identity(),torch=torch.__version__,precision='FP32, TF32 disabled',auxiliary='raw covariance/acf1,4,16 v1')
    completed=[]
    for point in selected:
        label=f"{point['arm']}_lambda{point['lambda_group']:g}"
        model,metadata=load_checkpoint(point['checkpoint']['path'],device)
        if sha256_file(point['checkpoint']['path'])!=point['checkpoint']['sha256']:raise ValueError('checkpoint changed')
        if metadata['global_step']!=2000 or metadata['arm']!=point['arm'] or metadata['train_config']['init_seed']!=seed:raise ValueError('checkpoint semantics mismatch')
        for bank_index,bank_info in enumerate(plan['banks']):
            bank=torch.from_numpy(np.load(bank_info['path']))
            for k in (32,64):
                identity=eval_identity(point['checkpoint']['path'],metadata,plan_path,plan,bank_info,k,scaler_path,crops,versions)
                path=out/f'{label}_bank{bank_index}_K{k}.json';previous=cached(path,identity)
                if previous is None:
                    result=evaluate_model(model,batch,refs,scaler,sg,bank[:,:k],torch.tensor(plan['permutations'][0]['donors']),device,k)
                    pred=result.pop('all_predictions');result['permutations']=extra_shuffles(pred,batch,plan['permutations'])
                    result.update(seed=seed,arm=point['arm'],lambda_group=point['lambda_group'],K=k,bank=bank_index,origin=point['origin'])
                    if k==32:
                        result['auxiliary']=auxiliary_metrics(pred,batch,pool)
                        result['source_subset_sensitivity']=subset_scores(pred,batch,scaler,refs)
                    save_cached(path,identity,result)
                    print('EVAL',seed,label,bank_index,k,result['aggregate']['conditional']['loss'],result['aggregate']['marginal']['loss'],flush=True)
                else:print('REUSED',path.name,flush=True)
                completed.append(dict(path=str(path),identity_sha256=canonical_hash(identity),result_sha256=json.loads(path.read_text())['result_sha256']))
        del model,metadata
        if device.startswith('cuda'):torch.cuda.empty_cache()
    completion=out/('completed_reused.json' if only_reused else 'completed.json')
    print('COMPLETE',complete(completion,completed),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('runs/phase23/tradeoff_v1'));p.add_argument('--seed',type=int,default=123)
    p.add_argument('--device',default='cuda:4');p.add_argument('--evaluation-name',default='evaluation_v1');p.add_argument('--only-reused',action='store_true')
    a=p.parse_args();configure();evaluate(a.root,a.seed,a.device,a.evaluation_name,a.only_reused)


if __name__=='__main__':main()
