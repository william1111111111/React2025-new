"""Fixed, GT-free teacher combinations for a prespecified diversity band.

VAL64 selects global recipes; VAL507 is reported separately from full VAL571.
No per-context GT-dependent selection or new teacher optimization is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from coreset_reactor.evaluate import official_prediction
from coreset_reactor.fast_frc import candidate_frc
from coreset_reactor.oracle_select import Candidate, SharedQualityPool
from coreset_reactor.teacher_cache import sha256_file
from coreset_reactor.paired_data import PairedReactionDataset
from coreset_reactor.diagnose_teacher_target_fit import aligned_targets
from coreset_reactor.learned_common_va_experiment import paired_summary

ROOT = Path(__file__).resolve().parents[1]
KEYS = ('FRC', 'exact_FRD', 'FRDiv', 'FRVar', 'raw_FRC')


def recipe_grid():
    result = []
    for step in (600, 1200):
        for alpha in (0., .05, .10, .15, .20, .25):
            result.append(dict(name=f'T1_{step}_blend{alpha:.2f}', step=step,
                               kind='blend', alpha=alpha))
        for rank in range(1, 10):
            result.append(dict(name=f'T1_{step}_replace_q{rank}', step=step,
                               kind='replace', rank=rank))
    return result


def combine(diverse, quality, recipe):
    """Inputs are unrounded probabilities; preserve q0 from the diverse bank."""
    if diverse.shape != quality.shape or diverse.shape[0] != 10:
        raise ValueError('Expected two matched [10,T,25] banks')
    result = diverse.clone()
    if recipe['kind'] == 'blend':
        alpha = recipe['alpha']
        if not 0 <= alpha <= 1: raise ValueError('Invalid interpolation')
        result[1:] = (1-alpha)*diverse[1:]+alpha*quality[1:]
    elif recipe['kind'] == 'replace':
        rank = recipe['rank']
        if not 1 <= rank < 10: raise ValueError('Keep paired anchor q0')
        result[rank] = quality[rank]
    else:
        raise ValueError(recipe['kind'])
    return official_prediction(result)


def paths(convergence):
    return {label: convergence/'eval'/f'{arm}_s{step:06d}_val571_full.json'
            for label, arm, step in [('T1_600','T1',600), ('T1_1200','T1',1200), ('T0_2400','T0',2400)]}


def load_banks(convergence):
    sources = paths(convergence)
    metrics = {k: json.loads(p.read_text()) for k,p in sources.items()}
    banks = {k: torch.load(p.with_suffix('.predictions.pt'), map_location='cpu') for k,p in sources.items()}
    ids = banks['T1_600']['clip_ids']
    assert len(ids) == 571
    for key in banks:
        assert banks[key]['clip_ids'] == ids
        assert [r['target_cache_key'] for r in metrics[key]['rows']] == [r['target_cache_key'] for r in metrics['T1_600']['rows']]
    return banks, metrics


def target_at(cache, row):
    key = row['target_cache_key']
    state = torch.load(cache/f'{key}.pt', map_location='cpu')
    assert state['fingerprint'] == key
    return state['targets']


def basic(pred):
    from framework.metrics import compute_s_mse, compute_FRVar
    return {'FRDiv': float(compute_s_mse([pred])), 'FRVar': float(compute_FRVar([pred]))}


def pilot(args):
    args.root.mkdir(parents=True, exist_ok=False)
    grid = recipe_grid()
    indices = sorted(np.random.default_rng(1234).choice(571, 64, replace=False).tolist())
    provenance = {'git_sha': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                  'source_sha256': sha256_file(Path(__file__)), 'grid': grid, 'pilot_indices': indices,
                  'goal': {'FRC_min': 1.10, 'FRDiv_min': .17, 'FRDiv_max': .20},
                  'selection_rule': 'Among pilot goal-pass arms, highest FRC per family plus highest remaining FRC; at most3; no full-VAL reselection',
                  'convergence': str(args.convergence.resolve()), 'cache': str(args.cache.resolve()),
                  'data_root': str(args.data_root.resolve()), 'asset_root': str(args.asset_root.resolve()),
                  'scope': 'Exploratory VAL tuning; same all-session processed GT; final ten; paired q0 retained; no new training'}
    (args.root/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    banks, metrics = load_banks(args.convergence)
    provenance['input_hashes'] = {k: {'metrics': sha256_file(p), 'predictions': sha256_file(p.with_suffix('.predictions.pt')),
                                    'checkpoints': metrics[k]['checkpoint_hashes']} for k,p in paths(args.convergence).items()}
    (args.root/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    rows=[]; start=time.monotonic()
    for index in indices:
        original = metrics['T1_600']['rows'][index]
        target = target_at(args.cache, original)
        quality = banks['T0_2400']['soft_predictions'][index]
        outputs = [combine(banks[f'T1_{r["step"]}']['soft_predictions'][index], quality, r) for r in grid]
        frc = candidate_frc(target.numpy(), torch.cat(outputs).numpy()).reshape(len(grid),10).sum(1)
        values = {r['name']: {'FRC':float(f), **basic(p)} for r,p,f in zip(grid,outputs,frc)}
        np.testing.assert_allclose(values['T1_600_blend0.00']['FRC'],sum(original['frc']),atol=1e-7)
        rows.append({'index':index, 'clip_id':original['clip_id'], 'metrics':values})
        if len(rows)%8==0: print(json.dumps({'pilot_completed':len(rows),'elapsed_s':time.monotonic()-start}),flush=True)
    means = {r['name']:{k:float(np.mean([x['metrics'][r['name']][k] for x in rows])) for k in ('FRC','FRDiv','FRVar')} for r in grid}
    eligible = sorted([r for r in grid if means[r['name']]['FRC']>=1.10 and .17<=means[r['name']]['FRDiv']<=.20],
                      key=lambda r:(-means[r['name']]['FRC'],r['name']))
    selected=[]
    for kind in ('replace','blend'):
        match=next((r for r in eligible if r['kind']==kind),None)
        if match: selected.append(match)
    selected += [r for r in eligible if r not in selected][:3-len(selected)]
    if not selected:
        raise RuntimeError('No pilot arm met target; retain pilot rows and inspect before expanding search')
    summary={'means':means,'selected':selected,'rows':rows,'seconds':time.monotonic()-start}
    (args.root/'pilot.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'selected':[{**r,**means[r['name']]} for r in selected]},indent=2),flush=True)


def full(args):
    prov=json.loads((args.root/'provenance.json').read_text())
    if sha256_file(Path(__file__)) != prov['source_sha256']: raise ValueError('Source drift')
    selected=json.loads((args.root/'pilot.json').read_text())['selected']
    banks,meta=load_banks(Path(prov['convergence']))
    data=PairedReactionDataset(Path(prov['data_root']), 'val',750,crop_mode='center',
                               normalization_dir=Path(prov['asset_root'])/'external/FaceVerse')
    facial=Path(prov['data_root'])/'val/facial-attributes'
    sessions={}
    for p in sorted((facial/'listener').glob('*/*.npy')): sessions.setdefault(p.parent.name,[]).append(p)
    ledger=args.root/'rows'; ledger.mkdir(exist_ok=True)
    recipes=[dict(name='baseline',step=600,kind='blend',alpha=0.),*selected]
    pool=SharedQualityPool(args.workers,40,max_targets=max(r['gt_count'] for r in meta['T1_600']['rows']))
    indices=list(range(571))[args.shard_index::args.shard_count]
    started=time.monotonic()
    try:
        for completed,index in enumerate(indices,1):
            dest=ledger/f'{index:06d}.json'
            if dest.exists(): raise FileExistsError(dest)
            original=meta['T1_600']['rows'][index]; sample=data[index]
            assert sample['clip_id']==original['clip_id']
            target=target_at(Path(prov['cache']), original)
            quality=banks['T0_2400']['soft_predictions'][index]
            tick=time.perf_counter()
            outputs=[combine(banks[f'T1_{r["step"]}']['soft_predictions'][index],quality,r) for r in recipes]
            composition_s=(time.perf_counter()-tick)/len(recipes)
            candidates=[]; lookup={}; indices_by_arm=[]
            for recipe,output in zip(recipes,outputs):
                arm_indices=[]
                for slot,pred in enumerate(output):
                    digest=hashlib.sha256(pred.numpy().tobytes()).digest()
                    if digest not in lookup:
                        lookup[digest]=len(candidates)
                        candidates.append(Candidate(pred,recipe['name'],'teacher',[(recipe['name'],slot)]))
                    arm_indices.append(lookup[digest])
                indices_by_arm.append(arm_indices)
            frc,frd=pool.score(candidates,target)
            np.testing.assert_allclose(frc[indices_by_arm[0]], original['frc'],atol=1e-7,rtol=1e-6)
            np.testing.assert_allclose(frd[indices_by_arm[0]].sum(),original['exact_FRD'],atol=1e-7,rtol=1e-8)
            paired=(facial/sample['clip_id'].replace('speaker/','listener/',1)).with_suffix('.npy')
            raw_paths=[paired,*[p for p in sessions[sample['session_id']] if p!=paired]]
            raw=aligned_targets([torch.from_numpy(np.load(p,allow_pickle=False).astype(np.float32)) for p in raw_paths],
                                int(sample['source_lengths']),int(sample['crop_start']))
            raw_frc=candidate_frc(raw.numpy(),torch.stack([c.trajectory for c in candidates]).numpy())
            values={r['name']:{'FRC':float(frc[j].sum()),'exact_FRD':float(frd[j].sum()),'raw_FRC':float(raw_frc[j].sum()),
                               'frc_per_slot':frc[j].tolist(),'frd_per_slot':frd[j].tolist(),**basic(p)}
                    for r,p,j in zip(recipes,outputs,indices_by_arm)}
            row={'index':index,'clip_id':sample['clip_id'],'session_id':sample['session_id'],
                 'target_cache_key':original['target_cache_key'],'metrics':values,'unique_scored_predictions':len(candidates),
                 'composition_seconds':composition_s}
            dest.write_text(json.dumps(row,indent=2,allow_nan=False)+'\n')
            if completed%10==0 or completed==len(indices):
                progress={'completed':completed,'total':len(indices),'elapsed_s':time.monotonic()-started}
                (args.root/f'shard{args.shard_index}.progress.json').write_text(json.dumps(progress)+'\n')
                print(json.dumps(progress),flush=True)
    finally: pool.close()


def summarize(args):
    prov=json.loads((args.root/'provenance.json').read_text()); pilot=json.loads((args.root/'pilot.json').read_text())
    rows=[json.loads((args.root/f'rows/{i:06d}.json').read_text()) for i in range(571)]
    arms=['baseline',*[r['name'] for r in pilot['selected']]]; pilot_set=set(prov['pilot_indices'])
    subsets={'VAL64':[r for r in rows if r['index'] in pilot_set], 'VAL507':[r for r in rows if r['index'] not in pilot_set], 'VAL571':rows}
    summaries={}
    for scope,subset in subsets.items():
        means={arm:{key:float(np.mean([r['metrics'][arm][key] for r in subset])) for key in KEYS} for arm in arms}
        delta={arm:{key:paired_summary([r['metrics'][arm][key]-r['metrics']['baseline'][key] for r in subset],
                                      [r['session_id'] for r in subset]) for key in KEYS} for arm in arms[1:]}
        summaries[scope]={'contexts':len(subset),'metrics':means,'deltas':delta,
            'goal_pass':{arm:means[arm]['FRC']>=1.10 and .17<=means[arm]['FRDiv']<=.20 for arm in arms},
            'FRD_nonworse':{arm:means[arm]['exact_FRD']<=means['baseline']['exact_FRD'] for arm in arms}}
    per_session={s:{arm:{k:float(np.mean([r['metrics'][arm][k] for r in rows if r['session_id']==s])) for k in KEYS} for arm in arms}
                 for s in sorted({r['session_id'] for r in rows})}
    source_meta={k:json.loads(p.read_text()) for k,p in paths(Path(prov['convergence'])).items()}
    deploy=[]
    for recipe in pilot['selected']:
        step=recipe['step']; assignments=[]
        for rank in range(10):
            bank='T0_2400' if recipe['kind']=='replace' and rank==recipe['rank'] else f'T1_{step}'
            assignments.append({'slot':rank,'bank':bank,'checkpoint':source_meta[bank]['checkpoint'][rank],
                                'sha256':source_meta[bank]['checkpoint_hashes'][rank]})
        if recipe['kind']=='blend':
            assignments += [{'slot':rank,'bank':'T0_2400','checkpoint':source_meta['T0_2400']['checkpoint'][rank],
                             'sha256':source_meta['T0_2400']['checkpoint_hashes'][rank]} for rank in range(1,10)]
        deploy.append({**recipe,'assignments':assignments,'model_count':len(assignments),
                       'parameters':len(assignments)*3720129, 'GT_required_at_inference':False})
    result={'subsets':summaries,'per_session':per_session,'recipes':deploy,'provenance':prov,
            'composition_seconds_mean':float(np.mean([r['composition_seconds'] for r in rows]))}
    (args.root/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (args.root/'inference_recipes.json').write_text(json.dumps(deploy,indent=2)+'\n')
    lines=['# Teacher quality at FRDiv 0.17–0.20','',
           'REACT2025; deterministic all-session opposite-role GT, existing processed-GT cache, official AU rounding, exact rolling DTW. FRC is the established sum over ten outputs. This is not official sampled-ten-GT evaluation.', '',
           'A fixed 30-arm VAL64 grid chooses at most three global recipes. Parameters are fixed across contexts; inference uses speaker-conditioned teacher outputs only. q0 remains the paired anchor. No new teacher training. VAL507 excludes pilot contexts but shares sessions and has been used in earlier studies: exploratory evidence, not an independent TEST claim.', '',
           '| Scope | Recipe | FRC ↑ | exact FRD ↓ | FRDiv | FRVar | Raw-GT FRC ↑ | Goal pass |',
           '|---|---|---:|---:|---:|---:|---:|---|']
    for scope,s in summaries.items():
        for arm,m in s['metrics'].items():
            lines.append(f'| {scope} | {arm} | '+' | '.join(f'{m[k]:.6f}' for k in KEYS)+f" | {s['goal_pass'][arm]} |")
    lines += ['', '## Frozen global recipes', '',
              'Blend: for q1–q9, y=(1-alpha)*T1 + alpha*T0_2400 before AU rounding; q0 uses T1 unchanged. Replacement: use the matching-rank T0_2400 model at one fixed slot and T1 models elsewhere. Different banks have different fixed training targets, so blend is output fusion, not weight interpolation.']
    for r in deploy:
        lines += ['',f"- `{r['name']}`: {r['model_count']} models, {r['parameters']:,} parameters; recipe/checkpoint hashes in the local inference manifest."]
    lines += ['', '## Distribution and uncertainty', '',
              '| Full VAL arm / metric | Paired delta | Session-bootstrap 95% CI |', '|---|---:|---|']
    for arm in arms[1:]:
        for key in ('FRC','exact_FRD','FRDiv','raw_FRC'):
            d=summaries['VAL571']['deltas'][arm][key]; lo,hi=d['session_bootstrap_CI95']
            lines.append(f"| {arm} / {key} | {d['mean']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines += ['', 'Full context metrics, per-session summaries, quantiles and input hashes are retained locally. Baseline FRC/FRD is reproduced for every context before scoring recipes.', '',
              'Budget: zero new training updates / temporal-loss calls; internal model count 10 for replacement or 19 for blend; final output count10. Replacement has the same parameter count and forward count as baseline. Blend costs nine extra teacher forwards. Runtime selector overhead is zero (fixed recipe).', '',
              f"Cached-output composition: {1000*result['composition_seconds_mean']:.3f} ms/recipe/context on CPU, excluding model inference and evaluation. No new end-to-end GPU latency benchmark is claimed.", '',
              f"Base git SHA `{prov['git_sha']}`; experiment source SHA256 `{prov['source_sha256']}`. The full-VAL result is validation-selected and cannot establish independent generalization."]
    args.report.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'full':summaries['VAL571']['metrics'],'goal_pass':summaries['VAL571']['goal_pass']},indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('action',choices=('pilot','full','summarize'))
    p.add_argument('--root',type=Path,required=True)
    for name in ('convergence','cache','data-root','asset-root','report'): p.add_argument('--'+name,type=Path)
    p.add_argument('--workers',type=int,default=10); p.add_argument('--shard-index',type=int,default=0); p.add_argument('--shard-count',type=int,default=1)
    args=p.parse_args(); torch.set_num_threads(4)
    {'pilot':pilot,'full':full,'summarize':summarize}[args.action](args)


if __name__=='__main__': main()
