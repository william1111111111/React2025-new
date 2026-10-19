"""Frozen-student input dependence diagnostics; no training or TEST scoring.

All-session scores are not assumed to identify conditioning. Raw paired CCC,
prediction change and an explicitly non-official motion-energy lag diagnostic
are reported separately. Reference selection for retrieval is TRAIN-only.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from coreset_reactor.teacher_cache import sha256_file, atomic_torch_save
from coreset_reactor.fast_frc import pairwise_frc
from coreset_reactor.evaluate import official_prediction
from coreset_reactor.paired_data import PairedReactionDataset
from coreset_reactor.oracle_select import Candidate, SharedQualityPool
from regnn.conditional_data import ConditionalReactionDataset, relative_time_resample_reaction, crop_and_pad
from regnn.expert_student import ExpertConditionalREGNN, ExpertStudentConfig
from regnn.conditional_model import ConditionalREGNNConfig

ROOT=Path(__file__).resolve().parents[1]
KEYS=('speaker_audio','speaker_emotion','speaker_3dmm')
ALL_ARMS=('baseline','audio_train_mean','visual_train_mean','temporal_blocks',
          'global_train_templates','train_nearest_set','same_session_donor','cross_session_donor')
FULL_ARMS=ALL_ARMS[:6]


def read(p):return json.loads(Path(p).read_text())


def write(p,x):
    p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False,default=str)+'\n');t.replace(p)


def descriptors(sample,length):
    return [torch.cat((sample[k][:length].mean(0),sample[k][:length].std(0,unbiased=False))) for k in KEYS]


def nearest_train(groups,query,scales):
    # Equal modality weights, not one weight per feature: audio does not win
    # simply by having 768 channels. No session ID or VAL listener information.
    distance=sum((((g-q)/s)**2).mean(1) for g,q,s in zip(groups,query,scales))/3
    return int(distance.argmin()),float(distance.min())


def farthest_templates(values,count=10):
    x=np.asarray(values,dtype=np.float64)
    if len(x)<count:raise ValueError('Not enough TRAIN templates')
    x=(x-x.mean(0))/np.maximum(x.std(0),.03)
    first=int(np.square(x).mean(1).argmin());chosen=[first]
    nearest=np.square(x-x[first]).mean(1)
    for _ in range(1,count):
        nearest[chosen]=-np.inf;j=int(nearest.argmax());chosen.append(j)
        nearest=np.minimum(nearest,np.square(x-x[j]).mean(1))
    return chosen


def donor_maps(sessions,lengths,seed=20260930):
    # Exact length750 cohort only: never stretch/crop donor predictions or
    # silently include identity substitutions in the intervention estimate.
    eligible=[i for i,l in enumerate(lengths) if l==750]
    groups={s:[i for i in eligible if sessions[i]==s] for s in sorted(set(sessions))}
    eligible=[i for i in eligible if len(groups[sessions[i]])>=2]
    rng=np.random.default_rng(seed);within={}
    for group in groups.values():
        if len(group)<2:continue
        order=rng.permutation(group).tolist()
        within.update({i:order[(p+1)%len(order)] for p,i in enumerate(order)})
    if not eligible:raise ValueError('No equal-length cohort')
    cost=rng.random((len(eligible),len(eligible)))
    forbidden=np.array([[sessions[i]==sessions[j] for j in eligible] for i in eligible])
    cost[forbidden]=1e9
    rows,cols=linear_sum_assignment(cost)
    if (cost[rows,cols]>=1e9).any():raise ValueError('No cross-session bijection exists')
    across={eligible[i]:eligible[j] for i,j in zip(rows,cols)}
    return within,across,eligible


def block_order(length,clip_id,seed=20260930):
    blocks=[np.arange(i,min(i+25,length)) for i in range(0,length,25)]
    g=np.random.default_rng(int(hashlib.sha256(f'{seed}|{clip_id}'.encode()).hexdigest()[:16],16))
    p=g.permutation(len(blocks))
    if len(blocks)>1 and np.array_equal(p,np.arange(len(blocks))):p=np.roll(p,1)
    return torch.as_tensor(np.concatenate([blocks[i] for i in p]),dtype=torch.long)


def motion_proxy(prediction,speaker):
    # Descriptive, not official TLCC/FRSyn or a causal appropriateness score.
    # Mean across ten trajectories, temporal first-difference magnitude.
    p=prediction.float().diff(dim=1).abs().mean((0,2)).numpy()
    s=speaker.float().diff(dim=0).abs().mean(1).numpy()
    values=[];usable=False
    for lag in range(-49,50):
        x,y=(p[lag:],s[:-lag]) if lag>0 else ((p[:lag],s[-lag:]) if lag<0 else (p,s))
        if len(x)<3 or x.std()<1e-10 or y.std()<1e-10:values.append(0.)
        else:values.append(float(np.corrcoef(x,y)[0,1]));usable=True
    peak=int(np.argmax(values))
    return {'zero_lag_corr':values[49],'peak_corr':values[peak],
            'peak_abs_lag_frames':abs(peak-49) if usable else None,'usable':usable}


def split_audit(data):
    """Filename/content overlap is not proof of participant disjointness."""
    catalog={s:sorted((data/s/'facial-attributes').glob('*/*/*.npy')) for s in ('train','val','test')}
    hashes={}
    for split in ('train','val'):
        hashes[split]={p.relative_to(data/split/'facial-attributes').as_posix():sha256_file(p) for p in catalog[split]}
    train_set=set(hashes['train'].values());val_set=set(hashes['val'].values())
    train_names={p.stem for p in catalog['train']};val_names={p.stem for p in catalog['val']}
    test_equal={}
    for modality in ('facial-attributes','audio-features','coefficients'):
        files={s:{p.relative_to(data/s/modality).as_posix():p for p in (data/s/modality).glob('*/*/*.npy')}
               for s in ('val','test')}
        common=set(files['val'])&set(files['test'])
        count=sum(sha256_file(files['val'][key])==sha256_file(files['test'][key]) for key in sorted(common))
        test_equal[modality]={'val':len(files['val']),'test':len(files['test']),
                             'same_paths':len(common),'same_content':count}
    identical=all(v['val']==v['test']==v['same_paths']==v['same_content'] and v['val']>0 for v in test_equal.values())
    return {'split_sizes':{s:len(p) for s,p in catalog.items()},
            'train_val_shared_facial_content_hashes':len(train_set&val_set),
            'train_val_shared_filename_stems':sorted(train_names&val_names),
            'train_val_shared_relative_paths':sorted(set(hashes['train'])&set(hashes['val'])),
            'feature_hashes':hashes,
            'participant_disjointness':'NOT VERIFIED: no authoritative participant identity mapping used',
            'root_csv_warning':'Root train.csv/val.csv begin with UDIVA paths, unlike MARS directory features; not used as split evidence',
            'test_independence':'BLOCKED: VAL and TEST feature content identical' if identical else 'NOT VERIFIED: need participant/recording provenance',
            'test_val_sha256_equality_audit':test_equal,
            'interpretation':'Shared content can include trivial constant sequences; inspect before treating as leakage. No TEST scoring.'}


def prepare(a):
    base=read(a.baseline);bank=torch.load(a.baseline.with_suffix('.predictions.pt'),map_location='cpu')
    if base['metrics']['contexts']!=571 or bank['clip_ids']!=[r['clip_id'] for r in base['rows']]:
        raise ValueError('Expected full aligned VAL571 baseline')
    checkpoint=Path(base['checkpoint'])
    if sha256_file(checkpoint)!=base['checkpoint_sha256'] or bank['checkpoint_sha256']!=base['checkpoint_sha256']:
        raise ValueError('Frozen baseline changed')
    cache_path=a.assets/'coreset_reactor/reports/mode_reference_student_seed1/train_cache/manifest.json'
    cache=read(cache_path);meta=[]
    if cache['contexts']!=3320 or cache['split']!='train':raise ValueError('Expected all TRAIN views')
    for shard in cache['shards']:
        values=torch.load(cache_path.parent/shard,map_location='cpu')
        meta.extend({k:row[k] for k in ('sample_index','sample_id','view_id','start','length','view_epoch')} for row in values)
    dataset=ConditionalReactionDataset(str(a.data),'train',clip_length=750,training=True,target_mode='paired',seed=1)
    dataset.set_epoch(0)
    train_groups=[[] for _ in KEYS];train_ids=[]
    for i,row in enumerate(meta):
        sample=dataset[(row['sample_index'],row['view_id'])]
        if (sample['clip_id'],int(sample['start']),int(sample['length']))!=(row['sample_id'],row['start'],row['length']):
            raise ValueError('TRAIN source-view mismatch')
        for group,d in zip(train_groups,descriptors(sample,row['length'])):group.append(d)
        train_ids.append(sample['clip_id']+'.npy')
        if (i+1)%250==0:write(a.root/'prepare_progress.json',{'train_sources':i+1,'total':len(meta)})
    groups=[torch.stack(g) for g in train_groups]
    scales=[g.std(0,unbiased=False).clamp_min(1e-4) for g in groups]
    # Time-weighting is deliberately not used: each fixed TRAIN view contributes equally.
    means=[g[:,:g.shape[1]//2].mean(0) for g in groups]
    paths=sorted((a.data/'train/facial-attributes').glob('*/*/*.npy'));gt_desc=[]
    raw_hashes={}
    for p in paths:
        y=dataset._load_tensor(p);gt_desc.append(torch.cat((y.mean(0),y.std(0,unbiased=False))).numpy())
        raw_hashes[p.relative_to(a.data/'train/facial-attributes').as_posix()]=sha256_file(p)
    selected=farthest_templates(gt_desc)
    fixed_path=a.assets/'coreset_reactor/reports/maxfrdiv_teacher_pilot_600_v3/manifests/T1/rank_00.json'
    fixed=read(fixed_path)
    if fixed['split']!='train' or set(train_ids)!=set(fixed['mapping']):raise ValueError('Retrieval identities mismatch')
    sessions=[r['clip_id'].split('/')[1] for r in base['rows']]
    lengths=[len(x[0]) for x in bank['soft_predictions']]
    within,across,cohort=donor_maps(sessions,lengths)
    plan={'seed':20260930,'full_contexts':571,'donor_cohort_count':len(cohort),'donor_cohort':cohort,
          'within':within,'across':across,'lengths':lengths,
          'global_templates':[p.relative_to(a.data/'train/facial-attributes').as_posix() for p in [paths[i] for i in selected]],
          'template_selection':'TRAIN mean/std50D, standardized, farthest-min; first nearest TRAIN mean; no VAL selection',
          'retrieval':'1NN TRAIN speaker mean/std audio-emotion-3DMM, TRAIN-scaled, equal modality weight; return its fixed10 GT',
          'source_cache_hash':cache['cache_manifest_hash'],'fixed_mapping_hash':sha256_file(fixed_path),
          'train_gt_hashes':raw_hashes}
    write(a.root/'frozen_plan.json',plan)
    atomic_torch_save({'groups':groups,'scales':scales,'means':means,'train_ids':train_ids,
                       'mapping':fixed['mapping'],'train_meta':meta},a.root/'train_retrieval.pt')
    return plan


def worker(a):
    from framework.metrics import compute_FRVar,compute_s_mse
    torch.set_num_threads(2);torch.cuda.set_device(0);device=torch.device('cuda:0')
    plan=read(a.root/'frozen_plan.json');base=read(a.baseline)
    bank=torch.load(a.baseline.with_suffix('.predictions.pt'),map_location='cpu')
    train=torch.load(a.root/'train_retrieval.pt',map_location='cpu')
    state=torch.load(base['checkpoint'],map_location='cpu');c=state['student_config']
    model=ExpertConditionalREGNN(ExpertStudentConfig(backbone=ConditionalREGNNConfig(**c['backbone']),
        num_experts=c['num_experts'],expert_embedding_dim=c['expert_embedding_dim'])).to(device).eval()
    model.load_state_dict(state['state_dict'],strict=True)
    data=PairedReactionDataset(a.data,'val',750,crop_mode='center',normalization_dir=a.assets/'mam_reactor/external/FaceVerse')
    target_cache=a.assets/'coreset_reactor/reports/fixed_teacher_distillation/val571_postprocessed_gt_cache'
    raw={};rows=[];pool=SharedQualityPool(8,10,750,max_targets=max(r['gt_count'] for r in base['rows']))
    def target(relative):
        if relative not in raw:
            raw[relative]=torch.from_numpy(np.load(a.data/'train/facial-attributes'/relative,allow_pickle=False).astype(np.float32))
        return raw[relative]
    def generate(conditions,length):
        with torch.inference_mode():
            return model.generate(*[v[None].to(device) for v in conditions],
                                  torch.tensor([length],device=device))['prediction'][0].cpu()
    started=time.time()
    try:
        for i in range(a.shard,len(data),3):
            sample=data[i];length=int(sample['source_lengths']);cid=sample['clip_id']
            if cid!=base['rows'][i]['clip_id'] or length!=plan['lengths'][i]:raise ValueError('VAL view mismatch')
            conditions=[sample[k][:length] for k in KEYS]
            original=bank['soft_predictions'][i]
            if i==a.shard:
                torch.testing.assert_close(generate(conditions,length),original,rtol=1e-5,atol=1e-6)
            nearest,distance=nearest_train(train['groups'],descriptors(sample,length),train['scales'])
            retrieved=train['train_ids'][nearest];rmeta=train['train_meta'][nearest]
            gt_ids=train['mapping'][retrieved]
            retrieval_targets=[]
            for k,rid in enumerate(gt_ids):
                value=target(rid)
                # Rank0 is the retrieved TRAIN view's native paired trajectory;
                # all other ranks use relative time just as fixed-target teachers.
                if k==0:value=value[rmeta['start']:rmeta['start']+rmeta['length']]
                retrieval_targets.append(relative_time_resample_reaction(value,length))
            p=block_order(length,cid)
            outputs={'baseline':original,
                'audio_train_mean':generate([train['means'][0].expand(length,-1),*conditions[1:]],length),
                'visual_train_mean':generate([conditions[0],train['means'][1].expand(length,-1),train['means'][2].expand(length,-1)],length),
                'temporal_blocks':generate([v[p] for v in conditions],length),
                'global_train_templates':torch.stack([relative_time_resample_reaction(target(r),length) for r in plan['global_templates']]),
                'train_nearest_set':torch.stack(retrieval_targets)}
            if str(i) in plan['within']:
                outputs['same_session_donor']=bank['soft_predictions'][plan['within'][str(i)]]
                outputs['cross_session_donor']=bank['soft_predictions'][plan['across'][str(i)]]
            payload=torch.load(target_cache/(base['rows'][i]['target_cache_key']+'.pt'),map_location='cpu')
            if payload['fingerprint']!=base['rows'][i]['target_cache_key']:raise ValueError('GT cache mismatch')
            gt=payload['targets'];pair_length=int(sample['pair_lengths']);paired=sample['paired_target'][:pair_length]
            native_proxy=motion_proxy(paired[None],conditions[1][:pair_length])
            for arm,soft in outputs.items():
                pred=official_prediction(soft)
                if soft.shape!=(10,length,25):raise ValueError('Wrong prediction shape')
                frc=pairwise_frc(gt.numpy(),pred.numpy())
                _,frd=pool.score([Candidate(pred[k],f'{arm}_{k}',arm,[(arm,k)]) for k in range(10)],gt)
                paired_ccc=pairwise_frc(paired[None].numpy(),pred[:,:pair_length].numpy())[:,0]
                row={'arm':arm,'context_index':i,'clip_id':cid,'session':sample['session_id'],'length':length,
                     'target_cache_key':base['rows'][i]['target_cache_key'],'gt_count':len(gt),
                     'FRC':float(frc.max(1).sum()),'exact_FRD':float(frd.sum()),'FRDiv':float(compute_s_mse([pred])),
                     'FRVar':float(compute_FRVar([pred])),'soft_FRDiv':float(compute_s_mse([soft])),
                     'native_paired_q0_CCC':float(paired_ccc[0]),'native_paired_mean_CCC':float(paired_ccc.mean()),
                     'native_paired_best_CCC':float(paired_ccc.max()),'native_paired_length':pair_length,
                     'soft_prediction_change_MSE':float((soft-original).square().mean()),
                     'motion_energy_proxy':motion_proxy(soft,conditions[1]),'native_GT_motion_energy_proxy':native_proxy,
                     'retrieved_train_id':retrieved if arm=='train_nearest_set' else None,
                     'retrieval_distance':distance if arm=='train_nearest_set' else None}
                if arm=='baseline':
                    for key in ('exact_FRD','FRDiv'):
                        np.testing.assert_allclose(row[key],base['rows'][i][key],atol=1e-5,rtol=1e-5)
                    np.testing.assert_allclose(row['FRC'],sum(base['rows'][i]['frc']),atol=1e-6,rtol=1e-5)
                rows.append(row)
            write(a.root/f'shard{a.shard}.progress.json',{'contexts':1+(i-a.shard)//3,'total':len(range(a.shard,len(data),3)),
                'last_clip':cid,'elapsed_seconds':time.time()-started})
            write(a.root/f'shard{a.shard}.rows.json',rows)
    finally:pool.close()
    write(a.root/f'shard{a.shard}.complete.json',{'complete':True,'rows':len(rows),'seconds':time.time()-started})


def aggregate(a):
    plan=read(a.root/'frozen_plan.json');rows=sum([read(a.root/f'shard{i}.rows.json') for i in range(3)],[])
    reference={r['context_index']:r for r in rows if r['arm']=='baseline'}
    keys=('FRC','exact_FRD','FRDiv','FRVar','soft_FRDiv','native_paired_q0_CCC',
          'native_paired_mean_CCC','native_paired_best_CCC','soft_prediction_change_MSE')
    result={}
    for arm in ALL_ARMS:
        items=sorted([r for r in rows if r['arm']==arm],key=lambda r:r['context_index'])
        expected=set(range(571)) if arm in FULL_ARMS else set(plan['donor_cohort'])
        if {r['context_index'] for r in items}!=expected or len(items)!=len(expected):raise ValueError('Incomplete/duplicate arm cohort')
        values=np.array([[r[k] for k in keys] for r in items]);b=np.array([[reference[r['context_index']][k] for k in keys] for r in items])
        sessions=np.array([r['session'] for r in items]);sg=sorted(set(sessions))
        sums=np.stack([(values-b)[sessions==s].sum(0) for s in sg]);counts=np.array([(sessions==s).sum() for s in sg])
        draws=np.random.default_rng(20260930).integers(0,len(sg),(4000,len(sg)))
        boot=sums[draws].sum(1)/counts[draws].sum(1)[:,None]
        result[arm]={'contexts':len(items),'metrics':dict(zip(keys,values.mean(0).tolist())),
            'matched_baseline_metrics':dict(zip(keys,b.mean(0).tolist())),
            'delta_vs_same_context_baseline':{k:{'mean':float((values-b)[:,j].mean()),
                'session_bootstrap_ci95':np.quantile(boot[:,j],[.025,.975]).tolist()} for j,k in enumerate(keys)},
            'motion_energy_proxy_means':{k:(float(np.mean([r['motion_energy_proxy'][k] for r in items if r['motion_energy_proxy']['usable']]))
                if any(r['motion_energy_proxy']['usable'] for r in items) else None)
                for k in ('zero_lag_corr','peak_corr','peak_abs_lag_frames')},
            'motion_proxy_usable_fraction':float(np.mean([r['motion_energy_proxy']['usable'] for r in items]))}
    write(a.root/'summary.json',{'dataset':'REACT2025','arms':result,
        'scope':'Frozen checkpoint perturbation and TRAIN-template/retrieval controls; not retrained modality ablations',
        'invariance_warning':'Equal-length same-session output permutations may leave all-session aggregate quality invariant; not evidence of ignoring input.',
        'native_paired_warning':'One actual reaction is a diagnostic, not the only appropriate reaction.',
        'test_not_scored':True,'limitations':'Single model/seed, VAL previously inspected, OOD perturbations; no human preference or causal claim.'})
    lines=['# Frozen student input dependence audit','',
           'REACT2025 full VAL571. TRAIN-only templates/retrieval. Frozen student, no new training.',
           'All-session AU-rounded FRC/exact FRD. Native paired CCC is a separate diagnostic, not official FRC.',
           'Donor controls use only the equal-length750 cohort; baseline comparisons use exactly those contexts.','',
           '| Arm | N | FRC | exact FRD | FRDiv | Native paired mean CCC | Change MSE |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for arm,r in result.items():
        m=r['metrics'];lines.append(f"| {arm} | {r['contexts']} | {m['FRC']:.6f} | {m['exact_FRD']:.6f} | {m['FRDiv']:.6f} | {m['native_paired_mean_CCC']:.6f} | {m['soft_prediction_change_MSE']:.6f} |")
    lines+=['','Same-session aggregate invariance does not establish input neglect. Inspect paired/context changes and cross-session controls.',
        'Motion-energy lag is descriptive, not official TLCC/FRSyn and not proof of appropriate responding.',
        'Mean replacement and temporal permutation can be out of distribution; these are interventions, not retrained architecture ablations.',
        'No TEST metrics were computed. See split audit before any independent-test claim.',
        'Session bootstrap intervals in summary.json are conditional on one training run; no human-rating results claimed.','']
    (a.root/'INPUT_DEPENDENCE_AUDIT.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--assets',type=Path,default=Path('/public/zhaowenjie/react2025_new'))
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--gpus',type=int,nargs=3,default=[1,5,6])
    p.add_argument('--shard',type=int,choices=range(3))
    a=p.parse_args();a.data=a.assets/'data'
    if a.shard is not None:worker(a);return
    if len(set(a.gpus))!=3:raise ValueError('Three distinct GPUs')
    a.root.mkdir(parents=True,exist_ok=False);started=time.time()
    def status(stage,**kw):write(a.root/'status.json',{'stage':stage,'pid':os.getpid(),'started':started,'updated':time.time(),**kw})
    try:
        status('preparing_train_only_controls')
        sources={}
        for folder in ('coreset_reactor','mam_reactor/regnn','mam_reactor/framework/metrics'):
            for src in (ROOT/folder).rglob('*.py'):
                rel=src.relative_to(ROOT);dst=a.root/'source_snapshot'/rel
                dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);sources[str(rel)]=sha256_file(src)
        write(a.root/'experiment.json',{'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'source_hashes':sources,'baseline_sha256':sha256_file(a.baseline),'baseline':str(a.baseline),
            'arms':ALL_ARMS,'seed':20260930,'gpus':a.gpus,'test_policy':'No TEST model evaluation or tuning',
            'predeclared_interpretation':'Input perturbation sensitivity is necessary diagnostic evidence, not proof of semantic appropriateness.',
            'primary_metrics':['all-session FRC/exact FRD/FRDiv','native paired mean CCC','prediction change MSE'],
            'no_new_training':True})
        write(a.root/'split_audit.json',split_audit(a.data))
        prepare(a);status('evaluating_full_val_three_shards')
        def launch(shard,gpu):
            cmd=[sys.executable,'-u','-m','coreset_reactor.input_dependency_audit','--root',str(a.root),
                 '--assets',str(a.assets),'--baseline',str(a.baseline),'--shard',str(shard)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
            with (a.root/f'shard{shard}.log').open('x') as f:
                subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(launch,s,g) for s,g in enumerate(a.gpus)]
            for f in futures:f.result()
        aggregate(a);status('complete',wall_seconds=time.time()-started)
    except Exception as error:status('failed',error=repr(error));raise


if __name__=='__main__':main()
