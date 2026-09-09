"""Descriptive finite-grid report tables; no interpolated/mixed checkpoints."""
import csv,json,itertools
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .evaluate_phase23 import points
from .phase15_audit import sha256_file,canonical_hash
from .train_phase22 import write

ROOT=Path('runs/phase23/tradeoff_v1');SEEDS=(123,42,2026)


def table(path,rows):
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,lineterminator='\n',fieldnames=list(dict.fromkeys(key for row in rows for key in row)));writer.writeheader();writer.writerows(rows)


def flatten(d,prefix=''):
    out={}
    for key,value in d.items():
        name=prefix+key
        if isinstance(value,dict):out.update(flatten(value,name+'_'))
        elif isinstance(value,(int,float)) and not isinstance(value,bool):out[name]=value
    return out


def interval(values,seed=23031):
    values=np.asarray(values);rng=np.random.default_rng(seed);means=values[rng.integers(len(values),size=(10000,len(values)))].mean(1)
    return dict(mean=float(values.mean()),session_bootstrap_low=float(np.quantile(means,.025)),session_bootstrap_high=float(np.quantile(means,.975)))


def main():
    root=ROOT;out=root/'analysis_v1';out.mkdir(exist_ok=False);fig=out/'figures';fig.mkdir();raw=[];cache={};per_input=[];per_session=[];subsets=[]
    for seed in SEEDS:
        assert (root/'evaluation_v1'/f'seed_{seed}'/'completed.json').exists()
        for path in sorted((root/'evaluation_v1'/f'seed_{seed}').glob('*_K*.json')):
            envelope=json.loads(path.read_text());r=envelope['result'];assert canonical_hash(r)==envelope['result_sha256']
            key=(seed,r['arm'],r['lambda_group'],r['K'],r['bank']);cache[key]=r
            base=dict(seed=seed,arm=r['arm'],lambda_group=r['lambda_group'],K=r['K'],bank=r['bank'],step=2000,origin=r['origin'])
            row=dict(**base,**flatten(r['aggregate']));row['shuffle_gap_3perm']=float(np.mean([x['mean_gap'] for x in r['permutations']]))
            if 'auxiliary' in r:row.update(flatten(r['auxiliary']['aggregate'],'aux_'))
            raw.append(row)
            for x in r['per_input']:per_input.append(dict(**base,clip_id=x['clip_id'],session_id=x['session_id'],**flatten({k:x[k] for k in ('conditional','conditional_correct_common','conditional_shuffled','diversity')})))
            for perm in r['permutations']:
                for x in perm['per_session']:per_session.append(dict(**base,permutation_seed=perm['seed'],**x))
            for x in r.get('source_subset_sensitivity',[]):subsets.append(dict(**base,subset_rank=x['subset_rank'],positions=str(x['within_session_positions']),group_ES=x['macro_ES']))
    assert len(raw)==84
    table(out/'per_bank_metrics.csv',raw);table(out/'per_input_metrics.csv',per_input);table(out/'per_session_shuffle.csv',per_session);table(out/'source_subset_sensitivity.csv',subsets)
    grouped={}
    for r in raw:grouped.setdefault((r['seed'],r['arm'],r['lambda_group'],r['K']),[]).append(r)
    numeric=[k for k in raw[0] if k not in ('seed','arm','lambda_group','K','bank','step','origin')]
    numeric=list(dict.fromkeys(numeric+[k for r in raw for k in r if k.startswith('aux_')]))
    rows=[]
    for (seed,arm,weight,k),rr in grouped.items():
        assert {r['bank'] for r in rr}=={0,1}
        row=dict(seed=seed,arm=arm,lambda_group=weight,K=k,step=2000,banks='mean(bank0,bank1)',origin=rr[0]['origin'])
        row.update({key:float(np.mean([r[key] for r in rr])) for key in numeric if all(key in r for r in rr)});rows.append(row)
    for row in rows:
        c0=next(r for r in rows if r['seed']==row['seed'] and r['arm']=='C0' and r['K']==row['K'])
        row['conditional_delta_C0']=row['conditional_loss']-c0['conditional_loss'];row['conditional_relative_delta_C0']=row['conditional_delta_C0']/c0['conditional_loss'];row['engineering_screen_1pct']=row['conditional_relative_delta_C0']<=.01
    table(out/'per_seed_frontier.csv',rows)
    means=[]
    for (arm,weight,k) in sorted({(r['arm'],r['lambda_group'],r['K']) for r in rows}):
        rr=[r for r in rows if (r['arm'],r['lambda_group'],r['K'])==(arm,weight,k)];assert len(rr)==3
        row=dict(arm=arm,lambda_group=weight,K=k,n_train_seeds=3,banks='mean(bank0,bank1)',screen_pass_seeds=sum(r['engineering_screen_1pct'] for r in rr))
        for key in numeric+['conditional_delta_C0','conditional_relative_delta_C0']:
            if all(key in r for r in rr):
                v=[r[key] for r in rr];row[key+'_mean']=float(np.mean(v));row[key+'_std']=float(np.std(v,ddof=1));row[key+'_variance']=float(np.var(v,ddof=1))
        means.append(row)
    table(out/'frontier_mean_std.csv',means)
    differences=[]
    for seed,k,weight in itertools.product(SEEDS,(32,64),(.03,.1,.3)):
        for comparator in ('C1','C0'):
            lhs=[cache[(seed,'C2',weight,k,b)] for b in (0,1)];rhs=[cache[(seed,comparator,weight if comparator=='C1' else 0.,k,b)] for b in (0,1)]
            for metric in ('conditional','marginal','shuffle_gap'):
                if metric=='shuffle_gap':
                    a=np.mean([[x['gap'] for x in p['per_session']] for r in lhs for p in r['permutations']],axis=0);b=np.mean([[x['gap'] for x in p['per_session']] for r in rhs for p in r['permutations']],axis=0)
                else:
                    a=np.mean([[x[metric]['loss'] for x in r['per_session']] for r in lhs],axis=0);b=np.mean([[x[metric]['loss'] for x in r['per_session']] for r in rhs],axis=0)
                differences.append(dict(seed=seed,K=k,lambda_group=weight,contrast='C2-'+comparator,metric=metric,**interval(a-b),unit='20 session labels; unknown cross-session participant dependence',scope='conditional on trained seed, fixed Development80, 2 banks, 3 fixed permutations'))
    table(out/'paired_session_contrasts.csv',differences)
    # Direct per-seed conditional degradation uncertainty for every point, not equivalence.
    quality=[]
    for seed,arm,weight in {(r['seed'],r['arm'],r['lambda_group']) for r in rows if r['K']==32}:
        v=np.mean([[x['conditional']['loss'] for x in cache[(seed,arm,weight,32,b)]['per_session']] for b in (0,1)],axis=0)
        c=np.mean([[x['conditional']['loss'] for x in cache[(seed,'C0',0.,32,b)]['per_session']] for b in (0,1)],axis=0)
        quality.append(dict(seed=seed,arm=arm,lambda_group=weight,**interval(v-1.01*c),quantity='conditional_ES - 1.01*C0_ES',interpretation='engineering screen; bootstrap conditional on fixed model/list, not statistical equivalence'))
    table(out/'quality_screen_uncertainty.csv',quality)
    # Locked discrete development choice: best mean group ES among points passing every seed screen.
    selection=[]
    for arm in ('C1','C2'):
        candidates=[r for r in means if r['arm']==arm and r['K']==32 and r['screen_pass_seeds']==3]
        chosen=min(candidates,key=lambda r:r['marginal_loss_mean']) if candidates else None
        selection.append(dict(arm=arm,selected_lambda=None if chosen is None else chosen['lambda_group'],rule='minimum mean group ES among discrete points with relative-C0 conditional loss <=1% in each of 3 seeds',confirmation_executed=False,statistical_equivalence_claimed=False))
    write(out/'development_choice.json',dict(choices=selection,selection_stage='after Phase23 Development80 sweep, before any future confirmation; confirmation pending'))
    # Full 7-point frontiers, no interpolation of latent checkpoints.
    colors={'C0':'black','C1':'tab:blue','C2':'tab:orange'}
    f,axes=plt.subplots(1,3,figsize=(15,4.5))
    for ax,seed in zip(axes,SEEDS):
        for arm in ('C0','C1','C2'):
            rr=sorted([r for r in rows if r['seed']==seed and r['arm']==arm and r['K']==32],key=lambda r:r['lambda_group'])
            ax.plot([r['conditional_loss'] for r in rr],[r['marginal_loss'] for r in rr],'o-',color=colors[arm],label=arm)
            for r in rr:ax.annotate(str(r['lambda_group']),(r['conditional_loss'],r['marginal_loss']),xytext=(3,5),textcoords='offset points',fontsize=8)
        c0=next(r for r in rows if r['seed']==seed and r['arm']=='C0' and r['K']==32);ax.axvline(c0['conditional_loss']*1.01,color='gray',ls='--',label='C0 +1% screen')
        ax.set(title=f'Seed {seed}: K32, mean of banks 0+1',xlabel='Conditional trajectory ES (lower)',ylabel='Session descriptor ES (lower)');ax.legend(fontsize=8);ax.grid(alpha=.2)
    f.tight_layout();f.savefig(fig/'frontier_per_seed.png',dpi=180);f.savefig(fig/'frontier_per_seed.pdf');plt.close(f)
    f,ax=plt.subplots(figsize=(7,5))
    for arm in ('C0','C1','C2'):
        rr=sorted([r for r in means if r['arm']==arm and r['K']==32],key=lambda r:r['lambda_group'])
        ax.errorbar([r['conditional_loss_mean'] for r in rr],[r['marginal_loss_mean'] for r in rr],xerr=[r['conditional_loss_std'] for r in rr],yerr=[r['marginal_loss_std'] for r in rr],fmt='o-',capsize=3,color=colors[arm],label=arm)
        for r in rr:ax.annotate(str(r['lambda_group']),(r['conditional_loss_mean'],r['marginal_loss_mean']),xytext=(4,6),textcoords='offset points')
    ax.set(title='Development-80: K32, two banks, mean ± training-seed SD (n=3)',xlabel='Conditional trajectory ES (lower)',ylabel='Session descriptor ES (lower)');ax.legend();ax.grid(alpha=.2);f.tight_layout();f.savefig(fig/'frontier_mean_std.png',dpi=180);plt.close(f)
    curves=[]
    for seed in SEEDS:
        assert (root/'learning_curves_bank0'/f'seed_{seed}'/'completed.json').exists()
        for path in (root/'learning_curves_bank0'/f'seed_{seed}').glob('*_step*.json'):
            r=json.loads(path.read_text())['result'];curves.append(dict(seed=seed,arm=r['arm'],lambda_group=r['lambda_group'],step=r['step'],K=32,bank=0,aggregation='bank0 only',conditional_ES=r['aggregate']['conditional']['loss'],group_ES=r['aggregate']['marginal']['loss']))
    table(out/'learning_curves_bank0.csv',curves)
    for seed in SEEDS:
        f,axes=plt.subplots(1,2,figsize=(12,4))
        for arm in ('C0','C1','C2'):
            for weight in ([0.] if arm=='C0' else [.03,.1,.3]):
                rr=sorted([r for r in curves if (r['seed'],r['arm'],r['lambda_group'])==(seed,arm,weight)],key=lambda r:r['step'])
                for ax,key in zip(axes,['conditional_ES','group_ES']):ax.plot([r['step'] for r in rr],[r[key] for r in rr],marker='.',label=f'{arm} lambda={weight:g}');ax.set(xlabel='Actual optimizer steps',ylabel=key);ax.grid(alpha=.2)
        axes[1].legend(fontsize=8);f.suptitle(f'Seed {seed}, Development-80 K32, BANK0 ONLY (not two-bank final mean)');f.tight_layout();f.savefig(fig/f'learning_curves_seed{seed}_bank0.png',dpi=170);plt.close(f)
    task=[]
    for path in (root/'task_development_full').glob('seed*.json'):
        r=json.loads(path.read_text())['result'];task.append(dict(seed=r['seed'],arm=r['arm'],lambda_group=r['lambda_group'],K=10,protocol=r['protocol'],frames=r['total_source_frames'],**r['metrics']))
    table(out/'task_development_full.csv',task)
    training=[]
    for seed in SEEDS:
        for point in points(root,seed):
            s=json.loads((Path(point['run'])/point['arm']/'attempt_000/summary.json').read_text());training.append(dict(seed=seed,arm=point['arm'],lambda_group=point['lambda_group'],origin=point['origin'],**{key:s[key] for key in ('requested_steps','actual_optimizer_steps','training_rows','source_occurrences','unique_sources','source_population','reference_exposures','parameter_count','optimizer_compute_seconds','peak_allocated_bytes','initialization_hash')}))
    table(out/'training_budget.csv',training)
    write(out/'summary.json',dict(final_rows=rows,mean_std=means,new_training_count=12,reused_training_count=9,selection=selection,confirmation='pending; never evaluated'))
    print('REPORT TABLES COMPLETE',out,flush=True)


if __name__=='__main__':main()
