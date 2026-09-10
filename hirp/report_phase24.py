"""Paired bank Monte Carlo uncertainty, not population/seed replication."""
import csv,json,math
from pathlib import Path
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('runs/phase24/evaluation_v1')


def csvwrite(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),lineterminator='\n');w.writeheader();w.writerows(rows)


def stats(x):
    x=np.array(x,dtype=float);se=float(x.std(ddof=1)/np.sqrt(len(x)));mu=float(x.mean());half=float(t.ppf(.975,len(x)-1))*se
    return dict(mean=mu,MC_SE=se,MC_CI_low=mu-half,MC_CI_high=mu+half,banks=len(x))


def main():
    out=ROOT/'analysis_v1';out.mkdir(exist_ok=False);rows=[];cache={}
    assert (ROOT/'mc/completed.json').exists()
    for path in sorted((ROOT/'mc').glob('seed*.json')):
        r=json.loads(path.read_text())['result'];a=r['aggregate'];row=dict(seed=r['seed'],arm=r['arm'],lambda_group=r['lambda_group'],bank=r['bank'],conditional_ES=a['conditional']['loss'],conditional_cross=a['conditional']['cross_distance'],conditional_self=a['conditional']['self_distance'],group_ES=a['marginal']['loss'],group_cross=a['marginal']['cross'],group_self=a['marginal']['self'],within=a['spread']['within_input_descriptor_self'],between=a['spread']['between_input_descriptor_distance'])
        rows.append(row);cache[(r['seed'],r['arm'],r['lambda_group'],r['bank'])]=row
    assert len(rows)==120
    csvwrite(out/'per_bank.csv',rows);summary=[];halves=[]
    keys=sorted({(r['seed'],r['arm'],r['lambda_group']) for r in rows})
    for seed,arm,l in keys:
        rr=[cache[seed,arm,l,b] for b in range(24001,24009)];s=dict(seed=seed,arm=arm,lambda_group=l)
        for metric in ('conditional_ES','conditional_cross','conditional_self','group_ES','group_cross','group_self','within','between'):
            for key,value in stats([r[metric] for r in rr]).items():s[metric+'_'+key]=value
        delta=[r['conditional_ES']-1.01*cache[seed,'C0',0.,r['bank']]['conditional_ES'] for r in rr];margin=stats(delta);s.update({'margin_1pct_'+key:value for key,value in margin.items()});s['screen_status']='satisfies' if margin['MC_CI_high']<0 else ('does_not_satisfy' if margin['MC_CI_low']>0 else 'interval_crosses_threshold');summary.append(s)
        for name,part in [('banks1_4',rr[:4]),('banks5_8',rr[4:])]:halves.append(dict(seed=seed,arm=arm,lambda_group=l,half=name,conditional_ES=float(np.mean([r['conditional_ES'] for r in part])),group_ES=float(np.mean([r['group_ES'] for r in part]))))
    csvwrite(out/'per_seed_8bank.csv',summary);csvwrite(out/'predetermined_halves.csv',halves)
    paired=[];contrasts=[];half_contrasts=[]
    for seed in (123,42,2026):
        for l in (.03,.1):
            for other in ('C0','C1'):
                rr=[]
                for b in range(24001,24009):
                    a=cache[seed,'C2',l,b];z=cache[seed,other,0. if other=='C0' else l,b];r=dict(seed=seed,lambda_group=l,contrast='C2-'+other,bank=b,conditional_delta=a['conditional_ES']-z['conditional_ES'],group_delta=a['group_ES']-z['group_ES']);paired.append(r);rr.append(r)
                for metric in ('conditional_delta','group_delta'):
                    contrasts.append(dict(seed=seed,lambda_group=l,contrast='C2-'+other,metric=metric,**stats([r[metric] for r in rr])))
                    for name,part in [('banks1_4',rr[:4]),('banks5_8',rr[4:])]:half_contrasts.append(dict(seed=seed,lambda_group=l,contrast='C2-'+other,metric=metric,half=name,mean=float(np.mean([r[metric] for r in part]))))
    csvwrite(out/'paired_per_bank.csv',paired);csvwrite(out/'paired_MC_intervals.csv',contrasts);csvwrite(out/'paired_half_diagnostics.csv',half_contrasts)
    means=[]
    for arm,l in sorted({(r['arm'],r['lambda_group']) for r in summary}):
        rr=[r for r in summary if (r['arm'],r['lambda_group'])==(arm,l)];s=dict(arm=arm,lambda_group=l,training_seeds=3)
        for m in ('conditional_ES','group_ES'):
            v=[r[m+'_mean'] for r in rr];s[m+'_mean']=float(np.mean(v));s[m+'_seed_std']=float(np.std(v,ddof=1))
        means.append(s)
    csvwrite(out/'training_seed_mean_std.csv',means)
    history=list(csv.DictReader(Path('runs/phase23/tradeoff_v1/analysis_v1/per_seed_frontier.csv').open()))
    csvwrite(out/'historical_frontier_all_lambdas.csv',[dict(record_type='preserved Phase23 historical K32/K64, not Phase24 new measurement',**r) for r in history])
    multi=[];single=[];borders=[]
    assert (ROOT/'task_multitarget/completed_HiRP.json').exists()
    for path in sorted((ROOT/'task_multitarget').glob('*.json')):
        if path.name.startswith('completed'):continue
        r=json.loads(path.read_text())['result'];base=dict(seed=r['seed'],arm=r['arm'],lambda_group=r['lambda_group'])
        multi.append(dict(**base,**{k:v for k,v in r['multi_target'].items() if not k.endswith('_time')}));single.append(dict(**base,**{k:v for k,v in r['single_paired'].items() if not k.endswith('_time')}))
        for i,row in enumerate(r['exports']):borders.append(dict(**base,input_index=i,**{k:v for k,v in row['checks'].items() if k!='boundaries'}))
    csvwrite(out/'task_multi_target.csv',multi);csvwrite(out/'task_single_paired.csv',single);csvwrite(out/'boundary_diagnostics.csv',borders)
    f,axes=plt.subplots(1,3,figsize=(15,4.5))
    for ax,seed in zip(axes,(123,42,2026)):
        for arm,color in [('C0','black'),('C1','tab:blue'),('C2','tab:orange')]:
            rr=sorted([r for r in summary if r['seed']==seed and r['arm']==arm],key=lambda r:r['lambda_group']);ax.plot([r['conditional_ES_mean'] for r in rr],[r['group_ES_mean'] for r in rr],'o-',color=color,label=arm)
            for r in rr:
                ax.errorbar(r['conditional_ES_mean'],r['group_ES_mean'],xerr=t.ppf(.975,7)*r['conditional_ES_MC_SE'],yerr=t.ppf(.975,7)*r['group_ES_MC_SE'],color=color,capsize=3);ax.annotate(str(r['lambda_group']),(r['conditional_ES_mean'],r['group_ES_mean']),xytext=(4,4),textcoords='offset points')
        ax.set(title=f'Seed {seed}, 8 independent banks, K64',xlabel='Conditional ES (lower)',ylabel='Group ES (lower)');ax.legend();ax.grid(alpha=.2)
    f.suptitle('Development80: 95% bank Monte Carlo intervals; not population or training-seed uncertainty');f.tight_layout();f.savefig(out/'frontier_MC.png',dpi=170);plt.close(f)
    f,axes=plt.subplots(1,2,figsize=(12,4.5))
    labels=[('C0',0.),('C1',.03),('C1',.1),('C2',.03),('C2',.1),('MAM',None)]
    for ax,data,title in zip(axes,(single,multi),('Single paired','Official-rule multi-target')):
        vals=[];errors=[];names=[]
        for arm,l in labels:
            v=[r['FRC'] for r in data if r['arm']==arm and r['lambda_group']==l]
            if not v:continue
            vals.append(float(np.mean(v)));errors.append(float(np.std(v,ddof=1)) if len(v)>1 else 0.);names.append(arm+' '+(str(l) if l is not None else 'native (n=1)'))
        ax.bar(names,vals,yerr=errors,capsize=3,color=['gray','steelblue','steelblue','orange','orange','purple'][:len(vals)]);ax.set(title=title,ylabel='FRC, native sum over K10 candidates');ax.tick_params(axis='x',rotation=30)
    f.suptitle('Development80 full sequences; HiRP mean ± training-seed SD (n=3), MAM one archive seed')
    f.tight_layout();f.savefig(out/'task_FRC.png',dpi=170);plt.close(f)
    (out/'summary.json').write_text(json.dumps(dict(per_seed=summary,mean_std=means,contrasts=contrasts,multi_target=multi,single_paired=single,scope='development only; banks are MC repetitions, not training seeds'),indent=2)+'\n')
    print('ANALYSIS COMPLETE',flush=True)


if __name__=='__main__':main()
