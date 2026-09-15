import json
from pathlib import Path
import numpy as np
from reward_policy.common import read,write,sha
OLD=Path('runs/reaction_flow/reward_policy_v1');OUT=Path('runs/reaction_flow/reward_signal_v2')
GROUPS={'AU':np.arange(96)%24<15,'VA':(np.arange(96)%24>=15)&(np.arange(96)%24<17),'expression':np.arange(96)%24>=17}

def quant(x):
    x=np.asarray(x).reshape(-1);x=x[np.isfinite(x)]
    return {str(q):float(np.quantile(x,q)) if len(x) else None for q in (0,.1,.25,.5,.75,.9,.99,1)}

def load_records(folder):return [read(p) for p in sorted((OLD/folder).glob('*.json'))]

def pair_data(row,reference,scales,cohort):
    s={k:np.asarray(v) if isinstance(v,list) else v for k,v in row['scores'].items()};ref=reference['scores'];rec=row['record'];source=cohort[rec['source_indices'][0]];start=750*rec['block_indices'][0];n=min(750,source['source_length']-start);ns=[min(n,source['paired_length']-start)]+[n]*3
    mask=np.stack([(np.arange(96)//24<min(4,n,nn)) for nn in ns]);sq=((s['phi'][:,None]-s['target_phi'][None])/np.array(scales['phi']))**2*mask[None];d2=sq.sum(-1)/mask.sum(-1)[None]
    cf=np.quantile(np.max(ref['ccc'],1),.1);dc=np.quantile(np.min(ref['dtw'],1),.9);cg=s['ccc']>=cf;dg=s['dtw']<=dc;motion=s['domain_valid']&(s['speed']<=scales['speed_ceiling'])&(s['accel']<=scales['accel_ceiling']);eligible=cg&dg&motion[:,None]
    return dict(scores=s,mask=mask,sq=sq,d2=d2,cg=cg,dg=dg,motion=motion,eligible=eligible,n=n)

def analyze(records,refs,scales,cohort,h2=1.):
    arrays={k:[] for k in ('d2','qualified_d2','exponent','similarity','qualified_similarity','gains','coord_contribution','qualified_coord_share','within_quality_variance','source_quality_offset')};rows=[];column_std=[];rates=[]
    for row,ref in zip(records,refs):
        x=pair_data(row,ref,scales,cohort);d=x['d2'];e=x['eligible'];sim=np.exp(-d/(2*h2));u=sim*e;k=len(u);cov=u.max(0).mean();gains=np.array([cov-np.delete(u,i,axis=0).max(0).mean() for i in range(k)])
        rows.append(dict(step=row['record']['step'],frames=x['n'],CCC_pass=float(x['cg'].mean()),DTW_pass=float(x['dg'].mean()),separate_best_pass=float((x['cg'].any(1)&x['dg'].any(1)).mean()),same_pair_pass=float((x['cg']&x['dg']).mean()),domain_pass=float(x['scores']['domain_valid'].mean()),motion_pass=float(x['motion'].mean()),eligible_pair_rate=float(e.mean()),target_columns_eligible=int(e.any(0).sum()),coverage=float(cov),nontrivial_gain_fraction=float((gains>1e-6).mean()),zero_affinity_fraction=float((sim==0).mean()),tiny_positive_fraction=float(((sim>0)&(sim<1e-12)).mean())))
        arrays['d2'].extend(d.flatten());arrays['qualified_d2'].extend(d[e]);arrays['exponent'].extend((-d/(2*h2)).flatten());arrays['similarity'].extend(sim.flatten());arrays['qualified_similarity'].extend(sim[e]);arrays['gains'].extend(gains)
        arrays['coord_contribution'].append(x['sq'].sum((0,1)));share=x['sq']/x['sq'].sum(-1,keepdims=True).clip(1e-300);arrays['qualified_coord_share'].extend(share.max(-1)[e])
        c=x['scores']['ccc'].max(1);dd=x['scores']['dtw'].min(1);q=c/scales['c']-dd/scales['d'];arrays['within_quality_variance'].append(q.var());arrays['source_quality_offset'].append(q.mean())
        for j in range(e.shape[1]):
            if e[:,j].sum()>1:column_std.append(float(sim[e[:,j],j].std()))
    contrib=np.sum(arrays.pop('coord_contribution'),axis=0);contrib=contrib/contrib.sum();variance=dict(mean_within_source=float(np.mean(arrays['within_quality_variance'])),variance_source_means=float(np.var(arrays['source_quality_offset'])))
    return dict(episodes=len(rows),h2=h2,rows=rows,quantiles={k:quant(v) for k,v in arrays.items()},mean_rates={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ('step','frames')},coordinate_contribution=contrib.tolist(),group_contribution={g:float(contrib[m].sum()) for g,m in GROUPS.items()},eligible_column_affinity_std=quant(column_std),near_constant_column_fraction=float(np.mean(np.array(column_std)<1e-4)) if column_std else 1.,variance=variance)

def main():
    cohort=read('runs/reaction_flow/mode_supervision_v1/cohort.json');scales=read(OLD/'scales.json');cal=load_records('calibration');reach=load_records('reachability');rr=load_records('reachability_reference');refs=load_records('reference_table');fit=[i for i in range(64) if i//20%2==0];hold=[i for i in range(64) if i not in fit]
    fitrows=[cal[i] for i in fit];holdrows=[cal[i] for i in hold];dist=np.concatenate([pair_data(r,r,scales,cohort)['d2'][pair_data(r,r,scales,cohort)['eligible']] for r in fitrows]);assert len(dist)
    h2=max(1e-6,float(np.median(dist)/(2*np.log(2))));after=analyze(holdrows,holdrows,scales,cohort,h2)
    blocked=after['quantiles']['qualified_coord_share']['0.5']>.5 or after['near_constant_column_fraction']>=.9 or after['mean_rates']['eligible_pair_rate']<.001
    write(OUT/'BANDWIDTH.json',dict(version='old-geometry-bandwidth-v2',h2=h2,fit_indices=fit,hold_indices=hold,fit_qualified_pairs=len(dist),median_fit_D2=float(np.median(dist)),bandwidth_only_blocked=bool(blocked),heldout=after))
    results={name:analyze(rs,rs if name!='reachability' else rr,scales,cohort) for name,rs in [('calibration',cal),('reachability',reach),('reference_table',refs)]}
    write(OUT/'EXISTING_DATA_AUDIT.json',dict(cached=results,missing='per-candidate action reward matrices and output trajectories were not cached during old 500-step updates; reference tables are independent parent draws, not policy action draws; per-group native motion traces absent',old_thresholds=dict(speed=scales['speed_ceiling'],accel=scales['accel_ceiling']),bandwidth_only_blocked=bool(blocked)))
    print('h2',h2,'blocked',blocked,'median dominant coordinate share',after['quantiles']['qualified_coord_share']['0.5'],'eligible',after['mean_rates']['eligible_pair_rate'],'constant',after['near_constant_column_fraction'],flush=True)
if __name__=='__main__':main()
