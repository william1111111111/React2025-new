"""CPU-only analysis of frozen exports. No generation or official result writes."""
import csv, hashlib, importlib.util, itertools, json, math, subprocess, warnings
from pathlib import Path
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent
ROOT=Path('runs/reaction_flow/shared_noise_v1')
GROUPS={'AU':(0,15),'VA':(15,17),'expression':(17,25),'all':(0,25)}
inputs={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    inputs[str(p)]=sha(p)
    return json.loads(Path(p).read_text())
def array(item):
    p=item['path'];assert sha(p)==item['sha256'],p
    inputs[p]=item['sha256'];a=np.load(p);assert np.isfinite(a).all();return a
paths={'T0':Path('runs/reaction_flow/task_dynamics_v1/task_multitarget/seed123_T0-task_step14000.json'),**{a:ROOT/'task_multitarget'/f'seed123_{a}_step16000.json' for a in ('G0-local','G1-shared')}}
results={m:read(p)['result'] for m,p in paths.items()}
targets=read('runs/phase24/evaluation_v1/processed_targets/completed.json')['result']['files']
frcpath=Path('/home/zhengshiyi/react2025/framework/metrics/FRC.py');inputs[str(frcpath)]=sha(frcpath)
spec=importlib.util.spec_from_file_location('official_frc',frcpath);frc=importlib.util.module_from_spec(spec);spec.loader.exec_module(frc)
# Preserve float32 per-channel mean/var/std reductions, float64 np.cov Pearson,
# clipping/nan_to_num from official 25-channel branch, epsilon 1e-8.
def channels(p,y):
    values=[]
    for c in range(25):
        a=p[:,c];b=y[:,c]
        mp,my=np.mean(a),np.mean(b);vp,vy=np.var(a),np.var(b);sp,sy=np.std(a),np.std(b)
        cv=np.cov(b,a);defined=bool(cv[0,0]>0 and cv[1,1]>0)
        pearson=float(np.clip(cv[0,1]/np.sqrt(cv[0,0])/np.sqrt(cv[1,1]),-1,1)) if defined else float('nan')
        with np.errstate(invalid='ignore',divide='ignore'):
            official_r=frc.corrcoef(b,a)[0,1]
        mean_term=(my-mp)**2;den=vy+vp+mean_term+1e-8
        scale_den=vy+vp+1e-8
        scale=2*sy*sp/scale_den;mean_factor=scale_den/den
        calibration=2*sy*sp/den;ccc=official_r*calibration
        values.append(dict(channel=c,mu_p=float(mp),mu_y=float(my),var_p=float(vp),var_y=float(vy),covariance_population=float(cv[0,1]*(len(a)-1)/len(a)),pearson_r=pearson,pearson_defined=defined,official_r=float(official_r),std_ratio=float(sp/sy) if sy>0 else float('nan'),mean_mismatch=float(mean_term),mean_fraction=float(mean_term/den),scale_factor=float(scale),mean_factor=float(mean_factor),calibration=float(calibration),ccc=float(ccc),GT_zero_mean_ccc=float(official_r*scale),constant_p=bool(np.all(a==a[0])),constant_y=bool(np.all(b==b[0]))))
    return values
records=[];pairs=[];max_matrix_error=0.;max_identity_error=0.;block_rows=[]
for i in range(80):
    y=array(targets[i]);g0=np.asarray(results['G0-local']['per_input'][i]['CCC_candidate_target']).argmax(1)
    for m,res in results.items():
        p=array(res['exports'][i]);assert p.shape==y.shape and p.shape[0]==10 and p.shape[2]==25
        matrix=np.asarray(res['per_input'][i]['CCC_candidate_target']);assert np.isfinite(matrix).all()
        native=matrix.argmax(1)
        cache={}
        for pairing,indices in [('A_native',native),('B_fixed_G0',g0)]:
            for k,j in enumerate(indices):
                key=(k,int(j))
                if key not in cache:
                    vals=channels(p[k],y[j])
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore');official,per_channel=frc.concordance_correlation_coefficient(y[j],p[k])
                    err=max(abs(v['ccc']-c) for v,c in zip(vals,per_channel));max_identity_error=max(max_identity_error,err)
                    score=np.mean([v['ccc'] for v in vals]);max_matrix_error=max(max_matrix_error,abs(score-matrix[k,j]))
                    assert err<1e-12 and abs(score-matrix[k,j])<1e-10,(m,i,k,j,err,score,matrix[k,j])
                    cache[key]=vals
                vals=cache[key]
                pairs.append(dict(pairing=pairing,model=m,source=i,candidate=k,target=int(j),target_switched=int(native[k]!=g0[k]),ccc=np.mean([v['ccc'] for v in vals])))
                records.extend(dict(pairing=pairing,model=m,source=i,candidate=k,target=int(j),**v) for v in vals)
        # Matched adjacent blocks: ddof1 covariance and per-channel Pearson over K.
        for group,(a,b) in GROUPS.items():
            if group=='all':continue
            q=p[...,a:b].astype(np.float64);means=[q[:,s:s+750].mean(1) for s in range(0,p.shape[1],750)]
            for j,(u,v) in enumerate(zip(means,means[1:])):
                cu=u-u.mean(0);cv=v-v.mean(0);cov=(cu*cv).sum(0)/9;vu=(cu**2).sum(0)/9;vv=(cv**2).sum(0)/9
                den=np.sqrt(vu*vv);r=np.divide(cov,den,out=np.full_like(cov,np.nan),where=den>0)
                block_rows.append(dict(model=m,source=i,group=group,left_block=j,right_block=j+1,left_frames=min(750,p.shape[1]-750*j),right_frames=min(750,p.shape[1]-750*(j+1)),candidate_covariance=cov.mean(),correlation=np.nanmean(r),undefined_fraction=np.mean(~np.isfinite(r))))
    print('source',i+1,flush=True)
df=pd.DataFrame(records);df.to_csv(OUT/'ccc_channels.csv',index=False)
pairdf=pd.DataFrame(pairs);pairdf.to_csv(OUT/'ccc_pairs.csv',index=False)
# Source is the aggregation unit: candidate/channel distributions first per source,
# then source means or quantiles across those source means.
source_rows=[]
for (pairing,m,i),sub in df.groupby(['pairing','model','source'],sort=False):
    for group,(a,b) in GROUPS.items():
        q=sub[(sub.channel>=a)&(sub.channel<b)];defined=q[q.pearson_defined]
        r=dict(pairing=pairing,model=m,source=i,group=group)
        for col in ['ccc','GT_zero_mean_ccc','mean_mismatch','mean_fraction','std_ratio','scale_factor','mean_factor','pearson_r','constant_p','constant_y']:
            r[col]=q[col].mean()
        r['pearson_undefined_fraction']=1-len(defined)/len(q)
        for col in ['pearson_r','std_ratio']:
            for quant in [.1,.5,.9]:r[f'{col}_within_q{int(quant*100)}']=q[col].quantile(quant)
        source_rows.append(r)
sources=pd.DataFrame(source_rows);sources.to_csv(OUT/'ccc_per_source.csv',index=False)
summary=sources.groupby(['pairing','model','group']).mean(numeric_only=True).drop(columns='source').reset_index()
for col in ['pearson_r','std_ratio','mean_fraction','ccc']:
    for quant in [.1,.5,.9]:
        vals=sources.groupby(['pairing','model','group'])[col].quantile(quant).rename(f'{col}_source_q{int(quant*100)}').reset_index()
        summary=summary.merge(vals,on=['pairing','model','group'])
summary.to_csv(OUT/'ccc_summary.csv',index=False)
frc_rows=[]
for (pairing,m),q in pairdf.groupby(['pairing','model']):
    score=q.groupby('source').ccc.sum().mean();native=results[m]['multi_target']['FRC']
    if pairing=='A_native':assert abs(score-native)<1e-10
    zero=10*summary[(summary.pairing==pairing)&(summary.model==m)&(summary.group=='all')].GT_zero_mean_ccc.iloc[0]
    frc_rows.append(dict(pairing=pairing,model=m,reconstructed_FRC=score,archived_native_FRC=native,GT_zero_mean_sensitivity=zero,target_switch_fraction=q.target_switched.mean()))
pd.DataFrame(frc_rows).to_csv(OUT/'ccc_frc.csv',index=False)
# Exact symmetric product attribution, algebraic not causal. Undefined Pearson is
# official zero for this reconstruction only; it stays NaN in shape statistics.
attrs=[];factors=['official_r','scale_factor','mean_factor']
for pairing in ['A_native','B_fixed_G0']:
    base=df[(df.pairing==pairing)&(df.model=='G0-local')].set_index(['source','candidate','channel'])
    for model in ['T0','G1-shared']:
        alt=df[(df.pairing==pairing)&(df.model==model)].set_index(['source','candidate','channel']).loc[base.index]
        x=base[factors].to_numpy();z=alt[factors].to_numpy();contrib=np.zeros_like(x)
        for order in itertools.permutations(range(3)):
            cur=x.copy()
            for idx in order:
                before=cur.prod(1);cur[:,idx]=z[:,idx];contrib[:,idx]+=(cur.prod(1)-before)/6
        assert np.max(np.abs(contrib.sum(1)-(alt.ccc.to_numpy()-base.ccc.to_numpy())))<1e-12
        tab=pd.DataFrame(contrib,columns=['r_contribution','scale_contribution','mean_contribution'],index=base.index).reset_index()
        for group,(a,b) in GROUPS.items():
            q=tab[(tab.channel>=a)&(tab.channel<b)]
            for source,v in q.groupby('source'):
                attrs.append(dict(pairing=pairing,model_minus_G0=model,source=source,group=group,**v[['r_contribution','scale_contribution','mean_contribution']].mean().to_dict()))
at=pd.DataFrame(attrs);at.to_csv(OUT/'ccc_attribution_per_source.csv',index=False)
at.groupby(['pairing','model_minus_G0','group']).mean(numeric_only=True).drop(columns='source').to_csv(OUT/'ccc_attribution_summary.csv')
# Existing spread decomposition, source equal and dimension weighted.
parts=[]
for p,hist in [(ROOT/'decomposition.csv',False),(Path('runs/reaction_flow/t0_next_v1/decomposition_per_source.csv'),True)]:
    inputs[str(p)]=sha(p);d=pd.read_csv(p)
    if hist:d=d[d.model.isin(['T0','MAM'])]
    parts.append(d[['model','source','group','total','DC','slow','fast']])
d=pd.concat(parts);error=(d.total-d[['DC','slow','fast']].sum(axis=1)).abs().max();assert error<1e-12
assert (d.groupby(['model','group']).source.nunique()==80).all()
s=d.groupby(['model','group'])[['total','DC','slow','fast']].mean().reset_index()
weights={'AU':.6,'VA':.08,'expression':.32};weighted=d.copy()
for c in ['total','DC','slow','fast']:weighted[c]*=weighted.group.map(weights)
overall=weighted.groupby(['model','source'])[['total','DC','slow','fast']].sum().groupby('model').mean().reset_index();overall['group']='all'
pd.concat([s,overall]).to_csv(OUT/'decomposition_summary.csv',index=False)
blocks=pd.DataFrame(block_rows);old=pd.read_csv(ROOT/'block_diagnostics.csv');inputs[str(ROOT/'block_diagnostics.csv')]=sha(ROOT/'block_diagnostics.csv')
keys=['model','source','group','left_block','right_block','left_frames','right_frames'];matched=blocks.merge(old,on=keys,validate='one_to_one',suffixes=('_new','_old'))
assert len(matched)==len(old);block_error=(matched.candidate_covariance_new-matched.candidate_covariance_old).abs().max();assert block_error<1e-12
blocks.to_csv(OUT/'block_matched.csv',index=False)
bs=blocks.groupby(['model','source','group'])[['candidate_covariance','correlation','undefined_fraction']].mean().reset_index();bs.to_csv(OUT/'block_per_source.csv',index=False)
bs.groupby(['model','group']).mean(numeric_only=True).drop(columns='source').to_csv(OUT/'block_summary.csv')
# Compare G1/G0 identical source/block pairs, preserve tail lengths.
index=['source','group','left_block','right_block','left_frames','right_frames']
b0=blocks[blocks.model=='G0-local'].set_index(index);b1=blocks[blocks.model=='G1-shared'].set_index(index);assert b0.index.equals(b1.index)
delta=b1[['candidate_covariance','correlation']]-b0[['candidate_covariance','correlation']];delta.to_csv(OUT/'block_G1_minus_G0.csv')
# Training diagnostic gradients: same scheduled probe steps, actual weighted ratios.
grad_rows=[]
for model in ['G0-local','G1-shared']:
    p=ROOT/model/'attempt_000/training.jsonl';inputs[str(p)]=sha(p)
    for line in p.read_text().splitlines():
        r=json.loads(line)
        if 'FM_gradient' in r:
            grad_rows.append(dict(model=model,step=r['phase_step'],FM_gradient=r['FM_gradient'],task_gradient=r['task_gradient'],weighted_task_to_FM=r['lambda_task']*r['task_gradient']/r['FM_gradient'],cosine=r['FM_task_cosine']))
grad=pd.DataFrame(grad_rows);grad.to_csv(OUT/'gradient_probes.csv',index=False)
grad.groupby('model').agg({'weighted_task_to_FM':['mean','median','min','max'],'cosine':['mean','median']}).to_csv(OUT/'gradient_summary.csv')
probe=read(ROOT/'G1-shared_noise_factor_probe.json');probe_rows=[]
for r in probe['rows']:
    for group,v in r['groups'].items():probe_rows.append(dict(source=r['source'],probe=r['probe'],group=group,**v))
pd.DataFrame(probe_rows).groupby(['probe','group']).mean(numeric_only=True).drop(columns='source').to_csv(OUT/'factor_probe_summary.csv')
read(ROOT/'FINAL_RESULTS.json');read(ROOT/'PROTOCOL.json')
(OUT/'verification.json').write_text(json.dumps(dict(HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sources=80,K=10,ccc_matrix_max_error=max_matrix_error,ccc_identity_max_error=max_identity_error,spread_identity_max_error=float(error),block_covariance_max_error=float(block_error),block_pairs_per_model=int(len(b0)),block_sources=int(bs.source.nunique()),tail_pairs_per_model=int((b0.reset_index().right_frames<750).sum()),tail_min_frames=int(b0.reset_index().right_frames.min()),prior_mean_variance_ratio=1+.05*749,prior_mean_sd_ratio=math.sqrt(1+.05*749),inputs=inputs,analysis_sha256=sha(__file__)),indent=2))
print('DONE',json.dumps(frc_rows),flush=True)
