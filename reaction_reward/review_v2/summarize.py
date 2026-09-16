"""Summarize all saved selections; no inference, no new distances."""
import numpy as np
from reaction_reward.common import ROOT,OUT as OLD,read,write,sha
OUT=ROOT/'runs/reaction_reward/review_v2'
METRICS=['sum_best_CCC','sum_best_native_DTW','S_MSE']
def aggregate(rows,method,baseline):
 delta={k:np.array([r['selection'][method][k]-r['selection'][baseline][k] for r in rows]) for k in METRICS};both=(delta['sum_best_CCC']>0)&(delta['sum_best_native_DTW']<0)
 return dict(windows=len(rows),date_groups=len({r['group'] for r in rows}),mean_delta={k:float(v.mean()) for k,v in delta.items()},joint_quality_improvement_fraction=float(both.mean()),joint_nondegradation_fraction=float(((delta['sum_best_CCC']>=0)&(delta['sum_best_native_DTW']<=0)).mean()),qualified_retention={mode:dict(eligible_windows=sum(r['qualified_total']>0 for r in rows),window_macro_fraction=float(np.mean([r['qualified_retained'][mode]/r['qualified_total'] for r in rows if r['qualified_total']>0])) if any(r['qualified_total'] for r in rows) else None,candidate_micro_fraction=sum(r['qualified_retained'][mode] for r in rows)/sum(r['qualified_total'] for r in rows) if sum(r['qualified_total'] for r in rows) else None) for mode in [baseline,method]})
def main():
 results={};hashes={};lines=['# Existing fixed-pool selection: complete saved evidence','','Best-of16 window diagnostic, not full-recording FRC/FRD or native K10. Positive delta CCC and negative delta DTW are improvements. No new candidates or DTW were computed.','','|Arm|Generator|Baseline|ΔCCC sum|Δnative DTW sum|ΔS-MSE|Both quality improved|','|---|---|---|---:|---:|---:|---:|'];allrows=[]
 for arm in ['RM-Paired','RM-Multi','RM-Gen']:
  path=OLD/'audit'/arm/'selection_probe.json';rows=read(path);hashes[str(path)]=sha(path);assert len(rows)==288 and len({(r['window'],r['generator']) for r in rows})==288
  for r in rows:
   b=OLD/'bank'/r['window'];m=read(b/(r['generator']+'_scores.json'));p=read(b/'P2_scores.json');C=np.array(m['C']).max(1);D=np.array(m['D']).min(1);q=(C>=np.array(p['C']).max(1).mean())&(D<=np.array(p['D']).min(1).mean());assert q.sum()==r['qualified_count'];r['qualified_total']=int(q.sum());r['qualified_retained']={k:int(q[v['selected']].sum()) for k,v in r['selection'].items()}
   for k,v in r['selection'].items():assert len(v['selected'])==10 and len(set(v['selected']))==10;assert np.isclose(C[v['selected']].sum(),v['sum_best_CCC']);assert np.isclose(D[v['selected']].sum(),v['sum_best_native_DTW'])
   allrows.append(dict(r,arm=arm))
  results[arm]={}
  for gen in ['P2','N0','N1']:
   rs=[r for r in rows if r['generator']==gen];cell={}
   for method in ['RM_top10','metric_oracle_diagnostic']:
    for base in ['first10','random10']:
     x=aggregate(rs,method,base);x['by_date_group']={g:aggregate([r for r in rs if r['group']==g],method,base) for g in sorted({r['group'] for r in rs})};cell[method+'_vs_'+base]=x
     if method=='RM_top10':
      d=x['mean_delta'];lines.append(f"|{arm}|{gen}|{base}|{d[METRICS[0]]:+.6f}|{d[METRICS[1]]:+.6f}|{d[METRICS[2]]:+.6f}|{x['joint_quality_improvement_fraction']:.1%}|")
   results[arm][gen]=cell
 write(OUT/'SELECTION_SUMMARY.json',dict(total_saved_records=len(allrows),unique_windows=96,models=3,generators=3,inputs=hashes,results=results,qualification='per candidate bestCCC >= same-window P2 mean bestCCC AND minDTW <= same-window P2 mean minDTW; metric-derived proxy only',oracle='saved Pareto-dominance-count then CCC heuristic, GT-reading diagnostic, not mathematical optimal selection',bootstrap_claim='3 date components only; descriptive stratification, no claim of broad significance'))
 write(OUT/'SELECTION_RETENTION_PER_WINDOW.json',allrows);(OUT/'SELECTION_SUMMARY.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
