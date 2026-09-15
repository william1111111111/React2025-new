import json,numpy as np
from .audit import OUT
from reward_policy.common import read,write

def main():
    result={};lines=['# Reward signal v2: fixed100 update comparison','','This is a TRAIN signal-chain experiment, not a final FRC/FRD/diversity claim. R1 changes both feature scaling and kernel bandwidth; bandwidth-only and pooled-scale diagnostics were retained separately.','','| Arm | Updates | Median coverage gradient | Median quality gradient | Median actual optimizer delta | Final probe output RMS change |','|---|---:|---:|---:|---:|---:|']
    for a in ('R0-baselined','R1-calibrated'):
        d=OUT/'training'/a;rows=[json.loads(x) for x in (d/'training.jsonl').read_text().splitlines()];assert len(rows)==100
        probe=read(d/'probe_100.json');r=dict(updates=100,median_bonus_gradient=float(np.median([x['gradients']['bonus_norm'] for x in rows])),median_quality_gradient=float(np.median([x['gradients']['quality_norm'] for x in rows])),median_optimizer_delta=float(np.median([x['optimizer_delta_L2'] for x in rows])),final_parameter_relative_change=probe['parameter_relative_change'],final_probe_output_RMS=float(np.mean([x['prediction_delta_RMS'] for x in probe['episodes']])),final_probe_action_RMS=float(np.mean([x['action_delta_RMS'] for x in probe['episodes']])),probe=probe,formal_action_rollouts=2000,probe_action_rollouts=80,reference_rollouts_reused=2080,seconds=sum(x['seconds'] for x in rows));result[a]=r
        lines.append(f"| {a} | 100 | {r['median_bonus_gradient']:.6g} | {r['median_quality_gradient']:.6g} | {r['median_optimizer_delta']:.6g} | {r['final_probe_output_RMS']:.6g} |")
    ratio=result['R1-calibrated']['median_bonus_gradient']/max(1e-300,result['R0-baselined']['median_bonus_gradient']);diagnosis='coverage reward starvation reduced in this finite TRAIN comparison' if ratio>10 else 'coverage credit remains weak or inconsistent; inspect saved diagnostics'
    write(OUT/'RESULTS.json',dict(arms=result,diagnosis=diagnosis,bonus_gradient_ratio=ratio,official_metrics_run=False,scope='100 updates per arm, single seed; no budget extension',geometry_limit='AU distances remain dominant even after stratified scaling; no calibrated correctness claim'))
    lines+=['',diagnosis,'','All noise/source/target schedules, parent, policy initialization and optimization settings are shared. Native motion and action/parameter/covariance diagnostics are in probe_000/025/050/100.json. Old near-maximal speed/acceleration gates are not evidence of realistic motion.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n');write(OUT/'completion.json',dict(complete=True))
if __name__=='__main__':main()
