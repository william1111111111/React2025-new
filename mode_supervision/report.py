"""Observed final endpoint only; strict quality/diversity targets fixed beforehand."""
import json
from .prepare import OUT,write

def main():
    rows={}
    for arm in ('M0-control','M1-mode'):
        label=f'seed123_{arm}_step20000';task=json.loads((OUT/'task_multitarget'/(label+'.json')).read_text())['result'];frd=json.loads(sorted((OUT/'frd20'/label).glob('status_*.json'))[-1].read_text());assert frd['completed_pairs']==2000
        rows[arm]=dict(**task['metrics'],exact_FRD20=frd['FRD'])
    costs={}
    for arm in ('M0-control','M1-mode'):
        logs=[json.loads(x) for x in (OUT/'training'/arm/'training.jsonl').read_text().splitlines()]
        warm=OUT/'training'/arm/'prior_warmup.jsonl';wr=[json.loads(x) for x in warm.read_text().splitlines()] if warm.exists() else []
        costs[arm]=dict(decoder_updates=len(logs),warmup_updates=len(wr),decoder_and_concurrent_prior_seconds=sum(r['seconds'] for r in logs),prior_warmup_seconds=sum(r['seconds'] for r in wr),peak_process_gpu_bytes=max(r['peak_gpu_bytes'] for r in logs))
    m0,m1=rows['M0-control'],rows['M1-mode'];dec=dict(diversity_50pct=m1['smse']>=1.5*m0['smse'],FRC_preserves_M0=m1['FRC']>=m0['FRC'],FRD_preserves_M0=m1['exact_FRD20']<=m0['exact_FRD20'],FRC_preserves_T0=m1['FRC']>=.859622576,FRD_preserves_T0=m1['exact_FRD20']<=133.876667407)
    oracle=json.loads((OUT/'task_multitarget/seed123_M1-mode_step20000_oracle.json').read_text())['result']['metrics']
    write(OUT/'RESULTS.json',dict(main=rows,compute=costs,oracle_diagnostic_only=oracle,criteria=dec,decision='joint working goal met on development set; independent confirmation still needed' if all(dec.values()) else 'joint goal not met; report observed quality/diversity tradeoff',seed=123,decoder_steps_each=6000,prior_updates=8000,paid_API_calls=0))
    lines=['# Observable mode supervision','', '| Arm | FRC80 | exact FRD20 | S-MSE80 | FRVar80 |','|---|---:|---:|---:|---:|']
    for arm,r in rows.items():lines.append(f"| {arm} | {r['FRC']:.6f} | {r['exact_FRD20']:.6f} | {r['smse']:.6f} | {r['FRVar']:.6f} |")
    lines+=['','Formal results use source-predicted plans; oracle and interventions are separate diagnostics. Single seed, development results; added prior compute is not matched FLOPs.','',str(dec)]
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n');write(OUT/'completion.json',dict(complete=True))
if __name__=='__main__':main()
