from .common import OUT,ARMS,read,write

def main():
    base=read(OUT/'PROTOCOL.json')['parent_metrics'];results={};cost={}
    for arm in ARMS:
        result=read(OUT/'task_multitarget'/(arm+'_step500.json'))['result'];frd=read(sorted((OUT/'frd20'/(arm+'_step500')).glob('status_*.json'))[-1]);assert frd['completed_pairs']==2000
        m={k:result['metrics'][k] for k in ('FRC','smse','FRVar')};m['exact_FRD20']=frd['FRD'];m['working_goal_met']=m['FRC']>=base['FRC'] and m['exact_FRD20']<=base['exact_FRD20'] and m['smse']>=1.5*base['smse'];results[arm]=m
        rows=[__import__('json').loads(x) for x in (OUT/'training'/arm/'training.jsonl').read_text().splitlines()];assert len(rows)==500
        cost[arm]=dict(updates=len(rows),candidate_rollouts=sum(r['candidate_rollouts'] for r in rows),total_update_seconds=sum(r['seconds'] for r in rows),CPU_metric_seconds=sum(e['metric_seconds'] for r in rows for e in r['episodes']),peak_gpu_bytes=max(r['peak_gpu_bytes'] for r in rows))
    write(OUT/'RESULTS.json',dict(parent=base,arms=results,cost=cost,decision='joint working goal met for some policies; independent confirmation needed' if any(m['working_goal_met'] for m in results.values()) else 'joint working goal not met; no automatic reward grid or decoder RL',development_only=True))
    lines=['# Explicit sampling policy results','','| Model | FRC80 | exact FRD20 | S-MSE80 | FRVar80 |','|---|---:|---:|---:|---:|']
    for name,m in [('fixed P2',base),*results.items()]:lines.append(f"| {name} | {m['FRC']:.6f} | {m['exact_FRD20']:.6f} | {m['smse']:.6f} | {m['FRVar']:.6f} |")
    lines+=['','Same frozen generator, K10 source-only policy sampling. No target-selected inference; exact FRD uses the fixed 20-source subset. TRAIN block rewards are not official full-record metrics. Single-seed development experiment.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n');write(OUT/'completion.json',dict(complete=True))
if __name__=='__main__':main()
