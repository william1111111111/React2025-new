import json,time,os,shutil
from pathlib import Path
r=Path(__file__).resolve().parent
previous=None
while True:
    now=time.time()
    try:
        m=json.loads((r/'monitor_latest.json').read_text())
        steps={a:v.get('phase_step') for a,v in m.get('progress',{}).items()}
        alerts=list(m.get('failures',[]))
        if now-m['time']>900: alerts.append('queue_status_stale_over_15min')
        free=shutil.disk_usage(r).free/1024**3
        if free<50: alerts.append('disk_free_below_50GiB')
        done=json.loads((r/'queue_finished.json').read_text()) if (r/'queue_finished.json').exists() else None
        row=dict(time=now,steps=steps,training=m.get('training_active'),evaluation=m.get('evaluation_active'),frd=m.get('FRD_active'),frd_completed=m.get('FRD_completed'),free_GiB=round(free,1),alerts=alerts,finished=done)
        tmp=r/'light_monitor_latest.tmp';tmp.write_text(json.dumps(row,indent=2));tmp.replace(r/'light_monitor_latest.json')
        key=json.dumps([alerts,done,m.get('evaluation_active'),m.get('FRD_active'),m.get('FRD_completed')],sort_keys=True)
        if key!=previous:
            with (r/'light_monitor_events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            previous=key
        if done is not None:break
    except Exception as e:
        with (r/'light_monitor_events.jsonl').open('a') as f:f.write(json.dumps(dict(time=now,error=repr(e)))+'\n')
    time.sleep(300)
