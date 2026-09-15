"""Independent process lifecycle/progress recorder; never changes model or restarts jobs."""
import argparse,datetime,json,os,signal,subprocess,time,shutil,fcntl
from pathlib import Path


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def proc(pid):
    try:
        text=Path(f'/proc/{pid}/stat').read_text();v=text[text.rfind(')')+2:].split()
        return dict(pid=pid,state=v[0],ppid=int(v[1]),session=int(v[3]),start_ticks=int(v[19]),cpu_ticks=int(v[11])+int(v[12]),rss_bytes=int(v[21])*os.sysconf('SC_PAGE_SIZE'))
    except (OSError,ValueError,IndexError):return None


def append(path,row):
    with path.open('a') as f:f.write(json.dumps({'at':now(),**row},allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())


def atomic(path,row):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(row,indent=2)+'\n');tmp.replace(path)


def exit_info(code):
    return dict(returncode=code,signal=signal.Signals(-code).name if code is not None and code<0 else None,exit_code=code if code is not None and code>=0 else None)


def snapshot(root,pid,start):
    current=proc(pid);alive=current is not None and current['start_ticks']==start and current['state']!='Z'
    children=[]
    if alive:
        for d in Path('/proc').iterdir():
            if d.name.isdigit():
                q=proc(int(d.name))
                if q and q['session']==current['session']:children.append(q)
    arms={}
    for a in ('M0-control','M1-mode'):
        row={}
        for label,p in [('queue',root/(a+'_queue.json')),('training',root/'training'/a/'status.json')]:
            try:
                r=json.loads(p.read_text());latest=r.pop('latest',{})
                row[label]={k:r.get(k) for k in ('stage','actual','target','error') if k in r}
                row[label]['age_s']=round(time.time()-p.stat().st_mtime,1)
                row[label]['last_loss']=latest.get('loss')
            except (OSError,ValueError):pass
        logs=list(root.glob(a+'*.log'))
        if logs:
            newest=max(logs,key=lambda p:p.stat().st_mtime);row['latest_log']=dict(path=str(newest),age_s=round(time.time()-newest.stat().st_mtime,1),bytes=newest.stat().st_size)
        if row.get('queue',{}).get('stage')=='training' and row.get('training',{}).get('age_s',0)>600:row['warning']='training status stale >600s; investigate, not automatic restart'
        arms[a]=row
    try:
        gpu=subprocess.run(['nvidia-smi','-i','7','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8)
        gpu=gpu.stdout.strip() if gpu.returncode==0 else 'nvidia-smi failed'
    except (OSError,subprocess.TimeoutExpired):gpu='nvidia-smi unavailable'
    memory={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.split(':')[0] in ('MemAvailable','SwapFree','SwapTotal'):memory[line.split(':')[0]]=line.split(':')[1].strip()
    cgroup={}
    try:
        for line in Path(f'/proc/{pid}/cgroup').read_text().splitlines():
            hierarchy,controllers,rel=line.split(':',2)
            if hierarchy=='0':
                for name in ('memory.events','memory.current','memory.max','pids.events'):
                    f=Path('/sys/fs/cgroup')/rel.lstrip('/')/name
                    if f.exists():cgroup[name]=f.read_text().strip()
    except OSError:pass
    return dict(at=now(),queue_alive=alive,queue=current,processes=children,arms=arms,gpu7_mib_total_util=gpu,host_memory=memory,disk_free_bytes=shutil.disk_usage(root).free,cgroup=cgroup)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--attach',type=int);p.add_argument('--interval',type=float,default=30);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    folder=a.root/'runtime';folder.mkdir(exist_ok=True);lock=open(folder/'watch.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    events=folder/'events.jsonl';child=None;handle=None
    if a.attach:pid=a.attach
    else:
        command=a.command[1:] if a.command[:1]==['--'] else a.command
        if not command:raise ValueError('provide --attach PID or -- command')
        handle=open(folder/'supervised_queue.log','a');child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True);pid=child.pid
    initial=proc(pid)
    if initial is None:raise ValueError('queue is not live')
    append(events,dict(event='watch_started',watcher_pid=os.getpid(),queue=initial,mode='attached; exit code unavailable for non-child' if a.attach else 'parent; captures exit status',interval_s=a.interval))
    def stopped(signum,frame):
        append(events,dict(event='watcher_signal',signal=signal.Signals(signum).name,queue_pid=pid,action='monitor exits; running training not terminated'))
        raise SystemExit(128+signum)
    for sig in (signal.SIGTERM,signal.SIGHUP,signal.SIGINT):signal.signal(sig,stopped)
    previous=None
    while True:
        code=child.poll() if child else None
        row=snapshot(a.root,pid,initial['start_ticks']);atomic(folder/'heartbeat.json',row);append(folder/'heartbeats.jsonl',row)
        stages={k:{kk:vv for kk,vv in v.get('queue',{}).items() if kk!='age_s'} for k,v in row['arms'].items()};states=json.dumps(stages,sort_keys=True)
        if states!=previous:append(events,dict(event='stage_change',arms=stages));previous=states
        if not row['queue_alive'] or child is not None and code is not None:
            result=dict(event='queue_exited',**exit_info(code),exit_status_available=child is not None,arms=row['arms'],explanation='OS child status' if child else 'attached non-child disappeared or zombie; cause unknown; inspect host logs')
            append(events,result);atomic(folder/'exit.json',dict(at=now(),**result));return
        time.sleep(a.interval)

if __name__=='__main__':main()
