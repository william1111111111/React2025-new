import os,signal,subprocess,sys
from .watch import exit_info,proc

def test_exit_status():
    p=subprocess.Popen([sys.executable,'-c','import sys;sys.exit(7)']);assert exit_info(p.wait())['exit_code']==7
    p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);p.send_signal(signal.SIGTERM);r=exit_info(p.wait());assert r['signal']=='SIGTERM' and r['returncode']==-15

def test_process_identity():
    p=proc(os.getpid());assert p['pid']==os.getpid() and p['start_ticks']>0 and p['rss_bytes']>0


def test_append_existing_timestamp(tmp_path):
    import json
    from .watch import append
    p=tmp_path/'events.jsonl'
    append(p,dict(at='recorded',event='heartbeat'))
    append(p,dict(event='new'))
    rows=[json.loads(x) for x in p.read_text().splitlines()]
    assert rows[0]['at']=='recorded' and rows[1]['at']
