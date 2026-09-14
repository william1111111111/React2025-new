"""Frozen-pipeline expansion: two new accepted TRAIN transcripts per session."""
import json,collections,hashlib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/reaction_flow/semantic_auto_weak_v1';OUT=BASE/'expansion';OLD=ROOT/'runs/reaction_flow/semantic_alignment_pilot_v1'
def main():
    OUT.mkdir(exist_ok=True);jobdir=OUT/'job';jobdir.mkdir(exist_ok=True);dest=OUT/'source';dest.mkdir(exist_ok=True)
    oldjob=json.loads((OLD/'job/job.json').read_text());oldids={r['id'] for r in oldjob['records']}
    full=ROOT/'runs/reaction_flow/semantic_supervision_full_train_v1';selected=json.loads((full/'selected_records.json').read_text());mapping=json.loads((full/'private_recording_map.json').read_text())
    groups=collections.defaultdict(list)
    for r in mapping:
        if r['id'] not in oldids and r['id'] in selected:groups[r['relative'].split('/')[0]].append(r)
    rows=[]
    for session in sorted(groups):
        for r in sorted(groups[session],key=lambda r:r['relative'])[:2]:
            item=selected[r['id']];cs=json.loads((Path(item['folder'])/'candidates.json').read_text())['candidates']
            rows.append({**r,'transcript_sha256':item['result']['transcript_sha256'],'candidates':cs})
    assert len(rows)==34 and len(groups)==17
    (jobdir/'job.json').write_text(json.dumps({'records':rows,'teachers':oldjob['teachers']},indent=2))
    code=['semantic_supervision/align/automatic.py','semantic_supervision/align/support.py','semantic_supervision/align/lag_probe.py','semantic_supervision/models/auto_weak.py','semantic_supervision/models/semantic_flow.py']
    frozen={'selection':'two lexicographically first unused accepted TRAIN transcripts per session; no listener or DEV scores','records':34,'events':sum(len(r['candidates']) for r in rows),'code_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in code},'support_policy':json.loads((BASE/'POLICY.json').read_text()),'lag_policy':json.loads((BASE/'lag_policy.json').read_text()),'new_paid_api_calls':0}
    (OUT/'FROZEN.json').write_text(json.dumps(frozen,indent=2))
    cmd=json.loads((OLD/'sandbox_command.json').read_text());cmd=[str(dest) if x==str(OLD/'source') else str(jobdir) if x==str(OLD/'job') else x for x in cmd]
    (OUT/'sandbox_command.json').write_text(json.dumps(cmd,indent=2))
    with (OUT/'execution.log').open('a') as f:subprocess.run(cmd,env={},stdout=f,stderr=subprocess.STDOUT,check=True)
    records=[json.loads(p.read_text()) for p in dest.glob('*/result.json')];events=[e for r in records for e in r['events']]
    summary={'records_done':len(records),'records_planned':34,'events':len(events),'timing_supported':sum(e['time_status']=='automatic_supported' for e in events),'joint_support_before_lag':sum(e['automatic_weak_candidate'] for e in events),'human_reviewed':False,'training_eligible':0}
    (OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
if __name__=='__main__':main()
