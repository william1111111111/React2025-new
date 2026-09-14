"""Complete frozen source-only DEV caches, preserving val split and gate policy."""
import json,subprocess,shutil
from pathlib import Path
from semantic_supervision.align import recompute_support as r,build_weak_cache as b
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/reaction_flow/semantic_controlled_v1'
def main():
 old=OUT/'dev_raw';old.mkdir(exist_ok=True)
 link=old/'source'
 if not link.exists():link.symlink_to(OUT/'dev_source',target_is_directory=True)
 r.OLD=old;r.OUT=OUT/'dev_weak';r.main();base=r.OUT
 job=json.loads((OUT/'dev_job/job.json').read_text());assert len(list((OUT/'dev_source').glob('*/result.json')))==80
 support=json.loads((base/'support_events.json').read_text());rows=[]
 for row in job['records']:
  es=[x['event'] for x in support if x['clip_id']==row['id'] and x['joint_passed']]
  if es:rows.append({**row,'probe_events':es})
 dest=base/'lag';dest.mkdir(exist_ok=True);jobdir=base/'lag_job';jobdir.mkdir(exist_ok=True);(jobdir/'job.json').write_text(json.dumps({'split':'val','records':rows,'teachers':job['teachers']},indent=2))
 cmd=json.loads((OUT/'dev_sandbox.json').read_text());cmd=[str(dest) if x==str(OUT/'dev_source') else str(jobdir) if x==str(OUT/'dev_job') else 'semantic_supervision.align.lag_probe' if x=='semantic_supervision.align.automatic' else x for x in cmd]
 shutil.copyfile(ROOT/'runs/reaction_flow/semantic_auto_weak_v1/lag_policy.json',base/'lag_policy.json');(base/'lag_sandbox.json').write_text(json.dumps(cmd,indent=2))
 with (base/'lag.log').open('a') as f:subprocess.run(cmd,env={},stdout=f,stderr=subprocess.STDOUT,check=True)
 b.OLD=old;b.OUT=base;b.main()
 from semantic_supervision.models.auto_weak import read_weak
 index=json.loads((base/'cache/index.json').read_text());accepted=0;non_null=0
 for cid,m in index.items():
  obj=json.loads((base/'cache'/(cid+'.json')).read_text());assert obj['split']=='val'
  es=read_weak(base/'cache'/(cid+'.json'),split='val',expected_cache_sha256=m['cache_sha256'],expected_media_hashes=m['media_hashes'],expected_policy_sha256=m['policy_sha256']);accepted+=len(es or []);non_null+=bool(es)
 summary={'source_records':len(index),'non_null_sources':non_null,'weak_events':accepted,'split':'val','listener_used_for_semantics':False,'paid_api_calls':0,'frozen_parameters':True}
 (OUT/'DEV_READY.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
