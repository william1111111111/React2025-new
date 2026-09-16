"""Prospective sampler only: valid evidence, balanced group/source exposure."""
import random,collections,json
from reaction_reward.common import ROOT,OUT as OLD,read,write,sha
OUT=ROOT/'runs/reaction_reward/review_v2'
class EffectiveSampler:
 def __init__(self,records,seed=123):
  self.records=records;self.rng=random.Random(seed);self.pools=collections.defaultdict(lambda:collections.defaultdict(lambda:collections.defaultdict(list)));self.group_uses=collections.Counter();self.source_uses=collections.Counter();self.row_uses=collections.Counter()
  for i,r in enumerate(records):
   if r['split']=='RM_fit' and r['state']=='PREFERRED' and r['weight']>0:self.pools[r['family']][r['group']][r['source']].append(i)
 def draw(self,family,n):
  out=[];seen=set();pool=self.pools[family]
  for _ in range(n):
   available={g:[s for s in ss if s not in seen] for g,ss in pool.items()};available={g:s for g,s in available.items() if s}
   if not available:break
   gs=list(available);self.rng.shuffle(gs);g=min(gs,key=lambda g:self.group_uses[family,g]);ss=available[g];self.rng.shuffle(ss);s=min(ss,key=lambda s:self.source_uses[family,s]);ids=pool[g][s].copy();self.rng.shuffle(ids);i=min(ids,key=lambda i:self.row_uses[i]);out.append(i);seen.add(s);self.group_uses[family,g]+=1;self.source_uses[family,s]+=1;self.row_uses[i]+=1
  return out

def main():
 records=read(OUT/'REAL_EVIDENCE.json')+read(OLD/'GENERATED_EVIDENCE.json');sampler=EffectiveSampler(records);families={'base_context':16,'temporal':8,'weak_context':4,'generated':4};schedule=[];used=collections.defaultdict(list)
 for step in range(4000):
  batch={f:sampler.draw(f,n) for f,n in families.items()}
  for f,ids in batch.items():used[f].extend(ids);assert all(records[i]['split']=='RM_fit' and records[i]['state']=='PREFERRED' and records[i]['weight']>0 for i in ids);assert len({records[i]['source'] for i in ids})==len(ids)
  schedule.append(batch)
 summary={}
 for f,n in families.items():
  pool=[r for r in records if r['family']==f and r['split']=='RM_fit' and r['state']=='PREFERRED' and r['weight']>0];counts=collections.Counter(used[f]);perstep=[len(b[f]) for b in schedule];summary[f]=dict(valid_records=len(pool),independent_sources=len({r['source'] for r in pool}),date_groups=len({r['group'] for r in pool}),scheduled_presentations=len(used[f]),distinct_records_used=len(counts),zero_batches=perstep.count(0),minimum_actual_batch=min(perstep),maximum_actual_batch=max(perstep),max_record_reuse=max(counts.values(),default=0),source_presentations=dict(collections.Counter(records[i]['source'] for i in used[f])),date_group_presentations=dict(collections.Counter(records[i]['group'] for i in used[f])))
 oldlogs={}
 for arm in ['RM-Paired','RM-Multi','RM-Gen']:
  rows=[json.loads(x) for x in (OLD/'training'/arm/'training.jsonl').read_text().splitlines()];rows={r['step']:r for r in rows};nums=[r['exposure']['temporal']['active'] for r in rows.values()];oldlogs[arm]=dict(logged_updates=len(rows),time_zero_batches=nums.count(0),time_active_presentations=sum(nums),mean_time_active=sum(nums)/len(nums))
 write(OUT/'TRAIN_RECORD_INDEX.json',records);write(OUT/'SAMPLING_SCHEDULE.json',dict(executed_training=False,seed=123,steps=4000,family_slots=families,record_index_sha256=sha(OUT/'TRAIN_RECORD_INDEX.json'),rule='least-used date group then source then record; tie seeded; no repeated source within family batch; shortage stays empty',schedule=schedule));write(OUT/'SAMPLING_SUMMARY.json',dict(prospective=summary,actual_v1_logs=oldlogs,unknown_records_preserved=sum(r['state']=='UNKNOWN' for r in records),new_independent_samples_from_resampling=0));print(oldlogs);print({f:{k:v for k,v in s.items() if not k.endswith('presentations')} for f,s in summary.items()})
if __name__=='__main__':main()
