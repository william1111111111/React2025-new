"""Read-only validation plus new validation result. No targets enter annotator requests."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from jsonschema import Draft202012Validator
from .build import OUT,ANN,ROOT,read,sha,write

def jsonlines(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def reject_identifiers(x):
    if isinstance(x,dict):
        if set(x)&{'session_id','session','date','relative','target','targets','listener_reaction','listener_future'}:raise ValueError('forbidden model input field')
        for v in x.values():reject_identifiers(v)
    elif isinstance(x,list):
        for v in x:reject_identifiers(v)
    elif isinstance(x,str) and ('Camera-2024-' in x or '/data/' in x):raise ValueError('descriptive path in request')

def validate_support(support,events,observations):
    e=events[support['event_id']]
    for item in support['observed_support']:
        o=observations[item['observation_id']]
        if o['status']!='observed' or o['actions'] is None or o['relation_status']!='temporal_support':raise ValueError('unobserved or unassociated support')
        if o['split']!=e['split'] or o['fold']!=e['fold']:raise ValueError('cross-fold support')
        if not item['association_evidence']:raise ValueError('no association evidence')
    if support['status']=='supported' and not support['observed_support']:raise ValueError('missing is not no reaction')

def main():
    schemas={n:Draft202012Validator(read(ANN/'schema'/(n+'.schema.json'))) for n in ['speaker_events','listener_reactions','event_equivalence','reaction_support']}
    rows={}
    for split in ['train','val','test']:
        for name,validator in schemas.items():
            rr=jsonlines(ANN/split/(name+'.jsonl'));rows[split,name]=rr
            for r in rr:validator.validate(r);assert r['split']==split
            if split!='train':assert not rr
    events={r['event_id']:r for r in rows['train','speaker_events']};obs={r['observation_id']:r for r in rows['train','listener_reactions']}
    assert len(events)==600 and len(obs)==600
    resolver=read(ANN/'train/ASSET_RESOLVER_PRIVATE.json');mapping={r['clip_id']:r for r in read(ANN/'train/RECORDING_MAP_PRIVATE.json')}
    for e in events.values():
        assert e['fold']=='RM_fit' and not e['training_eligible']
        text=(ROOT/'data/train/text/speaker'/(mapping[e['clip_id']]['relative']+'.txt')).read_text();a,b=e['transcript_span'];assert text[a:b]==e['transcript']
        assert not e['human_reviewed'] and e['confidence'] is None
        if e['start_frame'] is not None:
            assert e['role_status']=='source_speaker_candidate'
            assert e['sync_evidence']['status']=='estimated' and not e['sync_evidence']['sync_verified']
            assert 0<=e['start_frame']<e['end_frame']
    for r in obs.values():
        assert r['actions'] is None and r['status']=='pending_observation' and not r['training_eligible']
        assert all(i in events for i in r['linked_candidate_ids'])
    for e in rows['train','event_equivalence']:
        a,b=events[e['event_a']],events[e['event_b']];assert a['clip_id']!=b['clip_id'] and a['fold']==b['fold']=='RM_fit'
        assert e['status']=='UNKNOWN' and e['equivalence'] is None
    for s in rows['train','reaction_support']:validate_support(s,events,obs)
    nr=0
    for kind,role in [('speaker','speaker'),('listener','listener'),('equivalence','speaker')]:
        for r in jsonlines(ANN/'train/requests'/(kind+'.jsonl')):
            reject_identifiers(r['input']);assert r['visible_roles']==[role];nr+=1
            def visit(v):
                if isinstance(v,dict):
                    for q in v.values():visit(q)
                elif isinstance(v,list):
                    for q in v:visit(q)
                elif isinstance(v,str) and (v.startswith('asset_') or v.startswith('pts_')):
                    assert resolver[v]['role']==role and resolver[v]['split']=='train'
            visit(r['input'])
    m=read(OUT/'evaluation/manifest.json');lock=read(OUT/'evaluation/LOCK.json');assert sha(OUT/'evaluation/manifest.json')==lock['manifest_sha256']
    assert len(m['sources'])==1142 and m['K']==m['R']==10
    for s in m['sources']:
        assert len(s['targets'])==10 and all('/data/val/' in r['path'] for r in s['targets'])
    write(OUT/'VALIDATION.json',dict(pass_=True,schema_records=sum(map(len,rows.values())),requests_checked=nr,source_only_roles_checked=True,fold_boundary_checked=True,transcript_substrings_verified=600,reviewed_observations=0,cache_ready=m['processor']['cache_ready'],limitations=['schema success does not validate semantics','PTS count equality does not independently verify synchronization','no visual annotation or human review completed']))
    print('validated',sum(map(len,rows.values())),'schema rows;',nr,'isolated requests')
if __name__=='__main__':main()
