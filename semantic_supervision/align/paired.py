"""TRAIN-only relation staging. Missing verified events means unknown, not causal links."""
import json
from pathlib import Path
from semantic_supervision.extract.common import output,wrapper

def main():
    source=json.loads(Path('/inputs/source/annotations.json').read_text());listener=json.loads(Path('/inputs/listener/annotations.json').read_text())
    sm={r['clip_id']:r for r in source};lm={r['clip_id']:r for r in listener};assert sm.keys()==lm.keys()
    rows=[];review=[]
    for key,s in sm.items():
        assert s['split']==lm[key]['split']=='train'
        rows.append(wrapper('event_links',key,min(s['duration_s'],lm[key]['duration_s']),[]))
        review.append(dict(clip_id=key,status='unknown',reason='verified speaker semantic events and paired sync unavailable',causal_claim=False,training_eligible=False,source_event_ids=[e['event_id'] for e in s['events']],listener_proposal_ids=[e['event_id'] for e in lm[key]['events']]))
    output('annotations.json',rows);output('review_status.json',review)
if __name__=='__main__':main()
