"""Word-anchored proposals for mixed-turn parents; no invented subevent class."""
import json
from pathlib import Path
import numpy as np
from .recompute_support import ROOT,OLD,OUT
from .support import support

def main():
    rows=json.loads((OUT/'support_events.json').read_text());job=json.loads((OLD/'job/job.json').read_text());mapping={r['id']:r for r in job['records']};children=[]
    for x in rows:
        if x['old_group']!='old_joint' or not x['support']['crosses_obvious_turn']:continue
        cid=x['clip_id'];e=x['event'];folder=OLD/'source'/cid;r=json.loads((folder/'result.json').read_text());sc=json.loads((folder/'active_speaker_scores.json').read_text());words=json.loads((folder/'words.json').read_text());text=(ROOT/'data/train/text/speaker'/(mapping[cid]['relative']+'.txt')).read_text()
        t=np.array(sc['times_s']);a=np.array(sc['raw_speaking_logit'],float);c=np.array(sc['shifted_audio_3s_logit'],float);groups=[];current=[]
        for w in words:
            if not(e['start_s']<=w['start_s'] and w['end_s']<=e['end_s']):continue
            mask=(t>=w['start_s'])&(t<w['end_s']);valid=mask&np.isfinite(a)&np.isfinite(c)
            positive=valid.any() and np.mean(a[valid]>0)>=.8
            if positive:current.append(w)
            elif current:groups.append(current);current=[]
        if current:groups.append(current)
        for i,g in enumerate(groups):
            lo=g[0]['start_s'];hi=g[-1]['end_s'];s=support(lo,hi,t,a,c,video_start=r['video_timeline']['first_pts'],video_end=r['video_timeline']['duration_s'],audio_start=0,audio_end=r['audio_duration_s'])
            children.append({'parent_event_id':e['event_id'],'proposal_id':e['event_id']+f'_words_{i}','text':text[g[0]['char_start']:g[-1]['char_end']],'start_s':lo,'end_s':hi,'event_type':'unknown','word_indices_char_range':[g[0]['char_start'],g[-1]['char_end']],'role_support':s,'ctc_mean':float(np.mean([w['ctc_mean_score'] for w in g])),'weak_input_eligible':False,'reason':'split child semantic class and local ASR agreement not independently established; parent class not inherited','human_reviewed':False})
    (OUT/'turn_split_proposals.json').write_text(json.dumps(children,ensure_ascii=False,indent=2));print('word-anchored split proposals',len(children))
if __name__=='__main__':main()
