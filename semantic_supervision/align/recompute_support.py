import collections,json,hashlib,re
from pathlib import Path
from .support import support,POLICY
ROOT=Path(__file__).resolve().parents[2];OLD=ROOT/'runs/reaction_flow/semantic_alignment_pilot_v1';OUT=ROOT/'runs/reaction_flow/semantic_auto_weak_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    OUT.mkdir(exist_ok=True);(OUT/'POLICY.json').write_text(json.dumps(POLICY,indent=2));all_events=[];old_ids=[];new_ids=[];strata=collections.defaultdict(collections.Counter)
    for f in sorted((OLD/'source').glob('*/result.json')):
        record=json.loads(f.read_text());scores=json.loads((f.parent/'active_speaker_scores.json').read_text());words=json.loads((f.parent/'words.json').read_text());vt=record['video_timeline']
        for e in record['events']:
            chosen=[w for w in words if w['start_s']>=e['start_s']-1e-6 and w['end_s']<=e['end_s']+1e-6]
            checks={'ctc_mean':e['ctc_mean_score']>=.5,'greedy_agreement':e['greedy_asr_char_agreement']>=.6,'word_duration':bool(chosen) and all(w['end_s']-w['start_s']<=2 for w in chosen),'no_digits':not bool(re.search(r'\d',e['transcript_evidence']))}
            s=support(e['start_s'],e['end_s'],scores['times_s'],scores['raw_speaking_logit'],scores['shifted_audio_3s_logit'],video_start=vt['first_pts'],video_end=vt['duration_s'],audio_start=0,audio_end=record['audio_duration_s'])
            timing=all(checks.values());passed=timing and s['passed']
            reasons=['timing:'+k for k,v in checks.items() if not v]+['role:'+k for k in s['rejection_reasons']]
            group='old_joint' if e['automatic_weak_candidate'] else 'old_timing_only' if e['time_status']=='automatic_supported' else 'old_timing_uncertain'
            strata[group].update(reasons)
            if e['automatic_weak_candidate']:old_ids.append(e['event_id'])
            if passed:new_ids.append(e['event_id'])
            all_events.append({'clip_id':record['clip_id'],'event':e,'old_group':group,'timing_rules':checks,'support':s,'rejection_reasons':reasons,'joint_passed':passed,'source_record_sha256':sha(f),'scores_sha256':sha(f.parent/'active_speaker_scores.json'),'words_sha256':sha(f.parent/'words.json')})
    report={'old_joint':len(old_ids),'new_joint':len(new_ids),'retained':sorted(set(old_ids)&set(new_ids)),'removed':sorted(set(old_ids)-set(new_ids)),'added':sorted(set(new_ids)-set(old_ids)),'old_42_ids':old_ids,'new_ids':new_ids,'reasons_by_old_stratum':strata}
    (OUT/'support_events.json').write_text(json.dumps(all_events,ensure_ascii=False,indent=2));(OUT/'support_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k in ['old_joint','new_joint','removed','added','reasons_by_old_stratum']},ensure_ascii=False))
if __name__=='__main__':main()
