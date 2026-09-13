"""Deterministic200-window proposal export; no human/teacher labels fabricated."""
import collections,csv,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/reaction_flow/semantic_supervision_pilot_v1'

def write_jsonl(path,rows):
    with path.open('w') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')

def main():
    review=OUT/'review';review.mkdir(exist_ok=True)
    mapping={r['id']:r for r in json.loads((OUT/'private_recording_map.json').read_text())}
    candidates=json.loads((OUT/'speaker/window_candidates.json').read_text())
    by=collections.defaultdict(list)
    for c in candidates:by[c['clip_id']].append(c)
    queues={}
    for key,rows in by.items():
        groups={g:[r for r in rows if r['acoustic_stratum']==g] for g in ['low_energy','typical_energy','high_energy']}
        # Spread candidate time locations within each acoustic stratum.
        for group in groups:groups[group]=sorted(groups[group],key=lambda r:hashlib.sha256(r['window_id'].encode()).hexdigest())
        q=[]
        while any(groups.values()):
            for g in groups:
                if groups[g]:q.append(groups[g].pop(0))
        queues[key]=q
    order=sorted(queues,key=lambda k:(mapping[k]['session'],mapping[k]['relative']))
    selected=[]
    while len(selected)<200:
        moved=False
        for key in order:
            if queues[key]:selected.append(queues[key].pop(0));moved=True
            if len(selected)==200:break
        if not moved:raise ValueError('fewer than200 source-only candidate windows')
    assert len({r['window_id'] for r in selected})==200
    # Fifty independent-review slots, chosen without viewing annotations/targets.
    double=set(r['window_id'] for r in sorted(selected,key=lambda r:hashlib.sha256(('double|'+r['window_id']).encode()).hexdigest())[:50])
    tasks=[];source_jobs=[];listener_jobs=[];links=[];answers=[]
    listener={r['clip_id']:r for r in json.loads((OUT/'listener/annotations.json').read_text())}
    for i,w in enumerate(selected):
        key=w['clip_id'];duration=listener[key]['duration_s']
        overlap=[e for e in listener[key]['events'] if e['end_s']>=w['start_s'] and e['start_s']<=w['end_s']]
        task=dict(task_id=f'window_{i+1:03d}',**w,recording_token=key,semantic_event_status='unknown_pending_review',paired_listener_status='TRAIN_only_unverified_sync',listener_review_end_s=min(w['end_s'],duration),double_review=w['window_id'] in double,reviewers_assigned=False)
        tasks.append(task)
        source_jobs.append(dict(task_id=task['task_id'],recording_token=key,start_s=w['start_s'],end_s=w['end_s'],input_stage='speaker',transcript_alignment=None,semantic_events=None,speaker_role='needs_verification',audio_video_sync='needs_verification',teacher_payload_excludes=['session_id','original_clip_id','listener','model_scores']))
        listener_jobs.append(dict(task_id=task['task_id'],recording_token=key,start_s=w['start_s'],end_s=min(w['end_s'],duration),input_stage='listener',numeric_proposals=overlap,visible_no_reaction=None,occlusion=None,psychological_state_annotation_prohibited=True,allowed_for_model_input=False))
        links.append(dict(task_id=task['task_id'],recording_token=key,source_event_ids=None,response_event_ids=None,link_status='unknown',sync_verified=False,allowed_for_model_input=False))
        for reviewer in ['A','B'] if task['double_review'] else ['A']:
            answers.append(dict(task_id=task['task_id'],reviewer_slot=reviewer,status='not_started',transcript_error=None,speaker_role_correct=None,sync_offset_s=None,source_events=None,visible_no_reaction=None,listener_observations=None,occlusion=None,paired_links=None,time_boundary_error_s=None,notes=None))
    write_jsonl(review/'proposal_windows.jsonl',tasks);write_jsonl(review/'speaker_tasks.jsonl',source_jobs);write_jsonl(review/'listener_tasks.jsonl',listener_jobs);write_jsonl(review/'paired_tasks.jsonl',links);write_jsonl(review/'reviewer_answers.template.jsonl',answers)
    write_jsonl(review/'adjudication.template.jsonl',[dict(task_id=t['task_id'],status='await_two_independent_reviews',agreement=None,adjudicated_events=None) for t in tasks if t['double_review']])
    sessions=collections.defaultdict(list)
    for key,r in mapping.items():sessions[r['session']].append(key)
    pair_jobs=[]
    for n in range(20):
        session=sorted(sessions)[n%len(sessions)];a,b=sorted(sessions[session])[:2]
        focus=[t for t in tasks if t['clip_id']==a][n//len(sessions)]
        pair_jobs.append(dict(task_id=f'cross_{n+1:02d}',record_A=a,record_B=b,record_A_review_focus=focus['task_id'],record_A_event_id=None,record_B_event_id=None,match_status=None,review_status='pending',time_alignment=None,visible_roles=['speaker'],for_training_only=True,selection_reason='same-session candidate pair only; no semantic match claimed; reviewer may reject',listener_inputs_present=False))
    write_jsonl(review/'cross_recording_20_tasks.jsonl',pair_jobs)
    distribution=dict(windows=200,independent_double_review_windows=50,reviewer_answer_slots=250,cross_recording_tasks=20,unique_cross_recording_pairs=len({(p['record_A'],p['record_B']) for p in pair_jobs}),sessions=dict(collections.Counter(mapping[t['clip_id']]['session'] for t in tasks)),acoustic_strata=dict(collections.Counter(t['acoustic_stratum'] for t in tasks)),semantic_content_strata='unknown; must be rebalanced after actual review',windows_may_overlap=True,not_independent_samples=True,human_reviews_completed=0,semantic_annotations_completed=0)
    (review/'distribution.json').write_text(json.dumps(distribution,indent=2))
    (review/'quality_metrics.json').write_text(json.dumps(dict(reviewed_windows=0,transcript_error_rate=None,event_precision=None,event_recall=None,boundary_error_distribution=None,inter_reviewer_agreement=None,teacher_calibration=None,semantic_unknown_rate=1.,occlusion_unknown_rate=1.,interpretation='unreviewed100%; unknown is not a measured model error rate'),indent=2))
    print(json.dumps(distribution,indent=2))
if __name__=='__main__':main()
