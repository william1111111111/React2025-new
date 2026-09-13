"""Full TRAIN text annotation with immutable attempts and bounded concurrency."""
import concurrent.futures,csv,fcntl,hashlib,json,shutil,time
from pathlib import Path
from . import batch_text as b
ROOT=b.ROOT
OUT=ROOT/'runs/reaction_flow/semantic_supervision_full_train_v1'
PILOT=b.OUT
BASE_PROMPT=b.PROMPT
REPAIR='严格检查后再输出：每项完整保留全部7个字段，包括null字段。最多16个候选，选择主要事件即可。引用必须是原文连续子串，逐字保留空格、大小写和标点，不用省略号拼接。'
def main():
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    mapping=json.loads((ROOT/'runs/reaction_flow/semantic_supervision_pilot_v1/private_recording_map.json').read_text())
    known={r['relative']:r['id'] for r in mapping}
    rows=[]
    for r in csv.DictReader((ROOT/'runs/reaction_flow/semantic_supervision_preflight_v1/train_inventory.csv').open()):
        assert r['split']=='train' and r['clip_id'].startswith('speaker/')
        rel=r['clip_id'].removeprefix('speaker/')
        assert '..' not in Path(rel).parts and not Path(rel).is_absolute()
        source=ROOT/'data/train/text/speaker'/(rel+'.txt')
        assert source.is_file()
        rows.append({'id':known.get(rel,'rec_'+hashlib.sha256(rel.encode()).hexdigest()[:16]),'relative':rel})
    assert len(rows)==1660 and len({r['id'] for r in rows})==1660
    b.write(OUT/'private_recording_map.json',rows)
    selected={};attempts=[];reused=[];start=time.monotonic();concurrency=16
    for row in rows:
        for folder in [PILOT/'repair_attempt_001'/row['id'],PILOT/row['id']]:
            f=folder/'result.json'
            if not f.exists():continue
            r=json.loads(f.read_text())
            if r['status']=='checks_passed' and r['transcript_sha256']==hashlib.sha256((ROOT/'data/train/text/speaker'/(row['relative']+'.txt')).read_bytes()).hexdigest():
                selected[row['id']]={'result':r,'folder':str(folder),'reused':True};reused.append(row['id']);break
    def snapshot(stage):
        totals={k:sum(r.get('usage',{}).get(k,0) for r in attempts) for k in ['prompt_tokens','completion_tokens','total_tokens']}
        data={'stage':stage,'total_records':1660,'reused_records':len(reused),'checks_passed':len(selected),'pending_records':1660-len(selected),'attempts_recorded':len(attempts),'requests_with_usage':sum(bool(r.get('usage')) for r in attempts),'usage_new_attempts':totals,'concurrency':concurrency,'elapsed_s':round(time.monotonic()-start,1),'training_eligible':0,'human_reviewed':0,'updated_unix':time.time()}
        b.write(OUT/'monitor_latest.json',data);print(json.dumps(data),flush=True)
    snapshot('starting')
    for attempt in range(3):
        b.OUT=OUT/f'attempt_{attempt:03d}';b.OUT.mkdir(exist_ok=True)
        b.PROMPT=BASE_PROMPT+(REPAIR if attempt else '')
        (b.OUT/'prompt.txt').write_text(b.PROMPT)
        pending=[r for r in rows if r['id'] not in selected]
        for offset in range(0,len(pending),16):
            wave=pending[offset:offset+16]
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                results=list(pool.map(b.one,wave))
            for r in results:
                attempts.append(r)
                if r['status']=='checks_passed':selected[r['clip_id']]={'result':r,'folder':str(b.OUT/r['clip_id']),'reused':False}
            network_failures=sum(r['status']=='request_failed' for r in results)
            if network_failures:concurrency=max(2,concurrency//2)
            snapshot(f'attempt_{attempt:03d}')
            if network_failures>=max(3,len(results)//2):
                snapshot('paused_provider_failures');return
        # Only explicit validation failures are repaired, never ambiguous network requests.
        blocked={r['clip_id'] for r in attempts if r['status'] in ['request_failed','previous_attempt_unresolved','input_rejected']}
        rows_for_retry=[r for r in rows if r['id'] not in selected and r['id'] not in blocked]
        if not rows_for_retry:break
        # Preserve original rows for final accounting, but skip blocked IDs on later rounds.
        for cid in blocked:
            pass
        if blocked:
            snapshot('paused_unresolved_requests');break
    output=[]
    for cid,item in sorted(selected.items()):
        obj=json.loads((Path(item['folder'])/'candidates.json').read_text())
        for i,c in enumerate(obj['candidates']):output.append({'clip_id':cid,'candidate_index':i,**c,'provenance_result':item['folder']+'/result.json','review_status':'pending','training_eligible':False})
    with (OUT/'review_candidates.jsonl').open('w') as f:
        for row in output:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    b.write(OUT/'selected_records.json',selected)
    b.write(OUT/'pending_review.json',[r for r in rows if r['id'] not in selected])
    snapshot('finished_with_pending_review' if len(selected)<1660 else 'finished')
    summary=json.loads((OUT/'monitor_latest.json').read_text());summary['candidate_count']=len(output)
    b.write(OUT/'FINAL_SUMMARY.json',summary)
    (OUT/'REPORT.md').write_text('# 全量 TRAIN 文本标注\n\nDeepSeek V4 Flash，Yunwu，关闭思考。只发送现有speaker转录，不上传音视频或listener数据。复用已通过检查的试点记录，每条最多3次有效输出尝试，保留全部版本。\n\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n\n自动检查不代表语义正确。时间与说话者归属未知，人工核验0，未用于训练。pending_review.json列出仍待处理记录；新请求用量不含复用试点的历史费用。\n')
if __name__=='__main__':main()
