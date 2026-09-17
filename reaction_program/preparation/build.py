"""CPU-only STEP2-5. Never calls an API or trains a model. Exclusive output writes."""
import copy, hashlib, json, subprocess
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/reaction_program/milestone1_v1'
ANN=ROOT/'annotations/reaction_program_v1'
LEG=Path('/home/zhengshiyi/react2025')
SEM=ROOT/'runs/reaction_flow/semantic_supervision_full_train_v1'

def digest(x):return hashlib.sha256(x).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def lines(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:
        for x in rows:f.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n')
def uid(s):return digest(s.encode())[:20]

def freeze_evaluation():
    old=ROOT/'runs/reaction_flow/generator_reward_nft_full_test_v1/manifest.json'
    original=read(old); cache=read(old.parent/'targets.json')['files'];sources=copy.deepcopy(original['sources'])
    verified={};missing=[]
    for i,s in enumerate(sources):
        assert len(s['targets'])==10
        role,session,base=s['clip_id'].split('/');opp='listener' if role=='speaker' else 'speaker'
        expected=f'{opp}/{session}/{base}.npy'
        assert s['targets'][0]['path'].endswith(expected)
        for entry in list(s['files'].values())+s['targets']:
            entry['path']=str(ROOT/entry['path'].replace('data/test/','data/val/'))
            p=Path(entry['path'])
            if str(p) not in verified:
                actual=sha(p); assert actual==entry['sha256'],str(p)
                shape=list(np.load(p,mmap_mode='r').shape);verified[str(p)]=dict(sha256=actual,shape=shape)
            entry['shape']=verified[str(p)]['shape']
        assert all(t['path'].split('/facial-attributes/')[1].startswith(opp+'/'+session+'/') for t in s['targets'])
        assert all(v['shape'][0]==s['length'] for v in s['files'].values())
        assert len(s['frame_pts'])==s['length'] and np.all(np.diff(s['frame_pts'])>0)
        c=copy.deepcopy(cache[i]);c['path']=str(ROOT/c['path']);c['present']=Path(c['path']).exists()
        if c['present']:assert sha(c['path'])==c['sha256']
        else:missing.append(i)
        s['processed_target']=c;s['direction']='original_speaker' if role=='speaker' else 'reverse'
    metric_paths=[LEG/'framework/metrics'/x for x in ['FRC.py','FRD.py','S_MSE.py','FRVar.py','metric.py']]
    deps=metric_paths+[LEG/'dataset/react_2025.py',LEG/'framework/modules/post_processor.py',LEG/'configs/model/emotion_autoencoder.yaml',LEG/'pretrained_models/post_processor/checkpoint.pth']
    manifest=dict(version='reaction-program-eval-v1',split='val',population=1142,K=10,R=10,
        scope='full bidirectional local validation; local TEST is content-identical, NOT hidden confirmation',
        reference_policy='exact frozen inherited paired+9 membership/order/repeats; never sample in evaluation',
        source_manifest=dict(path=str(old),sha256=sha(old)),sources=sources,
        normalization=original['normalization'],processor=dict(seed_policy=original['new_processor_seed'],clip_length=1000,target_index_sha256=sha(old.parent/'targets.json'),missing_cache_indices=missing,cache_ready=not missing),
        implementation={str(p):sha(p) for p in deps},
        semantics='native legal source-only input strategy recorded per model; P2 NULL historical baseline separately identified',
        oracle_policy='GT codec/program diagnostics separately labelled; no GT-assisted source-only scores',
        model_policy='same manifest AND processed target hashes for A/B/C; K10, no best-of-extra/GT selection',
        precision='native model output -> FP32 unchanged official functions; no new AU rounding',
        full_metrics=['FRC','exact_FRD','S_MSE','temporal_S_MSE','FRVar'],historical_scores_not_new_EXP_A=True)
    write(OUT/'evaluation/manifest.json',manifest)
    write(OUT/'evaluation/VERIFIED_FILES.json',verified)
    dev=ROOT/'runs/phase24/evaluation_v1/multitarget_development_manifest.json'
    write(OUT/'evaluation/DEV80_DIAGNOSTIC_POINTER.json',dict(path=str(dev),sha256=sha(dev),scope='historical diagnostic only; distinct frozen references, no pooling with full1142'))
    write(OUT/'evaluation/LOCK.json',dict(manifest_sha256=sha(OUT/'evaluation/manifest.json'),K=10,R=10,population=1142,verified_unique_files=len(verified),missing_processed_targets=len(missing)))
    print('evaluation frozen',len(sources),'unique inputs',len(verified),'missing targets',len(missing),flush=True)

def weak_index():
    result=defaultdict(list)
    roots=[ROOT/'runs/reaction_flow/semantic_auto_weak_v1/cache',ROOT/'runs/reaction_flow/semantic_auto_weak_v1/expansion/weak/cache']
    for root in roots:
        for p in sorted(root.glob('rec_*.json')):
            x=read(p)
            if x.get('split')!='train' or x.get('visible_roles')!=['speaker']:continue
            for e in x['events']:result[(x['clip_id'],e['text'].strip())].append((e,p))
    return result

def build_candidates():
    folds=read(ROOT/'runs/reaction_reward/v1/SPLIT_MANIFEST.json')
    records={r['id']:r for r in folds['records']}
    mapping={r['id']:r['relative'] for r in read(SEM/'private_recording_map.json')}
    weak=weak_index();by_record=defaultdict(list)
    for line in (SEM/'review_candidates.jsonl').read_text().splitlines():
        c=json.loads(line);rel=mapping[c['clip_id']];r=records[rel]
        if r['split']!='RM_fit':continue
        text=(ROOT/'data/train/text/speaker'/(rel+'.txt')).read_text()
        if c['transcript_evidence'].strip() not in text:continue
        options=weak.get((c['clip_id'],c['transcript_evidence'].strip()),[])
        evidence=next(((e,p) for e,p in options if e['weak_input_eligible']),options[0] if options else None)
        by_record[rel].append((c,evidence))
    # One event per recording. Round robin sessions; prefer existing eligible evidence without using listener.
    sessions=defaultdict(list)
    for rel,cs in by_record.items():
        cs.sort(key=lambda t:(not bool(t[1] and t[1][0]['weak_input_eligible']),uid('123|'+t[0]['clip_id']+'|'+str(t[0]['candidate_index']))))
        sessions[records[rel]['session']].append((rel,*cs[0]))
    for s in sessions:sessions[s].sort(key=lambda t:(not bool(t[2] and t[2][0]['weak_input_eligible']),uid('123|'+t[0])))
    chosen=[];cursor=0
    while len(chosen)<600:
        added=0
        for s in sorted(sessions):
            if cursor<len(sessions[s]) and len(chosen)<600:chosen.append(sessions[s][cursor]);added+=1
        if not added:raise RuntimeError('fewer than 600 eligible TRAIN candidates')
        cursor+=1
    assets={};events=[];reactions=[];sp_requests=[];li_requests=[];private=[];trajectory_stats=[]
    def asset(path,role,kind,expected=None):
        path=Path(path);a='asset_'+uid(str(path));h=sha(path)
        if expected is not None:assert h==expected,str(path)
        assets[a]=dict(path=str(path),sha256=h,role=role,kind=kind,split='train')
        return a
    for rel,c,evidence in chosen:
        r=records[rel];clip=c['clip_id'];eid='evt_'+uid(clip+'|'+str(c['candidate_index']));oid='obs_'+uid(clip)
        private.append(dict(clip_id=clip,relative=rel,fold=r['split'],group=r['group'],session_id=r['session']))
        transcript_path=ROOT/'data/train/text/speaker'/(rel+'.txt');text=transcript_path.read_text();tx=asset(transcript_path,'speaker','transcript')
        source_assets={}
        for kind in ['audio','video-face-crop']:
            meta=r['files'][kind+'/speaker'];source_assets[kind]=asset(meta['path'],'speaker',kind,meta['sha256'])
        listener_meta=r['files']['video-face-crop/listener'];la=asset(listener_meta['path'],'listener','video-face-crop',listener_meta['sha256'])
        datarefs={}
        for kind in ['facial-attributes','coefficients']:
            meta=r['files'][kind+'/listener'];datarefs[kind]=asset(meta['path'],'listener',kind,meta['sha256'])
        e,p=evidence if evidence else (None,None)
        timing=None;frame=None;role='UNKNOWN';sync=None
        if e:
            timing=e['timing_evidence'];sync=e['sync_status'];role=e['role']
            if e['weak_input_eligible'] and sync['status']=='estimated' and sync.get('offset_s') is not None:
                a,b=timing['start_s'],timing['end_s'];lo,hi=sync['effective_audio_range_s'];pts=np.asarray(r['pts']['speaker'])
                if lo<=a<b<=hi:
                    a+=sync['offset_s'];b+=sync['offset_s'];end=pts[-1]+np.median(np.diff(pts))
                    if pts[0]<=a<b<=end:
                        left=max(0,int(np.searchsorted(pts,a,side='right'))-1);right=int(np.searchsorted(pts,b,side='left'))
                        if left<right<=len(pts):frame=[left,right]
        proposal=read(Path(c['provenance_result']))
        provenance=dict(raw_annotation_asset=str(Path(c['provenance_result']).parent),raw_result_sha256=sha(c['provenance_result']),model=proposal.get('model'),returned_model=proposal.get('returned_model'),model_revision='not supplied by original provider',prompt_sha256=proposal.get('prompt_sha256'),input_transcript_sha256=assets[tx]['sha256'],weak_evidence_sha256=sha(p) if p else None,weak_evidence_path=str(p) if p else None)
        ev=dict(schema_version='reaction-program-annotation-v1',clip_id=clip,session_id=r['session'],fold=r['split'],split='train',event_id=eid,status='candidate',
            start_frame=frame[0] if frame else None,end_frame=frame[1] if frame else None,anchor_frame=None,
            transcript=c['transcript_evidence'].strip(),transcript_span=[text.index(c['transcript_evidence'].strip()),text.index(c['transcript_evidence'].strip())+len(c['transcript_evidence'].strip())],
            event_type=[c['event_type']],prosody=None,visible_behavior=None,semantic_summary=c['description'],role_status=role,
            time_domain='speaker_video_pts_under_feature_row_assumption' if frame else 'unresolved',timing_evidence=timing,sync_evidence=sync,
            confidence=None,confidence_calibrated=False,human_reviewed=False,training_eligible=False,visible_roles=['speaker'],provenance=provenance)
        events.append(ev)
        react=dict(schema_version='reaction-program-annotation-v1',clip_id=clip,split='train',fold=r['split'],observation_id=oid,event_id=None,status='pending_observation',reaction_start=None,reaction_peak=None,reaction_end=None,actions=None,global_style=None,confidence=None,confidence_calibrated=False,human_reviewed=False,training_eligible=False,visible_roles=['listener'],numeric_assets=datarefs,video_asset=la,linked_candidate_ids=[eid],relation_status='UNKNOWN',time_domain='listener_native_pts',provenance=dict(split_manifest_sha256=sha(ROOT/'runs/reaction_reward/v1/SPLIT_MANIFEST.json') if not reactions else reactions[0]['provenance']['split_manifest_sha256']))
        reactions.append(react)
        # Payload whitelist excludes session/date/descriptive filenames, listener and targets.
        sp_requests.append(dict(task_id=eid,task='speaker_event',split='train',visible_roles=['speaker'],input=dict(transcript=text,evidence_quote=ev['transcript'],transcript_span=ev['transcript_span'],media_assets=list(source_assets.values()),frame_pts_asset='pts_'+uid(clip+'speaker'),proposed_frame_interval=frame,role_status=role,timing_evidence=timing,sync_evidence=sync),prompt='speaker_event.md',status='prepared_not_sent'))
        li_requests.append(dict(task_id=oid,task='listener_observation',split='train',visible_roles=['listener'],input=dict(video_asset=la,frame_pts_asset='pts_'+uid(clip+'listener'),numeric_assets=list(datarefs.values()),named_AU_mapping_available=False),prompt='listener_observation.md',status='prepared_not_sent'))
        for who in ['speaker','listener']:
            pts=r['pts'][who];pp=ANN/'train/pts'/('pts_'+uid(clip+who)+'.json');write(pp,dict(pts_seconds=pts,origin='ffprobe; historical inspected inventory',feature_row_mapping='count equality is an assumption, not sync verification'))
            assets[pp.stem]=dict(path=str(pp),sha256=sha(pp),role=who,kind='pts',split='train')
        y=np.load(r['files']['facial-attributes/listener']['path']);assert y.ndim==2 and y.shape[1]==25 and np.isfinite(y).all()
        trajectory_stats.append(dict(clip_id=clip,frames=len(y),AU_values=np.unique(y[:,:15]).tolist(),VA_min=y[:,15:17].min(0).tolist(),VA_max=y[:,15:17].max(0).tolist(),expression_sum_error=float(np.max(np.abs(y[:,17:].sum(-1)-1))),mean_abs_velocity=float(np.abs(np.diff(y,axis=0)).mean()) if len(y)>1 else None))
    lines(ANN/'train/speaker_events.jsonl',events);lines(ANN/'train/listener_reactions.jsonl',reactions)
    eq=[]
    buckets=defaultdict(list)
    for e in events:buckets[e['session_id']].append(e)
    # Candidate generation uses session only. No equivalence or support is asserted.
    for session,es in sorted(buckets.items()):
        for i in range(0,len(es)-1,2):
            a,b=es[i:i+2];eq.append(dict(schema_version='reaction-program-annotation-v1',split='train',fold='RM_fit',event_a=a['event_id'],event_b=b['event_id'],equivalence=None,status='UNKNOWN',reason=dict(same_intent=None,same_dialogue_stage=None,compatible_local_context=None),visible_roles=['speaker'],confidence_calibrated=False,human_reviewed=False,training_eligible=False,provenance=dict(candidate_rule='same-session proposal only; session excluded from annotator input')))
    lines(ANN/'train/event_equivalence.jsonl',eq)
    support=[dict(schema_version='reaction-program-annotation-v1',split='train',fold='RM_fit',event_id=e['event_id'],observed_support=[],LLM_inferred_support=[],status='UNKNOWN',training_eligible=False,missing_is_not_no_reaction=True) for e in events]
    lines(ANN/'train/reaction_support.jsonl',support)
    lines(ANN/'train/requests/speaker.jsonl',sp_requests);lines(ANN/'train/requests/listener.jsonl',li_requests)
    lines(ANN/'train/requests/equivalence.jsonl',[dict(task_id='eq_'+uid(e['event_a']+e['event_b']),task='speaker_event_equivalence',split='train',visible_roles=['speaker'],input=dict(event_a=e['event_a'],event_b=e['event_b'],event_outputs_required=True),prompt='event_equivalence.md',status='waiting_for_source_event_outputs') for e in eq])
    write(ANN/'train/ASSET_RESOLVER_PRIVATE.json',assets);write(ANN/'train/RECORDING_MAP_PRIVATE.json',private)
    write(OUT/'TRAIN_NUMERIC_AUDIT.json',trajectory_stats)
    write(ANN/'train/samples/real_pending_bundle.json',dict(speaker=events[0],listener=reactions[0],equivalence=eq[0],support=support[0],notice='real input provenance, pending annotations; no action or match invented'))
    for split in ['val','test']:
        for name in ['speaker_events','listener_reactions','event_equivalence','reaction_support']:lines(ANN/split/(name+'.jsonl'),[])
        write(ANN/split/'STATE.json',dict(status='empty_physically_isolated',listener_annotation_for_tuning=False))
    summary=dict(seed=123,count=len(events),recordings=len({e['clip_id'] for e in events}),sessions=dict(Counter(e['session_id'] for e in events)),groups=len({r['group'] for r in private}),source_pool_recordings=len(by_record),fold='RM_fit',role_supported=sum(e['role_status']=='source_speaker_candidate' for e in events),frame_timed_candidates=sum(e['start_frame'] is not None for e in events),unknown_role=sum(e['role_status'] in ['UNKNOWN','unknown'] for e in events),pending_listener_observations=len(reactions),confirmed_listener_observations=0,event_equivalence_candidates=len(eq),confirmed_event_equivalences=0,observed_support=0,API_calls=0,human_reviewed=0,training_eligible=0,selection='one source-transcript event per recording; round-robin sessions; prefer pre-existing eligible source-only evidence, then fixed SHA256(seed123); no listener value used for selection')
    write(OUT/'CANDIDATE_SUMMARY.json',summary);print(summary,flush=True)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    freeze_evaluation();build_candidates()
    write(OUT/'BUILD_COMPLETE.json',dict(git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),steps_completed=[2,4],training_started=False))
if __name__=='__main__':main()
