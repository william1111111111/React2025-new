"""Local TRAIN-only inventory and small media timeline preflight; no annotation."""
import collections,csv,hashlib,json,subprocess,wave
from pathlib import Path
import numpy as np
ROOT=Path('data/train').resolve();OUT=Path(__file__).resolve().parent

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rolepath(relative,folder,role,suffix):
    return ROOT/folder/role/relative.parent/relative.with_suffix(suffix).name
speaker=sorted((ROOT/'facial-attributes/speaker').glob('*/*.npy'));inventory=[];by_session=collections.defaultdict(list)
for p in speaker:
    rel=p.relative_to(ROOT/'facial-attributes/speaker');by_session[rel.parts[0]].append(rel)
    row=dict(clip_id='speaker/'+rel.with_suffix('').as_posix(),split='train',session=rel.parts[0])
    for name,folder,role,suffix in [('audio','audio','speaker','.wav'),('video','video-face-crop','speaker','.mp4'),('transcript','text','speaker','.txt'),('speaker_attributes','facial-attributes','speaker','.npy'),('audio_features','audio-features','speaker','.npy'),('coefficients','coefficients','speaker','.npy'),('listener_attributes','facial-attributes','listener','.npy'),('listener_video','video-face-crop','listener','.mp4')]:
        q=rolepath(rel,folder,role,suffix);row[name+'_path']=str(q);row[name+'_exists']=q.is_file()
    inventory.append(row)
with (OUT/'train_inventory.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=inventory[0],lineterminator='\n');w.writeheader();w.writerows(inventory)
# First two lexical recordings per session; metadata-only sample, not200 events.
pilot=[]
for session,rels in sorted(by_session.items()):
    for rel in rels[:2]:
        clip='speaker/'+rel.with_suffix('').as_posix();row=next(x for x in inventory if x['clip_id']==clip)
        q=dict(clip_id=clip,session=session,split='train',status='unannotated',speaker_only_inputs={},training_only_listener_inputs={},checks={})
        for name in ['audio','video','transcript']:
            p=Path(row[name+'_path']);q['speaker_only_inputs'][name]=dict(path=str(p),sha256=digest(p))
        for name in ['listener_attributes','listener_video']:
            p=Path(row[name+'_path']);q['training_only_listener_inputs'][name]=dict(path=str(p),sha256=digest(p))
        for name in ['speaker_attributes','audio_features','coefficients','listener_attributes']:
            p=Path(row[name+'_path']);a=np.load(p,mmap_mode='r',allow_pickle=False);q['checks'][name+'_shape']=list(a.shape)
        with wave.open(row['audio_path'],'rb') as f:
            q['checks'].update(audio_channels=f.getnchannels(),audio_rate=f.getframerate(),audio_samples=f.getnframes(),audio_seconds=f.getnframes()/f.getframerate())
        for name in ['video','listener_video']:
            output=subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=codec_type,r_frame_rate,avg_frame_rate,time_base,start_time,duration,nb_frames,sample_rate,channels:format=start_time,duration','-of','json',row[name+'_path']],text=True)
            meta=json.loads(output);q['checks'][name+'_probe']=meta
            stream=next(s for s in meta['streams'] if s['codec_type']=='video')
            frames=int(stream['nb_frames']) if stream.get('nb_frames','N/A')!='N/A' else None
            q['checks'][name+'_frames']=frames
        q['checks']['speaker_video_attribute_count_equal']=q['checks']['video_frames']==q['checks']['speaker_attributes_shape'][0]
        q['checks']['listener_video_attribute_count_equal']=q['checks']['listener_video_frames']==q['checks']['listener_attributes_shape'][0]
        text=Path(row['transcript_path']).read_text();q['checks']['transcript_characters']=len(text.strip())
        q['checks']['transcript_timestamp_marker_detected']=('-->' in text or any(t in text for t in ['"start"','"end"']))
        q['checks']['frame_count_match_is_not_sync_proof']=True
        pilot.append(q)
(OUT/'pilot_media_metadata.json').write_text(json.dumps(pilot,indent=2))
# Separate allowlists: speaker event stage cannot read listener assets; linkage
# is a separate TRAIN-only stage that references human/teacher-reviewed events.
with (OUT/'speaker_stage_inputs.jsonl').open('w') as f:
    for p in pilot:f.write(json.dumps(dict(clip_id=p['clip_id'],split='train',allowed_inputs=p['speaker_only_inputs'],annotation_status='not_run',temporal_alignment_status='unverified'))+'\n')
with (OUT/'listener_stage_inputs.jsonl').open('w') as f:
    for p in pilot:f.write(json.dumps(dict(clip_id=p['clip_id'].replace('speaker/','listener/',1),split='train',allowed_inputs=p['training_only_listener_inputs'],annotation_status='not_run',inference_allowed=False))+'\n')
with (OUT/'paired_stage_inputs.jsonl').open('w') as f:
    for p in pilot:f.write(json.dumps(dict(split='train',speaker_id=p['clip_id'],listener_id=p['clip_id'].replace('speaker/','listener/',1),status='await_verified_events_and_timeline',cross_recording_match='unknown',inference_allowed=False))+'\n')
summary=dict(train_records=len(speaker),sessions=len(by_session),missing={k:sum(not row[k] for row in inventory) for k in inventory[0] if k.endswith('_exists')},pilot_recordings=len(pilot),selection='first two lexicographic TRAIN recordings per session; metadata only, not200 annotated events',pilot_speaker_frame_matches=sum(p['checks']['speaker_video_attribute_count_equal'] for p in pilot),pilot_listener_frame_matches=sum(p['checks']['listener_video_attribute_count_equal'] for p in pilot),pilot_timestamp_marker_count=sum(p['checks']['transcript_timestamp_marker_detected'] for p in pilot),actual_annotations=0,model_or_API_calls=0,gpu_used=False,license_status='signed data agreement and model-use terms not supplied; no external upload',HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
(OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
