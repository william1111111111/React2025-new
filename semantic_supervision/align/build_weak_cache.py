import json,hashlib,collections
from pathlib import Path
from .recompute_support import ROOT,OLD,OUT,sha

def main():
    dest=OUT/'cache';dest.mkdir(exist_ok=True);support=json.loads((OUT/'support_events.json').read_text());index={}
    for cid in sorted({x['clip_id'] for x in support}):
        record=json.loads((OLD/'source'/cid/'result.json').read_text());lagfile=OUT/'lag'/(cid+'.json');lag=json.loads(lagfile.read_text()) if lagfile.exists() else {}
        sync={'status':lag.get('sync_status','unresolved'),'offset_s':lag.get('offset_s'),'audio_origin_s':lag.get('audio_origin_s',0.),'video_origin_s':record['video_timeline']['first_pts'],'origin_source':'WAV decoded samples and ffprobe video PTS','offset_source':'bounded source-only ASD lag probe' if lag else 'unresolved','effective_audio_range_s':lag.get('effective_audio_range_s',[0,0]),'offset_interval_s':lag.get('offset_interval_s'),'sync_verified':False}
        events=[]
        for x in support:
            if x['clip_id']!=cid:continue
            e=x['event'];rules={**x['timing_rules'],**x['support']['rules']}
            events.append({'event_id':e['event_id'],'text':e['transcript_evidence'],'event_type':e['event_type'],'role':'source_speaker_candidate' if x['joint_passed'] else 'unknown','role_evidence':x['support'],'timing_evidence':{'start_s':e['start_s'],'end_s':e['end_s'],'ctc_mean_score':e['ctc_mean_score'],'greedy_agreement':e['greedy_asr_char_agreement'],'independent_verification':False},'sync_status':sync,'time_domain':'speaker_audio_sample_clock','weak_input_eligible':x['joint_passed'],'rules':rules,'rejection_reasons':x['rejection_reasons'],'source_version_hashes':[x['source_record_sha256'],x['scores_sha256'],x['words_sha256']]+([sha(lagfile)] if lagfile.exists() else [])})
        obj={'schema_version':'auto-weak-v1','annotation_regime':'auto_weak','clip_id':cid,'split':record.get('split','train'),'human_reviewed':False,'training_eligible':False,'visible_roles':['speaker'],'input_hashes':[v['sha256'] for v in record['actual_visible_inputs']],'teacher_hashes':[v for k,v in record['teacher_provenance'].items() if 'sha256' in k or 'commit' in k],'policy_sha256':sha(OUT/'POLICY.json'),'code_hashes':[sha(Path(__file__)),sha(ROOT/'semantic_supervision/align/support.py'),sha(ROOT/'semantic_supervision/align/lag_probe.py'),sha(OUT/'lag_policy.json')],'events':events}
        p=dest/(cid+'.json');p.write_text(json.dumps(obj,ensure_ascii=False,indent=2));index[cid]={'cache_sha256':sha(p),'media_hashes':obj['input_hashes'],'policy_sha256':obj['policy_sha256']}
    (dest/'index.json').write_text(json.dumps(index,indent=2));print('cache records',len(index))
if __name__=='__main__':main()
