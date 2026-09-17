"""Freeze preparation provenance and the STEP1-5 handoff; run after validation."""
from .build import *

def main():
    channels=[]
    for i in range(25):
        group='AU_occurrence' if i<15 else 'VA' if i<17 else 'expression_probability'
        label=f'AU_channel_{i:02d}_identity_unresolved' if i<15 else ('valence_document_order' if i==15 else 'arousal_document_order' if i==16 else f'expression_channel_{i-17:02d}_identity_unresolved')
        channels.append(dict(index=i,group=group,label=label,exact_extractor_mapping_verified=False))
    coeff=[dict(index=i,group='expression_basis' if i<52 else 'rotation' if i<55 else 'translation',component=i if i<52 else i-52 if i<55 else i-55,anatomical_name=None,units_verified=False) for i in range(58)]
    if not (OUT/'CHANNELS.json').exists():write(OUT/'CHANNELS.json',dict(facial_attributes=channels,coefficients=coeff,source_evidence=[str(LEG/'README.md'),str(LEG/'framework/utils/losses.py'),str(LEG/'external/FaceVerse/FaceVerseModel.py')]))
    assets=read(ANN/'train/ASSET_RESOLVER_PRIVATE.json');receipts=[]
    for name in ['speaker','listener','equivalence']:
        for line in (ANN/'train/requests'/(name+'.jsonl')).read_text().splitlines():
            r=json.loads(line);ids=[]
            def collect(x):
                if isinstance(x,dict):
                    for v in x.values():collect(v)
                elif isinstance(x,list):
                    for v in x:collect(v)
                elif isinstance(x,str) and x in assets:ids.append(x)
            collect(r['input']);prompt=ANN/'prompts'/r['prompt']
            receipts.append(dict(task_id=r['task_id'],request_sha256=digest(json.dumps(r,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()),prompt_sha256=sha(prompt),input_asset_hashes={i:assets[i]['sha256'] for i in ids},requested_model=None,model_revision=None,status='prepared_not_sent',no_API_call=True))
    rp=ANN/'train/requests/PROVENANCE_ENVELOPES.jsonl'
    if not rp.exists():lines(rp,receipts)
    else:assert [json.loads(x) for x in rp.read_text().splitlines()]==receipts
    stats=read(OUT/'TRAIN_NUMERIC_AUDIT.json')
    numeric=dict(recordings=len(stats),all_AU_binary=all(set(r['AU_values'])<={0,1} for r in stats),VA_min=np.array([r['VA_min'] for r in stats]).min(0).tolist(),VA_max=np.array([r['VA_max'] for r in stats]).max(0).tolist(),max_expression_sum_error=max(r['expression_sum_error'] for r in stats),frame_range=[min(r['frames'] for r in stats),max(r['frames'] for r in stats)])
    if not (OUT/'TRAIN_NUMERIC_SUMMARY.json').exists():write(OUT/'TRAIN_NUMERIC_SUMMARY.json',numeric)
    evidence=[LEG/p for p in ['dataset/react_2025.py','framework/metrics/FRC.py','framework/metrics/FRD.py','framework/metrics/S_MSE.py','framework/metrics/FRVar.py','framework/modules/post_processor.py','framework/utils/losses.py','external/FaceVerse/FaceVerseModel.py','README.md','DATASET_GT_CONSTRUCTION_AUDIT.md','reaction_space/model.py','reaction_space/discrete_code.py','discuss/methods/ANCHOR_AND_DIVERSITY_BUDGET_ABLATIONS_20260901.md','innovation_runs/20260725_sacrt_gate0/76d32c98ca30b7b2/GATE0_DECISION.md']]
    evidence += [ROOT/p for p in ['hirp/README.md','hirp/config.py','hirp/phase22.py','hirp/model/hirp_net.py','mam_target/data.py','mam_target/model.py','mam_staged/train.py','reaction_flow/data.py','reaction_flow/config.py','reaction_flow/condition_encoder.py','reaction_flow/velocity_model.py','semantic_supervision/bert_experiments/train.py','semantic_supervision/bert_experiments/model.py','semantic_supervision/models/auto_weak.py','mode_supervision/README.md','hirp/setup_phase24.py','hirp/task_phase24.py','runs/mam_target/staged_v2/analysis/final_verified.json','runs/reaction_flow/bert_semantic_v1/RESULTS.json','runs/reaction_flow/generator_reward_nft_full_test_v1/COMPLETE_RESULTS.json','runs/reaction_flow/feature_audio_pilot_pts_v1/REPORT.md','runs/reaction_reward/vector_selection_v1/RESULTS.md']]
    write(OUT/'RECON_EVIDENCE.json',dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),read_key_files={str(p):sha(p) for p in evidence},unavailable_requested_paths=['discuss/diversity','discuss/sacrt'],input_request=dict(path='/home/zhengshiyi/.codex/attachments/134c2c93-0efb-4076-830a-01ea6d497d84/pasted-text.txt',sha256=sha('/home/zhengshiyi/.codex/attachments/134c2c93-0efb-4076-830a-01ea6d497d84/pasted-text.txt'))))
    baseline=ROOT/'runs/reaction_flow/bert_semantic_v1/training/P2-bert/checkpoints/step_001000.pt'
    write(OUT/'BASELINE_LOCK.json',dict(primary=dict(name='P2-bert-step1000',path=str(baseline),sha256=sha(baseline)),quality_reference=read(ROOT/'runs/mam_target/staged_v2/analysis/final_verified.json')[0],new_evaluation_performed=False,reason='P2 fixed recent generator, S0 retained as higher-FRC DEV80 reference; no single Pareto-dominant best'))
    files=[p for p in ANN.rglob('*') if p.is_file()]+[p for p in OUT.rglob('*') if p.is_file() and p.name!='build.log']+[ROOT/'METHOD_RECON.md']+[p for p in (ROOT/'reaction_program').rglob('*.py')]
    write(OUT/'ARTIFACTS.json',dict(files={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in files},scope='STEP1-5; no new generation, annotation call or training'))
if __name__=='__main__':main()
