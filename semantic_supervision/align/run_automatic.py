"""Prepare 250 existing candidates, then isolate local source-only teachers."""
import json,os,sys,hashlib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/reaction_flow/semantic_alignment_pilot_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    OUT.mkdir(exist_ok=True);jobdir=OUT/'job';jobdir.mkdir(exist_ok=True);dest=OUT/'source';dest.mkdir(exist_ok=True)
    selected=json.loads((ROOT/'runs/reaction_flow/semantic_supervision_full_train_v1/selected_records.json').read_text())
    pilot=json.loads((ROOT/'runs/reaction_flow/semantic_supervision_pilot_v1/private_recording_map.json').read_text())
    rows=[]
    for r in pilot:
        if r['id'] not in selected:continue
        item=selected[r['id']];candidates=json.loads((Path(item['folder'])/'candidates.json').read_text())['candidates']
        rows.append({'id':r['id'],'relative':r['relative'],'transcript_sha256':item['result']['transcript_sha256'],'candidates':candidates})
    assert len(rows)==33 and sum(len(r['candidates']) for r in rows)==250
    cache=Path.home()/'.cache/react2025';torchcache=Path.home()/'.cache/torch'
    teachers={'ctc_model':'torchaudio.WAV2VEC2_ASR_BASE_960H','ctc_weight_sha256':sha(torchcache/'hub/checkpoints/wav2vec2_fairseq_base_ls960_asr_ls960.pth'),'asd_model':'TalkNet-ASD pretrained TalkSet','asd_weight_sha256':sha(cache/'pretrain_TalkSet.model'),'asd_code_commit':subprocess.check_output(['git','-C',str(cache/'TalkNet-ASD'),'rev-parse','HEAD']).decode().strip(),'asd_weight_url':'https://huggingface.co/AlekseyKorshuk/talknet-asd/resolve/main/pretrain_TalkSet.model'}
    (jobdir/'job.json').write_text(json.dumps({'records':rows,'teachers':teachers},indent=2))
    policy={'scope':'250 existing events, 33 TRAIN recordings, 17 sessions','human_review':'omitted_by_user','timing_ctc_mean_min':.5,'greedy_asr_agreement_min':.6,'max_word_duration_s':2,'digits_in_evidence':'timing_uncertain','active_speech_fraction_min':.8,'shift_3s_logit_margin_min':.5,'calibrated':False,'auto_candidates_are_not_verified':True,'training_eligible':False,'gpu_physical_index':5}
    (OUT/'POLICY.json').write_text(json.dumps(policy,indent=2))
    cmd=['/usr/bin/bwrap','--unshare-all','--die-with-parent','--new-session']
    for k,v in {'PATH':'/venv/bin:/usr/bin:/bin','HOME':'/tmp','TORCH_HOME':'/torchcache','PYTHONPATH':'/code','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':'GPU-73af75e6-5d60-302d-0a7c-256f693b5143','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1'}.items():cmd+=['--setenv',k,v]
    for p in ['/usr','/lib','/lib64','/sys','/etc/ld.so.cache']:
        if Path(p).exists():cmd+=['--ro-bind',p,p]
    ph=Path(os.path.realpath(sys.executable)).parent.parent
    cmd+=['--ro-bind',str(ph),str(ph),'--ro-bind',str((ROOT/'.venv').resolve()),'/venv','--ro-bind',str(ROOT/'semantic_supervision'),'/code/semantic_supervision','--ro-bind',str(jobdir),'/job','--bind',str(dest),'/outputs','--tmpfs','/tmp','--proc','/proc','--dev','/dev','--chdir','/code']
    for d in ['/dev/nvidia5','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools']:
        if Path(d).exists():cmd+=['--dev-bind',d,d]
    data=(ROOT/'data/train').resolve()
    for h,g in [(data/'audio/speaker','/inputs/audio'),(data/'video-face-crop/speaker','/inputs/video'),(data/'text/speaker','/inputs/text'),(torchcache,'/torchcache'),(cache/'TalkNet-ASD','/teachers/TalkNet-ASD'),(cache/'pretrain_TalkSet.model','/teachers/pretrain_TalkSet.model'),(cache/'python','/teachers/python')]:cmd+=['--ro-bind',str(h),g]
    cmd+=['--','/venv/bin/python','-m','semantic_supervision.align.automatic']
    (OUT/'sandbox_command.json').write_text(json.dumps(cmd,indent=2))
    with (OUT/'execution.log').open('a') as log:subprocess.run(cmd,env={},stdout=log,stderr=subprocess.STDOUT,check=True)
if __name__=='__main__':main()
