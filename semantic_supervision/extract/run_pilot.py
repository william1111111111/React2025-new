"""Run independent CPU role stages with read-only media and no network/GPU mounts."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
CODE=REPO/'semantic_supervision'
OUT=REPO/'runs/reaction_flow/semantic_supervision_pilot_v1'
DATA=(REPO/'data/train').resolve()

def sandbox(role,module,mounts):
    dest=OUT/role;dest.mkdir(exist_ok=True)
    venv=(REPO/'.venv').resolve();pythonhome=Path(os.path.realpath(sys.executable)).parent.parent
    cmd=['/usr/bin/bwrap','--unshare-all','--die-with-parent','--new-session']
    for key,value in dict(PATH='/venv/bin:/usr/bin:/bin',HOME='/tmp',PYTHONPATH='/code',PYTHONNOUSERSITE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1').items():cmd+=['--setenv',key,value]
    for p in ['/usr','/lib','/lib64']:
        if Path(p).exists():cmd+=['--ro-bind',p,p]
    cmd+=['--ro-bind',str(pythonhome),str(pythonhome),'--ro-bind',str(venv),'/venv','--ro-bind',str(CODE),'/code/semantic_supervision','--ro-bind',str(OUT/'jobs'/role),'/job','--bind',str(dest),'/outputs','--tmpfs','/tmp','--proc','/proc','--dev','/dev','--chdir','/code']
    for host,guest in mounts:cmd+=['--ro-bind',str(host),guest]
    # Real permission probe: host data/other role and NVIDIA devices unavailable.
    probe="import pathlib,json; assert not pathlib.Path('/home/zhengshiyi/react/data').exists(); assert not list(pathlib.Path('/dev').glob('nvidia*')); assert not pathlib.Path('/inputs/forbidden').exists(); print('ROLE_SANDBOX_OK')"
    subprocess.run(cmd+['--','/venv/bin/python','-c',probe],check=True,env={})
    (dest/'sandbox_command.json').write_text(json.dumps(cmd+['--','/venv/bin/python','-m',module],indent=2))
    with (dest/'execution.log').open('x') as f:subprocess.run(cmd+['--','/venv/bin/python','-m',module],stdout=f,stderr=subprocess.STDOUT,check=True,env={})

def main():
    pilot=json.loads((REPO/'runs/reaction_flow/semantic_supervision_preflight_v1/pilot_media_metadata.json').read_text());mapping=[]
    for r in pilot:
        key='rec_'+hashlib.sha256(r['clip_id'].encode()).hexdigest()[:16]
        mapping.append(dict(id=key,original_clip_id=r['clip_id'],relative=r['clip_id'].removeprefix('speaker/'),session=r['session'],speaker_frames=r['checks']['speaker_video_frames'] if 'speaker_video_frames' in r['checks'] else r['checks']['video_frames'],listener_frames=r['checks']['listener_video_frames']))
    (OUT/'private_recording_map.json').write_text(json.dumps(mapping,indent=2))
    for role,field,prompt in [('speaker','speaker_frames','speaker_events.txt'),('listener','listener_frames','listener_observations.txt'),('relation',None,'paired_event_links.txt')]:
        folder=OUT/'jobs'/role;folder.mkdir(parents=True,exist_ok=True)
        rows=[{k:r[k] for k in ['id','relative']+([field] if field else [])} for r in mapping]
        (folder/'job.json').write_text(json.dumps(dict(records=rows,prompt_sha256=hashlib.sha256((CODE/'vendor/react_semantic_supervision/prompts'/prompt).read_bytes()).hexdigest()),indent=2))
    sandbox('speaker','semantic_supervision.extract.source',[(DATA/'audio/speaker','/inputs/audio'),(DATA/'video-face-crop/speaker','/inputs/video'),(DATA/'text/speaker','/inputs/text')])
    sandbox('listener','semantic_supervision.extract.listener',[(DATA/'facial-attributes/listener','/inputs/attributes'),(DATA/'video-face-crop/listener','/inputs/video')])
    sandbox('relation','semantic_supervision.align.paired',[(OUT/'speaker','/inputs/source'),(OUT/'listener','/inputs/listener')])
    print('ALL ROLE EXTRACTION STAGES COMPLETED',flush=True)
if __name__=='__main__':main()
