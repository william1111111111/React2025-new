"""Source-only full Development-80 export, unchanged local REACT attribute metrics."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from .phase22 import load_checkpoint
from .phase23_long import export_full
from .phase23_cache import cached,save_cached,complete
from .phase15_audit import sha256_file,canonical_hash
from .train_phase21 import configure
from .train_phase22 import write

LEGACY=Path('/home/zhengshiyi/react2025')
ROOT=Path('runs/phase23/tradeoff_v1')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--device',default='cuda:7');p.add_argument('--candidate-chunk',type=int,default=10);a=p.parse_args();configure()
    sys.path.insert(0,str(LEGACY))
    from framework.modules.post_processor import Processor
    from framework.utils.compute_metrics import compute_metrics
    root=a.root;out=root/'task_development_full';out.mkdir(exist_ok=True)
    plan=json.loads(Path('runs/phase22/replicated_v1/evaluation_plan/manifest.json').read_text())
    mean=torch.from_numpy(np.load('external/FaceVerse/mean_face.npy')).float();std=torch.from_numpy(np.load('external/FaceVerse/std_face.npy')).float().clamp_min(1e-8)
    noise=torch.from_numpy(np.load(plan['banks'][0]['path']))[0,:10].to(a.device)
    source_rows=[]
    for clip in plan['clip_ids']:
        files={key:Path('data/val')/directory/(clip+'.npy') for key,directory in [('speaker_audio','audio-features'),('speaker_emotion','facial-attributes'),('speaker_3dmm','coefficients')]}
        sizes={key:int(np.load(path,mmap_mode='r').shape[0]) for key,path in files.items()}
        if len(set(sizes.values()))!=1:raise ValueError('full-source modalities differ; explicit new protocol required: '+str((clip,sizes)))
        target=Path('data/val/facial-attributes')/(clip.replace('speaker/','listener/')+'.npy')
        source_rows.append(dict(clip_id=clip,length=sizes['speaker_emotion'],modality_lengths=sizes,files={key:dict(path=str(path),sha256=sha256_file(path)) for key,path in files.items()},target=dict(path=str(target),sha256=sha256_file(target),length=len(np.load(target,mmap_mode='r')))))
    versions={str(path):sha256_file(path) for path in [LEGACY/'framework/modules/post_processor.py',LEGACY/'framework/modules/emotion_autoencoder.py',LEGACY/'configs/model/emotion_autoencoder.yaml',LEGACY/'pretrained_models/post_processor/checkpoint.pth',LEGACY/'framework/utils/compute_metrics.py',*sorted((LEGACY/'framework/metrics').glob('*.py')),Path(__file__),Path('hirp/phase23_long.py')]}
    identity=dict(version='phase23-development-full-v1',sources=source_rows,noise_sha256=plan['banks'][0]['sha256'],noise_indices=list(range(10)),K=10,chunk_T=750,temporal_rule='official pad to next 750 including zero tail, process valid chunks, concatenate all valid frames; fixed global noise across chunks',processor_clip_len_test=1000,processor_seed=23009,versions=versions,normalization=plan['normalization'],population='Development-80, paired listener only, not official ten-GT benchmark',metrics=['FRC','FRVar','smse','temporal_smse','TLCC'])
    manifest=out/'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text())!=identity:raise ValueError('task identity changed; use new root')
    else:write(manifest,identity)
    targets_dir=out/'processed_targets';targets_dir.mkdir(exist_ok=True)
    target_manifest=targets_dir/'completed.json';targets=[]
    if target_manifest.exists():
        record=json.loads(target_manifest.read_text());assert record['identity']==canonical_hash(identity)
        for row in record['files']:
            assert sha256_file(row['path'])==row['sha256'];targets.append(torch.from_numpy(np.load(row['path'])))
    else:
        torch.manual_seed(23009);np.random.seed(23009)
        processor=Processor(cfg_dir=str(LEGACY),ckpt_dir=str(LEGACY/'pretrained_models/post_processor'),device=torch.device(a.device),clip_len_test=1000,num_preds=10)
        rows=[]
        for i,s in enumerate(source_rows):
            target=torch.from_numpy(np.load(s['target']['path'])).float()
            with torch.no_grad():processed=processor.forward([torch.zeros(10,s['length'],25)],[[target]])[0]
            assert processed.shape==(1,s['length'],25)
            path=targets_dir/f'{i:03d}.npy'
            with path.open('xb') as f:np.save(f,processed.numpy())
            rows.append(dict(path=str(path),sha256=sha256_file(path)));targets.append(processed)
        write(target_manifest,dict(identity=canonical_hash(identity),files=rows));del processor;torch.cuda.empty_cache()
    completed=[]
    for point in json.loads((root/'reused_checkpoints.json').read_text()):
        label=f"seed{point['seed']}_{point['arm']}";cp=point['checkpoint'];assert sha256_file(cp['path'])==cp['sha256']
        model,meta=load_checkpoint(cp['path'],a.device);case_identity=dict(protocol=identity,checkpoint_sha256=cp['sha256'],global_step=meta['global_step'],seed=point['seed'],arm=point['arm'],prior_mode=meta['prior_mode'],output_config=meta['output_config'])
        result_path=out/f'{label}.json';prior=cached(result_path,case_identity)
        if prior is not None:print('REUSED TASK',label,flush=True);completed.append(dict(path=str(result_path),sha256=sha256_file(result_path)));continue
        exportdir=out/'exports'/label;exportdir.mkdir(parents=True,exist_ok=True);predictions=[];speakers=[];exports=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats(a.device)
        for i,s in enumerate(source_rows):
            path=exportdir/f'{i:03d}.npy';sidecar=exportdir/f'{i:03d}.json';item_identity=dict(case_sha256=canonical_hash(case_identity),source=s)
            if sidecar.exists():
                item=json.loads(sidecar.read_text());assert item['identity']==item_identity and sha256_file(path)==item['sha256'];pred=torch.from_numpy(np.load(path))
            else:
                streams={key:torch.from_numpy(np.load(value['path'])).float() for key,value in s['files'].items()}
                face=streams['speaker_3dmm'];face=face[:,0] if face.ndim==3 else face;streams['speaker_3dmm']=(face-mean)/std
                chunk=a.candidate_chunk
                while True:
                    try:pred=export_full(model,**{key:x.to(a.device) for key,x in streams.items()},source_length=s['length'],noise=noise,candidate_chunk=chunk);break
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        if chunk==1:raise
                        chunk=max(1,chunk//2);print('OOM candidate chunk reduced',chunk,flush=True)
                with path.open('xb') as f:np.save(f,pred.numpy())
                write(sidecar,dict(identity=item_identity,sha256=sha256_file(path),candidate_chunk=chunk,shape=list(pred.shape)))
            assert pred.shape==(10,s['length'],25);predictions.append(pred);speakers.append(torch.from_numpy(np.load(s['files']['speaker_emotion']['path'])).float());exports.append(dict(path=str(path),sha256=sha256_file(path),shape=list(pred.shape)))
            print('EXPORT',label,i+1,len(source_rows),s['length'],flush=True)
        export_seconds=time.perf_counter()-start
        values=compute_metrics(speakers,predictions,targets,threads=4,metric_names=identity['metrics'])
        result=dict(seed=point['seed'],arm=point['arm'],lambda_group=point['lambda_group'],metrics=values,export_seconds=export_seconds,peak_memory_bytes=torch.cuda.max_memory_allocated(a.device),total_source_frames=sum(s['length'] for s in source_rows),exports=exports,protocol='Development-80 full available source sequence, official temporal chunks750, K10 bank0, single processed paired GT; NOT official ten-GT or hidden test',TLCC_note='Existing implementation returns first candidate per clip; preserved literally.',video_metrics_executed=False)
        save_cached(result_path,case_identity,result);completed.append(dict(path=str(result_path),sha256=sha256_file(result_path)));del model,predictions;torch.cuda.empty_cache();print('TASK COMPLETE',label,values,flush=True)
    complete(out/'completed.json',completed)


if __name__=='__main__':main()
