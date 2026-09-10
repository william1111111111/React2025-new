"""Phase24 immutable protocols and validated dependency/normalization contracts."""
import json,sys,inspect,random
from pathlib import Path
import torch
from .phase15_audit import sha256_file,canonical_hash
from .phase23_cache import cached,save_cached,complete
from .phase22 import load_checkpoint
from .train_phase22 import write
ROOT=Path('runs/phase24/evaluation_v1')
OLD=Path('runs/phase23/tradeoff_v1')
PLAN=Path('runs/phase22/replicated_v1/evaluation_plan/manifest.json')
LEGACY=Path('/home/zhengshiyi/react2025')
MAM=LEGACY/'mam_reactor'


def dependencies(root=Path('hirp')):
    # All package Python including model/data; offline report/setup/test utilities
    # cannot be imported by the runtime. Hash source, never outputs/identity JSON.
    excluded=('setup','report','write_','verify','visualize','audit')
    return {p.relative_to(root).as_posix():sha256_file(p) for p in sorted(root.rglob('*.py'))
            if 'tests' not in p.parts and not p.name.startswith(excluded)}


def verify_files(rows):
    for r in rows:
        if sha256_file(r['path'])!=r['sha256']:raise ValueError('frozen content mismatch: '+r['path'])


def normalization(plan):
    rows=plan['normalization'];verify_files(rows)
    names={Path(r['path']).name:r for r in rows}
    if not {'mean_face.npy','std_face.npy'}<=names.keys():raise ValueError('missing normalization')
    # Ensure the actual paths used by generation have exactly the frozen contents.
    for name in ('mean_face.npy','std_face.npy'):
        p=Path('external/FaceVerse')/name
        if sha256_file(p)!=names[name]['sha256']:raise ValueError('actual normalization mismatch: '+str(p))
    return rows


def guarded(plan,operation):
    normalization(plan)
    return operation()


def validate_candidate(point,meta):
    actual=(meta['arm'],meta['train_config']['init_seed'],meta['global_step'])
    if actual!=(point['arm'],point['seed'],point['steps']):raise ValueError('candidate arm/seed/step mismatch')
    effective=0. if meta['arm']=='C0' else meta['train_config']['lambda_group']
    if effective!=point['lambda_group']:raise ValueError('candidate lambda mismatch')
    if meta['train_config']['lambda_group']!=point['stored_lambda_group']:raise ValueError('stored lambda mismatch')
    for k in ('prior_mode','output_config','config','scales'):
        if canonical_hash(meta[k])!=canonical_hash(point[k]):raise ValueError('candidate semantic mismatch: '+k)


def load_candidate(point,device='cpu'):
    if sha256_file(point['checkpoint'])!=point['checkpoint_sha256']:raise ValueError('checkpoint hash mismatch')
    model,meta=load_checkpoint(point['checkpoint'],device);validate_candidate(point,meta);return model,meta


def runtime():
    p=json.loads(PLAN.read_text());normalization(p)
    frozen=json.loads((ROOT/'dependency_snapshot.json').read_text())
    if dependencies()!=frozen['hirp']:raise ValueError('runtime Python dependency changed; new namespace required')
    verify_files(frozen['legacy'])
    return p,frozen


def official_selection(paired,pool,rng):
    # Literal rule in locked ReactionDataset.__getitem__ test branch.
    c=pool if len(pool)<=1 else [p for p in pool if p!=paired]
    return [paired]+(rng.sample(c,9) if len(c)>=9 else rng.choices(c,k=9))


def legacy_snapshot():
    # Conservative source closure, plus actual runtime module paths below.
    paths=[]
    for sub in ('framework','dataset','regnn'):
        paths.extend((LEGACY/sub).rglob('*.py'))
    paths.extend((MAM/'regnn').rglob('*.py'))
    paths += [LEGACY/'configs/model/emotion_autoencoder.yaml',LEGACY/'pretrained_models/post_processor/checkpoint.pth',MAM/'checkpoints/mam_reactor_offline_evidence_epoch0003.pth']
    return [dict(path=str(p),sha256=sha256_file(p)) for p in sorted(set(paths)) if '__pycache__' not in p.parts]
