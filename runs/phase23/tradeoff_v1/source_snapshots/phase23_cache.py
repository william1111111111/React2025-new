"""Fail-closed evaluation identities; no automatic legacy-cache migration."""
import json
from pathlib import Path
from .phase15_audit import canonical_hash,sha256_file
from .train_phase22 import write

VERSION='hirp-development-es-v23.1'


def implementation_identity():
    files=['phase23_cache.py','evaluate_phase23.py','phase23_aux.py','evaluate_phase22.py','phase22.py',
           'group_features.py','group_scores.py','paired_data.py','losses/energy_score.py','phase15_metrics.py']
    root=Path(__file__).parent
    return {name:sha256_file(root/name) for name in files}


def eval_identity(checkpoint,metadata,plan_path,plan,bank,k,scaler_path,batch_manifest,versions):
    if sha256_file(bank['path'])!=bank['sha256']:raise ValueError('noise content changed')
    if sha256_file(scaler_path)!=plan['scaler_sha256']:raise ValueError('scaler content changed')
    actual=[]
    for row in plan['data_files']+plan['normalization']:
        digest=sha256_file(row['path'])
        if digest!=row['sha256']:raise ValueError('evaluation source/reference/normalization content changed: '+row['path'])
        actual.append(dict(path=row['path'],sha256=digest))
    source=[x for x in actual if '/speaker/' in x['path']]
    reference=[x for x in actual if '/listener/' in x['path']]
    paired_names={x.replace('speaker/','listener/')+'.npy' for x in plan['clip_ids']}
    paired=[x for x in reference if any(x['path'].endswith('/facial-attributes/'+name) for name in paired_names)]
    if len(paired)!=len(plan['clip_ids']):raise ValueError('paired target manifest incomplete')
    return dict(schema=VERSION,checkpoint_sha256=sha256_file(checkpoint),global_step=metadata['global_step'],
        arm=metadata['arm'],init_seed=metadata['train_config']['init_seed'],lambda_group=metadata['train_config']['lambda_group'] if metadata['arm']!='C0' else 0.,
        prior_mode=metadata['prior_mode'],output_config=metadata['output_config'],
        evaluation_plan_sha256=sha256_file(plan_path),population_sha256=canonical_hash(plan['clip_ids']),
        source_fingerprints=source,paired_fingerprints=paired,reference_fingerprints=reference,
        all_data_fingerprints_sha256=canonical_hash(actual),noise_bank_sha256=bank['sha256'],K=k,
        permutations_sha256=canonical_hash(plan['permutations']),scales=metadata['scales'],
        descriptor_scaler_sha256=sha256_file(scaler_path),descriptor_distance_version='75d-frozen-euclidean-joint-valid-eps1e-8-v2',
        T=plan['T'],crops_and_lengths=batch_manifest,mask_policy='conditional pair_lengths; descriptor source_lengths; shuffle min(recipient pair, donor source)',
        aggregation='20 equally weighted sessions; 4 equally weighted fixed sources/session; uniform unique full session VAL reference pool',
        shared_noise_self_policy='exclude identical noise index for all source pairs',
        metric_implementation=versions)


def cached(path,identity):
    path=Path(path)
    if not path.exists():return None
    saved=json.loads(path.read_text())
    if saved.get('eval_identity')!=identity:raise ValueError('cache identity mismatch or legacy metadata missing: '+str(path))
    if saved.get('eval_identity_sha256')!=canonical_hash(identity):raise ValueError('identity digest mismatch')
    if saved.get('result_sha256')!=canonical_hash(saved['result']):raise ValueError('cached result content mismatch')
    return saved['result']


def save_cached(path,identity,result):
    write(path,dict(eval_identity=identity,eval_identity_sha256=canonical_hash(identity),result=result,result_sha256=canonical_hash(result)))


def complete(path,rows):
    record=dict(schema=VERSION,cases=rows,cases_sha256=canonical_hash(rows))
    if Path(path).exists():
        if json.loads(Path(path).read_text())!=record:raise ValueError('completion manifest mismatch; choose new evaluation directory')
        return 'reused'
    write(path,record);return 'created'
