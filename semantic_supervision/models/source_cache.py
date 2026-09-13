"""Fail-closed label access. Unreviewed source labels return NULL, never targets."""
import importlib.util,json
from pathlib import Path
VENDOR=Path(__file__).resolve().parents[1]/'vendor/react_semantic_supervision'
spec=importlib.util.spec_from_file_location('semantic_vendor_validator',VENDOR/'validate_annotations.py')
_validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(_validator)

def source_events(record,provenance,*,split,expected_media_hashes):
    _validator.validate(record)
    if record['record_type']!='speaker_events' or record['split']!=split:
        raise ValueError('source type/split mismatch')
    if provenance['clip_id']!=record['clip_id']:raise ValueError('provenance identity mismatch')
    visible=provenance.get('actual_visible_inputs',[])
    if not visible or any(v.get('role')!='speaker' for v in visible):raise ValueError('non-source provenance')
    if sorted(v['sha256'] for v in visible)!=sorted(expected_media_hashes):raise ValueError('source media identity mismatch')
    if provenance.get('review_status')!='accepted' or not provenance.get('training_eligible',False):return None
    if not provenance.get('speaker_channel_role_verified') or not provenance.get('sync_verified'):return None
    accepted=[e for e in record['events'] if e['evidence_status']=='supported']
    known=set(provenance.get('evidence_refs',[]))
    if any(e['evidence_ref'] not in known for e in accepted):raise ValueError('unregistered evidence')
    return accepted

def training_observations(record,provenance,*,split):
    _validator.validate(record)
    if split!='train' or record['split']!='train' or record['record_type']!='listener_observations':
        raise ValueError('privileged labels are TRAIN-only')
    if provenance.get('clip_id')!=record['clip_id']:raise ValueError('identity mismatch')
    if provenance.get('review_status')!='accepted':return None
    return record['events']

def load_source_cache(annotation_file,provenance_file,clip_id,*,split,expected_media_hashes):
    """Read only two explicit source sidecars; no target cache/root argument."""
    annotations=json.loads(Path(annotation_file).read_text())
    provenance=json.loads(Path(provenance_file).read_text())
    a=[r for r in annotations if r['clip_id']==clip_id]
    p=[r for r in provenance if r['clip_id']==clip_id]
    if len(a)!=1 or len(p)!=1:raise ValueError('missing or duplicate source record')
    return source_events(a[0],p[0],split=split,expected_media_hashes=expected_media_hashes)
