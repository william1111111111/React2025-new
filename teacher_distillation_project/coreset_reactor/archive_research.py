"""Build a local-only research evidence archive without loading model pickles.

Original reports and weights are never changed. Text evidence is copied with
SHA256 verification; large tensors stay in place and are explicitly indexed.
This is an archival utility, not an evaluation or training entry point.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

TEXT = {'.json', '.jsonl', '.md', '.csv', '.log', '.py', '.sh', '.yaml', '.yml', '.toml', '.txt'}
TENSORS = {'.pth', '.pt', '.npz', '.npy'}
METRICS = {'FRC', 'FRD', 'exact_FRD', 'FRDiv', 'FRVar', 'TLCC', 'FRSyn',
           'S_MSE', 'FRDvs', 'soft_FRDiv', 'coverage', 'validity', 'utilization',
           'gt_coverage', 'prediction_validity', 'candidate_utilization'}
SKIP_PARTS = {'.git', '.venv', '__pycache__', '.pytest_cache', 'archives',
              'archive_20260930', '.ssh', 'node_modules'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pointer(parts):
    return '/' + '/'.join(str(p).replace('~', '~0').replace('/', '~1') for p in parts)


def metric_records(value, parts=()):
    """Extract dict records, preserving pointers; never infer a shared protocol.

    Lists of per-context/per-rank rows remain in raw evidence, not this index.
    Delta records are labelled instead of being silently treated as scores.
    """
    if not isinstance(value, dict):
        return
    scalar = {k: v for k, v in value.items()
              if k in METRICS and isinstance(v, (int, float)) and not isinstance(v, bool)
              and math.isfinite(v)}
    if scalar:
        kind = 'delta' if any('delta' in str(p).lower() or '_minus_' in str(p) for p in parts) else 'unclassified_metric_record'
        yield {'json_pointer': pointer(parts), 'record_kind': kind,
               'metrics': scalar,
               'context': {k: value[k] for k in ('contexts', 'n', 'seed', 'steps', 'split', 'protocol', 'dataset') if k in value}}
    for k, v in value.items():
        if isinstance(v, dict):
            yield from metric_records(v, parts + (k,))


def path_strings(value):
    if isinstance(value, dict):
        for v in value.values():
            yield from path_strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from path_strings(v)
    elif isinstance(value, str) and value.startswith('/') and len(value) < 1500:
        yield value


def roots(repo, assets):
    pairs = [('repo_coreset', repo / 'coreset_reactor'),
             ('persistent_coreset', assets / 'coreset_reactor'),
             ('repo_research_docs', repo / 'reports'),
             ('repo_mam_source', repo / 'mam_reactor'),
             ('legacy_mam_docs', assets / 'mam_reactor/docs'),
             ('legacy_mam_results', assets / 'mam_reactor/results')]
    for name in ('react2024_t1_runs_600', 'react2024_t1_runs_600_v2',
                 'react2024_t1_runs_600_v3', 'react2024_t1_manifests',
                 'react2024_t1_manifests_v2'):
        pairs.append((name, Path('/tmp') / name))
    return pairs


def discover(repo, assets):
    found = {}
    for label, root in roots(repo, assets):
        if not root.exists():
            continue
        for base, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_PARTS and not d.startswith('.'))
            for name in sorted(files):
                p = Path(base) / name
                if p.suffix not in TEXT | TENSORS:
                    continue
                if any(token in name.lower() for token in ('private_key', 'id_rsa', 'id_ed25519', 'credential', '.env')):
                    continue
                # No dataset or pretrained assets from the MAM source tree.
                if label == 'repo_mam_source' and p.suffix not in TEXT:
                    continue
                found[str(p.absolute())] = (label, p.relative_to(root))
    for p in sorted(Path('/tmp').iterdir()):
        if (p.suffix in TEXT and
            re.match(r'(m3_oracle_|react2024_t[01]_|cross_dataset_full\.|coreset_m1_|regnn_frc_sweep\.)', p.name)
            and p.is_file()):
            found[str(p)] = ('historical_tmp', Path(p.name))
    return found


def checkpoint_kind(path):
    if path.suffix == '.pth':
        return 'optimizer_state' if 'optimizer' in path.name.lower() else 'checkpoint'
    if path.suffix == '.pt' and ('checkpoint' in path.name.lower() or
                                 path.name in {'student.pt', 'model.pt', 'final.pt'}):
        return 'checkpoint'
    return 'tensor_cache_or_prediction'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--notes', type=Path, required=True)
    parser.add_argument('--hash-checkpoints', action='store_true')
    parser.add_argument('--resume-empty', action='store_true',
                        help='Only reuse a failed discovery directory containing status.json alone')
    args = parser.parse_args()
    out = args.output.absolute()
    if args.resume_empty and out.exists():
        if {p.name for p in out.iterdir()} != {'status.json'}:
            raise ValueError('Refusing to reuse a nonempty archive')
    else:
        out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    def save(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    save('status.json', {'stage': 'collecting', 'started_unix': started})
    found = discover(args.repo, args.assets)
    inventory, metrics, configs, errors, references, report_index = [], [], [], [], set(), []
    hash_cache = {}
    for index, (name, (label, rel)) in enumerate(sorted(found.items())):
        p = Path(name)
        try:
            before = p.stat()
            record = {'source_path': name, 'resolved_path': str(p.resolve()),
                      'scope': label, 'relative_path': str(rel), 'bytes': before.st_size,
                      'mtime_ns': before.st_mtime_ns}
            if p.suffix in TEXT:
                dest = out / 'evidence' / label / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest)
                sha = digest(dest)
                if digest(p) != sha:
                    raise RuntimeError('Source changed or copy hash mismatch')
                record.update(storage='copied', sha256=sha, archive_path=str(dest.relative_to(out)))
                if p.suffix == '.md':
                    content = dest.read_text(errors='replace')
                    title = next((s.lstrip('# ').strip() for s in content.splitlines() if s.startswith('#')), p.stem)
                    report_index.append({'source_path': name, 'archive_path': record['archive_path'], 'title': title})
                if p.suffix == '.json':
                    try:
                        value = json.loads(dest.read_text())
                    except (ValueError, UnicodeError) as e:
                        errors.append({'path': name, 'error': f'JSON parse: {e}', 'raw_copy_retained': True})
                    else:
                        references.update(path_strings(value))
                        for row in metric_records(value):
                            row.update(source_path=name, archive_path=record['archive_path'], sha256=sha)
                            metrics.append(row)
                        if any(s in p.name.lower() for s in ('config', 'provenance', 'experiment', 'launch', 'protocol', 'manifest', 'summary')):
                            configs.append({'source_path': name, 'archive_path': record['archive_path'],
                                            'sha256': sha, 'content': value})
            else:
                kind = checkpoint_kind(p)
                record.update(storage='in_place_not_copied', kind=kind, sha256=None)
                if args.hash_checkpoints and kind in {'checkpoint', 'optimizer_state'}:
                    key = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                    if key not in hash_cache:
                        hash_cache[key] = digest(p)
                    record['sha256'] = hash_cache[key]
            after = p.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError('Source mutated during archiving')
            inventory.append(record)
        except (OSError, RuntimeError) as e:
            errors.append({'path': name, 'error': str(e)})
        if index % 200 == 0:
            print(json.dumps({'stage': 'collecting', 'files': index, 'total': len(found),
                              'elapsed_seconds': round(time.time()-started, 1)}), flush=True)
    known = {r['source_path'] for r in inventory}
    missing_refs = []
    for name in sorted(references):
        p = Path(name)
        if p.suffix not in {'.pth', '.pt'} or name in known:
            continue
        if not name.startswith(('/tmp/react', '/tmp/coreset', '/public/zhaowenjie/react2025_new/', '/home/zhaowenjie/react2025_new/')):
            missing_refs.append({'path': name, 'status': 'outside_scanned_local_project'})
            continue
        if not p.is_file():
            missing_refs.append({'path': name, 'status': 'missing_at_snapshot'})
            continue
        st = p.stat()
        kind = checkpoint_kind(p)
        sha = digest(p) if args.hash_checkpoints and kind in {'checkpoint', 'optimizer_state'} else None
        inventory.append({'source_path': name, 'resolved_path': str(p.resolve()), 'scope': 'referenced_weight',
                          'relative_path': p.name, 'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns,
                          'storage': 'in_place_not_copied', 'kind': kind, 'sha256': sha})
    for filename, values in [('file_manifest.jsonl', inventory), ('metric_records.jsonl', metrics),
                             ('configs_and_provenance.jsonl', configs)]:
        with (out / filename).open('w') as f:
            for value in values:
                f.write(json.dumps(value, ensure_ascii=False) + '\n')
    with (out / 'metric_index.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['source_path', 'archive_path', 'json_pointer', 'record_kind', *sorted(METRICS)])
        writer.writeheader()
        for r in metrics:
            writer.writerow({k: r[k] for k in ('source_path', 'archive_path', 'json_pointer', 'record_kind')} | r['metrics'])
    with (out / 'weights_and_caches.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['source_path', 'resolved_path', 'kind', 'bytes', 'sha256', 'storage'])
        writer.writeheader()
        for r in inventory:
            if r['storage'] != 'copied':
                writer.writerow({k: r.get(k) for k in writer.fieldnames})
    save('report_index.json', report_index)
    save('missing_or_external_weight_references.json', missing_refs)
    save('collection_errors.json', errors)
    index_lines = ['# 全部原始文档索引', '',
                   '按来源目录保留原文。历史文档的“当前”“运行中”“未启动”均指其写作时点；最新结论以本归档正文和最新原始汇总为准。', '']
    for r in report_index:
        index_lines.append(f"- [{r['title'].replace('[', '').replace(']', '')}](<{r['archive_path']}>) — `{r['source_path']}`")
    (out / 'ALL_REPORTS.md').write_text('\n'.join(index_lines) + '\n')
    shutil.copytree(args.notes, out / 'notes')
    for name in ('README.md', 'METHOD_AND_PARAMETERS.md', 'RESULTS_AND_HISTORY.md', 'REPRODUCIBILITY.md'):
        shutil.copy2(args.notes / name, out / name)
    for name, command in [('git_head.txt', ['git', 'rev-parse', 'HEAD']),
                          ('git_status.txt', ['git', 'status', '--short']),
                          ('working_tree.patch', ['git', 'diff', '--no-ext-diff', '--binary']),
                          ('git_log.txt', ['git', 'log', '-80', '--format=%H %ad %s', '--date=iso-strict'])]:
        result = subprocess.run(command, cwd=args.repo, capture_output=True, text=True)
        (out / name).write_text(result.stdout)
        if result.returncode:
            errors.append({'path': name, 'error': result.stderr})
    python = args.assets / '.venv/bin/python'
    result = subprocess.run([str(python), '-m', 'pip', 'freeze'], capture_output=True, text=True)
    (out / 'environment_pip_freeze.txt').write_text(result.stdout or result.stderr)
    save('environment.json', {'python': str(python), 'pip_freeze_returncode': result.returncode,
                             'os_uname': list(os.uname()),
                             'note': 'Current environment snapshot, not proof every historical run used identical versions.'})
    save('scope.json', {'date': '2026-09-30', 'roots': [(a,str(b)) for a,b in roots(args.repo,args.assets)],
                       'standalone_tmp_allowlist': 'm3_oracle_*, react2024_t0_*, react2024_t1_*, cross_dataset_full.*, coreset_m1_*, regnn_frc_sweep.*',
                       'not_included': ['raw datasets', 'SSH keys/secrets', 'binary cache/trajectory payloads',
                                        'checkpoint payload copies', 'unrelated project branches'],
                       'privacy': 'LOCAL ONLY. Includes local paths, sample identities, unpublished code. Not safe to push wholesale.',
                       'completeness': 'All readable supported file types in listed roots at snapshot; unavailable artifacts explicitly listed. Not a claim that deleted historical files were recovered.'})
    save('collection_errors.json', errors)
    save('status.json', {'stage': 'complete' if not errors else 'complete_with_recorded_gaps',
                         'started_unix': started, 'finished_unix': time.time(),
                         'wall_seconds': time.time()-started, 'files_indexed': len(inventory),
                         'text_files_copied': sum(r['storage']=='copied' for r in inventory),
                         'original_reports': len(report_index), 'metric_records': len(metrics),
                         'config_provenance_records': len(configs),
                         'hashed_weight_paths': sum(r['storage']!='copied' and bool(r['sha256']) for r in inventory),
                         'unavailable_or_external_weight_references': len(missing_refs), 'errors': len(errors)})
    print((out / 'status.json').read_text(), flush=True)


if __name__ == '__main__':
    main()
