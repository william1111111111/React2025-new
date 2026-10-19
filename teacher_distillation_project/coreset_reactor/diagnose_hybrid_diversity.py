"""Post-hoc channel/temporal attribution; never changes selection guards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from coreset_reactor.analyze_t1_student_distillation import geometry, GROUPS
from coreset_reactor.teacher_cache import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    provenance = json.loads((args.root/'provenance.json').read_text())
    paths = {'T1': Path(provenance['paths']['convergence_root'])/'eval/T1_s000600_val571_full.json',
             'T1H': args.root/'val571_full.json'}
    summaries, hashes, reference_ids = {}, {}, None
    for arm, path in paths.items():
        metric = json.loads(path.read_text())['metrics']
        predictions = torch.load(path.with_suffix('.predictions.pt'), map_location='cpu')
        if reference_ids is None:
            reference_ids = predictions['clip_ids']
        assert predictions['clip_ids'] == reference_ids and len(reference_ids) == 571
        summaries[arm] = {}
        for kind in ('official_predictions', 'soft_predictions'):
            values = [geometry(x) for x in predictions[kind]]
            means = {key: float(np.mean([x[key] for x in values])) for key in values[0]}
            expected = metric['FRDiv' if kind == 'official_predictions' else 'soft_FRDiv']
            assert abs(means['FRDiv']-expected) < 1e-5
            summaries[arm][kind] = means
        hashes[arm] = sha256_file(path.with_suffix('.predictions.pt'))
    output = {'summary': summaries, 'prediction_sha256': hashes,
              'source_sha256': sha256_file(Path(__file__)),
              'purpose': 'post-hoc descriptive attribution; no new training or altered guards'}
    (args.root/'diversity_attribution.json').write_text(json.dumps(output, indent=2)+'\n')
    heading = '## Post-hoc diversity attribution'
    report = args.report.read_text()
    if heading in report:
        raise ValueError('Attribution is already appended')
    lines = ['', heading, '',
             'Computed after the primary experiment. Channel contributions use the full 25-channel denominator and sum to FRDiv. Static means between-slot temporal-mean offsets; dynamic means the remaining time-varying slot residual. These are descriptive, not causal ablations.', '',
             '| Representation / component | T1 | T1H | Change |', '|---|---:|---:|---:|']
    for kind in ('official_predictions', 'soft_predictions'):
        for component in [f'{group}_total' for group in GROUPS] + ['FRDiv', 'dynamic_FRDiv']:
            a, b = summaries['T1'][kind][component], summaries['T1H'][kind][component]
            lines.append(f'| {kind} / {component} | {a:.6f} | {b:.6f} | {b-a:+.6f} |')
    lines += ['', f"Attribution source SHA256: `{output['source_sha256']}`."]
    args.report.write_text(report+'\n'.join(lines)+'\n')
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    main()
