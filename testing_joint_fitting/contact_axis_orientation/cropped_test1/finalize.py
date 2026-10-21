"""Verify persisted outputs and add coverage details to the evaluation report."""
import hashlib
import json
from pathlib import Path

from .evaluate import DEFAULT


def main():
    archive = DEFAULT.resolve()
    root = archive / 'candidate_test1'
    protocol = json.loads((root / 'protocol.json').read_text())
    rows = json.loads((root / 'rows.json').read_text())
    summary = json.loads((root / 'summary.json').read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for name, expected in protocol['algorithm_sources'].items():
        assert sha(Path(name)) == expected, name
    assert sha(Path(__file__).with_name('evaluate.py')) == protocol['runner_sha256']
    assert sha(archive / 'run.json') == protocol['run_manifest_sha256']
    for name, expected in protocol['record_hashes'].items():
        assert sha(archive / 'test1/records' / name) == expected, name
    for row in rows:
        saved = json.loads((root / 'pairs' / f"pair_{row['pair_index']:05d}.json").read_text())
        assert saved == row
        assert sha(archive / 'test1' / row['prediction_file']) == row['prediction_sha256']
    failures = [r for r in rows if r['failed']]
    assert all(r['valid_fit_observations'] < 8 for r in failures)
    audit_path = root / 'audit.json'
    audit = json.loads(audit_path.read_text())
    audit.update(persisted_pair_records_verified=len(rows), prediction_hashes_reverified=len(rows),
        inference_record_hashes_reverified=len(protocol['record_hashes']),
        runner_and_algorithm_hashes_reverified=True,
        insufficient_fitting_observations=len(failures), optimizer_failures=0)
    audit_path.write_text(json.dumps(audit, indent=2)+'\n')
    text = (root / 'REPORT.md').read_text()
    section = ['## Fitting coverage and failure-inclusive metrics', '',
        'Of 832 trajectory-scoreable revolute pairs, 826 had sufficient observations in states 2–4 '
        'and returned a fit. Six pairs had only 0–7 valid observations there and could not be fitted; '
        'these are data-coverage failures, with no optimizer exceptions. Means/medians in the metric '
        'tables use the 826 successful fits, except input trajectory ADE, which uses all 832 pairs. '
        'Assigning 90° to each unavailable fit gives a failure-inclusive mean axis error of '
        f"{summary['overall']['axis_with_failure_penalty']['mean']:.6f}°. "
        'The within-15° rate uses all 832 pairs, including the six unavailable fits as failures.', '',
        '| Joint | Considered | Fitted | Insufficient observations | Mean axis error with 90° failure penalty |',
        '|---|---:|---:|---:|---:|']
    for ordinal, stats in summary['by_joint'].items():
        section.append(f"| {ordinal} | {stats['n']} | {stats['n']-stats['failures']} | "
            f"{stats['failures']} | {stats['axis_with_failure_penalty']['mean']:.4f}° |")
    section += ['', '| Pair | Scene | Joint | Valid fitting observations |', '|---|---:|---:|---:|']
    for row in failures:
        section.append(f"| {row['pair_index']} | {row['scene_index']} | {row['joint_ordinal']} | {row['valid_fit_observations']} |")
    section += ['', '## Per-object results', '',
        '| Object ID | Considered | Fitted | Mean axis ° | Median axis ° | Mean pivot distance | Mean rotation ° |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for obj, stats in summary['by_object'].items():
        section.append(f"| {obj} | {stats['n']} | {stats['n']-stats['failures']} | "
            f"{stats['axis']['mean']:.4f} | {stats['axis']['median']:.4f} | "
            f"{stats['pivot']['mean']:.6f} | {stats['rotation']['mean']:.4f} |")
    section += ['', 'Successful fits include 29 cases above 15° axis error. The means are therefore '
        'sensitive to a relatively small tail; 90% of successful fits have axis error ≤1.0853°. '
        'Low axis error does not imply low rotation/trajectory error: the rotation metric also '
        'depends on estimated rotation angles.', '',
        'All saved per-pair JSON outputs, prediction hashes, inference record hashes, '
        'runner hash and fitting dependency hashes were rechecked after completion.', '']
    marker = '## Fitting coverage and failure-inclusive metrics'
    if marker in text:
        text = text[:text.index(marker)]
    (root / 'REPORT.md').write_text(text+'\n'+'\n'.join(section))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
