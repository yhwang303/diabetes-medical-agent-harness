"""Read-only deployment diagnostics on a finished evaluation panel.

This does not score efficacy or select a model. CGM bins use the observation
available immediately before the decision, never the interval's future BG.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

R = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def aggregate(rows):
    ratios = [row['action_u_h'] / row['observed_anchor_u_h'] for row in rows]
    grid = Counter()
    for ratio in ratios:
        index = round(ratio * 4)
        grid[str(index / 4) if 0 <= index <= 8 and abs(ratio - index / 4) <= 1e-6 else 'off_quarter_grid'] += 1
    return dict(decisions=len(rows), ratio_mean=statistics.mean(ratios) if rows else None,
        ratio_min=min(ratios) if rows else None, ratio_max=max(ratios) if rows else None,
        at_anchor_count=sum(abs(row['action_u_h'] - row['observed_anchor_u_h']) <= 1e-6 for row in rows),
        below_anchor_count=sum(row['action_u_h'] < row['observed_anchor_u_h'] - 1e-6 for row in rows),
        above_anchor_count=sum(row['action_u_h'] > row['observed_anchor_u_h'] + 1e-6 for row in rows),
        requested_ratio_counts=dict(sorted(grid.items())))


def diagnose(name):
    if Path(name).name != name or name in ('.', '..'):
        raise ValueError('Expected a result directory name')
    folder = R / 'results' / name
    summary_path = folder / 'summary.json'
    summary = json.loads(summary_path.read_text())
    manifest = json.loads((folder / 'manifest.json').read_text())
    if summary['status'] != 'completed' or summary['count'] != summary['planned_count']:
        raise ValueError('Only finished panels can be diagnosed; retain incomplete evaluations separately')
    decisions_path = folder / 'decisions.jsonl'
    decisions = [json.loads(line) for line in decisions_path.read_text().splitlines()]
    by_case = defaultdict(list)
    for row in decisions:
        if not all(math.isfinite(row[k]) for k in ('action_u_h', 'observed_anchor_u_h')):
            raise ValueError('Nonfinite decision')
        if row['observed_anchor_u_h'] <= 0:
            raise ValueError('Nonpositive observed anchor')
        by_case[row['key']].append(row)
    if set(by_case) != {episode['key'] for episode in summary['episodes']}:
        raise ValueError('Decision and episode key sets differ')
    linked = 0
    max_request_error = 0.
    max_pump_error = 0.
    bins = defaultdict(list)
    per_case = {}
    for episode in summary['episodes']:
        raw_path = folder / episode['raw_path']
        if sha(raw_path) != episode['raw_sha256']:
            raise ValueError('Raw trajectory hash differs from the summary')
        raw = json.loads(raw_path.read_text())
        if episode['failure_reason'] is not None or not episode['raw_trajectory_available']:
            raise ValueError('Action linkage requires a complete genuine trajectory for every case')
        records = {record['minute']: record for record in raw['records']}
        rows = by_case[episode['key']]
        if [row['decision'] for row in rows] != list(range(len(rows))):
            raise ValueError('Decisions are not contiguous from zero')
        if len(rows) != episode['metrics']['planned_intervals']:
            raise ValueError('Decision count does not match the complete follow-up')
        for row in rows:
            end = manifest['protocol']['warmup_minutes'] + 5 * (row['decision'] + 1)
            previous = records[end - 5]
            delivered = records[end]
            error = abs(row['action_u_h'] - delivered['requested_basal_u_h'])
            if error > 1e-6:
                raise ValueError('Audited action does not match the simulator request')
            max_request_error = max(max_request_error, error)
            max_pump_error = max(max_pump_error, abs(row['action_u_h'] - delivered['delivered_basal_u_h']))
            g = previous['cgm_mg_dl']
            if not math.isfinite(g):
                raise ValueError('Invalid pre-decision CGM')
            label = '<70' if g < 70 else '70..100' if g < 100 else '100..140' if g < 140 else '140..180' if g <= 180 else '>180'
            bins[label].append(row)
            linked += 1
        per_case[episode['key']] = aggregate(rows)
    return dict(schema=1, panel=name, status='all_actions_linked_to_actual_requests',
        split=summary['split'], smoke=summary['smoke'], efficacy_or_causal_claim=False,
        summary_sha256=sha(summary_path), manifest_sha256=sha(folder / 'manifest.json'),
        decisions_sha256=sha(decisions_path), source_sha256=sha(Path(__file__).resolve()),
        linked_decisions=linked, max_request_absolute_error_u_h=max_request_error,
        max_requested_delivered_absolute_error_u_h=max_pump_error,
        at_anchor_tolerance_u_h=1e-6, quarter_grid_ratio_tolerance=1e-6,
        cgm_bin_definition='actual CGM at previous interval end, available before the decision',
        overall=aggregate(decisions), by_predecision_cgm={key: aggregate(value) for key, value in sorted(bins.items())},
        by_case=per_case)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = (R / args.output).resolve()
    output.relative_to((R / 'checks').resolve())
    result = diagnose(args.panel)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status=result['status'], output=str(output), overall=result['overall'])))


if __name__ == '__main__':
    main()
