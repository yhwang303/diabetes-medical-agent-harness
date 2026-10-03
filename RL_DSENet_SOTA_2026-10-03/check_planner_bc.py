"""Local NumPy/metadata fixtures only. No Torch import, training, simulator or remote."""
import argparse
import ast
import copy
import json
from pathlib import Path
import tempfile

import numpy as np
import planner_bc_common as common
from train_planner_bc import load_features, internal_diagnostics
from world_control_worker import score_predictions, QUANTILES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(); checks = []

    def check(name, condition):
        if not condition: raise AssertionError(name)
        checks.append(name)

    def rejects(name, function):
        try: function()
        except (ValueError, FileNotFoundError): checks.append(name)
        else: raise AssertionError(name)

    # Execute only the actual small plan constructor, without simulator imports.
    parsed = ast.parse((common.R / 'paired_collect.py').read_text())
    fn = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == 'plans')
    scope = {'np': np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'paired_collect.plans', 'exec'), scope)
    for anchor in (.25, 1.2, 11., 20.):
        anchors = np.full(9, anchor, dtype=np.float32)
        plans = np.stack([scope['plans'](float(a)) for a in anchors]).astype(np.float32)
        labels, actions = common.map_teacher(np.arange(9), plans, anchors)
        check('plan_mapping_' + str(anchor), labels.tolist() == list(common.PLAN_TO_ACTION)
              and np.array_equal(actions, np.clip(anchors * np.asarray(common.MULTIPLIERS, dtype=np.float32)[labels], 0, 20)))
        for j in range(1, 9):
            duration = 12 if j < 5 else 24
            check('duration_%s_%s' % (anchor, j), np.all(plans[0, j, duration:] == np.float32(anchor)))
    a = np.array([1.2], dtype=np.float32); plans = scope['plans'](float(a[0]))[None].astype(np.float32)
    q = np.full((1, 9, 72, 7), 120., dtype=np.float32); p = np.zeros((1, 9), dtype=np.float32)
    actions, chosen, _ = score_predictions(q, QUANTILES, p, p, plans, a, 'risk')
    labels, executable = common.map_teacher(chosen, plans, a)
    check('shared_teacher_tie_and_hold', chosen.tolist() == [0] and labels.tolist() == [4] and np.array_equal(actions, executable))
    wrong = plans.copy(); wrong[0, 0, 0] = 0
    rejects('wrong_first_action_rejected', lambda: common.map_teacher(np.array([0]), wrong, a))
    rejects('invalid_plan_index_rejected', lambda: common.map_teacher(np.array([9]), plans, a))

    protocol = common.read(common.R / 'protocol.json'); rows = []
    for patient in protocol['patients']:
        for seed in common.FIT_SEEDS + (common.CHECK_SEED,):
            for factor in protocol['bolus_factors']:
                job = dict(patient=patient, seed=seed, bolus_factor=factor)
                rows.append(dict(file='p%02d_s%d_b%.1f.npz' % (patient, seed, factor), job=job,
                                 origins=4, sha256='a' * 64, source_sha256='b' * 64))
    rows.sort(key=lambda x: x['file'])
    order = [dict(file=r['file'], count=r['origins'], sha256=r['sha256'], job=r['job']) for r in rows]
    manifest = dict(status='completed', split='train', horizon=72, stride=3, episodes=rows, origins=4 * len(rows))
    check('complete_train_metadata', common.validate_natural_manifest(manifest, protocol, order) == rows)
    for split in ('world_validation', 'development', 'confirmation'):
        bad = dict(manifest, split=split)
        rejects('reject_split_' + split, lambda: common.validate_natural_manifest(bad, protocol, order))
    for seed in (103011, 103101, 103901, 103201):
        bad = copy.deepcopy(manifest); bad['episodes'][0]['job']['seed'] = seed
        rejects('reject_nonBC_scene_' + str(seed), lambda: common.validate_natural_manifest(bad, protocol, order))
    bad = copy.deepcopy(manifest); bad['episodes'] = bad['episodes'][:-1]
    rejects('reject_missing_episode', lambda: common.validate_natural_manifest(bad, protocol, order))
    bad = copy.deepcopy(manifest); bad['episodes'][0]['file'] = '../escape.npz'
    rejects('reject_escaped_filename', lambda: common.validate_natural_manifest(bad, protocol, order))
    rejects('reject_world_file_order_change', lambda: common.validate_natural_manifest(manifest, protocol, list(reversed(order))))
    check('gate_boundaries_pass', common.agreement_gate(.9, .8, 100, 3)['passed'])
    for name, values in [('overall', (.899, .8, 100, 3)), ('nonhold', (.9, .799, 100, 3)),
                         ('count', (.9, .8, 99, 3)), ('episodes', (.9, .8, 100, 2)), ('hold_only', (1., None, 0, 0))]:
        check('gate_' + name + '_fails', not common.agreement_gate(*values)['passed'])
    rejects('gate_nan_rejected', lambda: common.agreement_gate(float('nan'), .9, 100, 3))

    with tempfile.TemporaryDirectory(prefix='bc_fixture_', dir=common.R / 'checks') as temporary:
        directory = Path(temporary); files = []
        for i in range(4):
            n = 50; name = 'fixture%d.npz' % i; plan = np.ones(n, dtype=np.int64)
            values = dict(features=np.zeros((n, 195), dtype=np.float32), teacher_plan_index=plan,
                teacher_action_index=np.zeros(n, dtype=np.int64), teacher_action_u_h=np.zeros(n, dtype=np.float32),
                teacher_plan_cost=np.zeros((n, 9), dtype=np.float64), anchor_u_h=np.ones(n, dtype=np.float32),
                origin_minute=np.arange(360, 360 + n * 15, 15, dtype=np.float32))
            values['teacher_plan_cost'][:, 1] = -1.
            with (directory / name).open('xb') as stream: np.savez_compressed(stream, **values)
            files.append(dict(file=name, count=n, sha256=common.sha(directory / name),
                              part='fit' if i == 0 else 'internal_check', full_source_episode=True))
        data = load_features(directory, dict(files=files)); ix = np.flatnonzero(~data['fit'])
        check('whole_episodes_stay_separate', len(ix) == 150 and data['fit'].sum() == 50
              and set(data['episode_index'][ix]) == {1, 2, 3})
        logits = np.zeros((len(ix), 9), dtype=np.float32); logits[:, 0] = 8
        diagnostics, _ = internal_diagnostics(logits, data, files, ix)
        check('synthetic_faithful_actor_passes', diagnostics['gate']['passed'] and diagnostics['overall_action_agreement'] == 1.)
        logits[:, 0] = 0; logits[:, 4] = 8
        diagnostics, _ = internal_diagnostics(logits, data, files, ix)
        check('synthetic_hold_collapse_fails', not diagnostics['gate']['passed'] and diagnostics['nonhold_action_agreement'] == 0.)
        bad_files = copy.deepcopy(files); bad_files[0]['sha256'] = '0' * 64
        rejects('cache_bytes_hash_rejected', lambda: load_features(directory, dict(files=bad_files)))
        bindings = common.snapshot([directory / files[0]['file']]); common.verify_files(bindings)
        with (directory / files[0]['file']).open('ab') as stream: stream.write(b'tamper')
        rejects('bound_artifact_change_rejected', lambda: common.verify_files(bindings))

    for filename in ('planner_bc_common.py', 'prepare_planner_bc.py', 'train_planner_bc.py', 'check_planner_bc.py'):
        ast.parse((common.R / filename).read_text(), feature_version=(3, 8)); checks.append(filename + '_python38_AST')
    report = dict(status='passed', checks=len(checks), names=checks,
        synthetic_mechanism_only=True, real_data_loaded=False, torch_imported=False,
        training_simulation_or_remote_run=False, CUDA_state_reload_and_gradients_not_tested=True)
    import sys
    if 'torch' in sys.modules: raise AssertionError('Mechanics accidentally imported Torch')
    if args.output:
        destination = args.output.resolve(); destination.relative_to(common.P)
        common.write_new(destination, report)
    print(json.dumps(report, allow_nan=False))


if __name__ == '__main__':
    main()
