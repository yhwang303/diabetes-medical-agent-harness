"""Standard-library serial scheduler for fixed PPO development checkpoints only.

No training, checkpoint selection, confirmation, new seed, or automatic retry.
Waits at most eight hours in total; polls completed history prefixes every 30 s.
An existing result/audit directory is an error, never overwritten or skipped.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

R = Path(__file__).resolve().parent
P = R.parent
B = P / 'RL_DSENet_2026-09-17'
E = P / 'RL_DSENet_公平低糖_2026-09-21'
POLL_SECONDS = 30
MAX_SECONDS = 8 * 60 * 60
ALLOWED_ITERATIONS = (8, 16, 32, 40)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x') as stream: json.dump(value, stream, indent=2, allow_nan=False)


def simple_name(name):
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('Expected a simple run/prefix name')


def validate_request(args):
    if args.kind not in ('ppo', 'world_ppo'): raise ValueError('Only the two original PPO evaluation kinds are accepted')
    simple_name(args.run); simple_name(args.name_prefix)
    if (not args.iterations or args.iterations != sorted(set(args.iterations))
            or any(i not in ALLOWED_ITERATIONS for i in args.iterations)):
        raise ValueError('Iterations must be a strictly increasing subset of 8,16,32,40')
    if type(args.batch_size) is not int or args.batch_size < 1: raise ValueError('Batch size must be positive')


def startup_binding(kind, run):
    """Read metadata and hash sources without importing the evaluator/runtime."""
    config = run / 'config.json'; provenance_path = run / 'provenance.json'
    c = read(config); provenance = read(provenance_path); protocol = read(R / 'protocol.json')
    if (c['name'] != run.name or c['seed'] != 260915 or protocol['training_seed'] != 260915
            or c['iterations'] != 40 or c['development_checkpoints'] != list(ALLOWED_ITERATIONS)
            or provenance['config'] != c or provenance['training_seed'] != 260915):
        raise ValueError('Expected the already-started, bound formal 40-iteration PPO run')
    if set(protocol['development_scenario_seeds']) & set(protocol['confirmation_scenario_seeds']):
        raise ValueError('Development/confirmation seed overlap')
    template = R / 'configs' / ('ppo_wide_risk.json' if kind == 'ppo' else 'ppo_world_risk.json')
    requested = read(template)
    if kind == 'ppo':
        if {k: v for k, v in c.items() if k != 'name'} != {k: v for k, v in requested.items() if k != 'name'}:
            raise ValueError('Wide PPO config is not the frozen formal contract')
    else:
        if (c.get('run_mode') != 'formal' or c.get('allow_world_smoke_budget') is not False
                or Path(c['output_dir']).resolve() != run or c.get('world_variant') not in ('point', 'quantile')
                or any(c.get(k) != v for k, v in requested.items() if k not in ('name', 'world_checkpoint', 'world_variant'))):
            raise ValueError('World PPO config is not the frozen formal contract')
    scorer = P / protocol['scorer']
    if sha(scorer) != protocol['scorer_sha256']: raise ValueError('Original scorer SHA mismatch')
    sources = {Path(__file__).resolve(), R / 'evaluate_candidates.py', R / 'controller_baselines.py',
        R / 'physiologic_features.py', E / 'ppo_env.py', E / 'evaluate.py', E / 'brake.py',
        P / 'RL进阶对比_2026-09-15/observable_history.py', scorer,
        R / 'ppo_wide_worker.py', B / 'world_model.py', B / 'forecast_model.py',
        P / 'RL_DITR创新_2026-09-16/response_operator.py', P / 'RL_DITR创新_2026-09-16/ditr_model.py'}
    sources.update(B.glob('dsenet/**/*.py'))
    artifacts = {config, provenance_path, template, R / 'protocol.json',
                 P / 'Loop数据集/训练管线_v2/prepared/normalization.json', R / 'checks/frozen_selected_version.json'}
    if kind == 'ppo':
        training_sources = [R / 'train_wide.py', R / 'ppo_wide_worker.py', R / 'physiologic_features.py', E / 'ppo_env.py']
        if set(provenance['source_sha256']) != {p.name for p in training_sources}:
            raise ValueError('Wide training source declaration mismatch')
        for path in training_sources:
            if sha(path) != provenance['source_sha256'][path.name]: raise ValueError('Wide training source SHA mismatch')
        sources.update(training_sources + [B / 'policy_bounded.py'])
        artifacts.update([R / 'configs/ppo_wide_smoke.json', B / 'results/D05_selected_world/config.json',
                          B / 'results/D06_selected_policy/config.json'])
        selected = read(R / 'checks/frozen_selected_version.json')
        for key, item in selected['weights'].items():
            if key != 'policy':
                path = (B / item['path']).resolve()
                if sha(path) != item['sha256']: raise ValueError('Frozen wide dependency SHA mismatch')
                artifacts.add(path)
        wc = read(B / 'results/D05_selected_world/config.json')
        context = (P / wc['context_checkpoint']).resolve()
        if sha(context) != wc['context_sha256']: raise ValueError('Frozen context SHA mismatch')
        artifacts.add(context)
    else:
        for relative, expected in provenance['source_sha256'].items():
            path = (P / relative).resolve(); path.relative_to(P)
            if sha(path) != expected: raise ValueError('World PPO training source SHA mismatch: ' + relative)
            sources.add(path)
        sources.update([R / 'ppo_world_worker.py', R / 'train_world_policy.py', R / 'world_control_worker.py'])
        wp = Path(c['world_checkpoint']).resolve(); wp.relative_to((R / 'results').resolve())
        world_run = wp.parent.parent if wp.parent.name == 'checkpoints' else wp.parent
        bound = read(world_run / 'provenance.json'); completed = read(world_run / 'completion.json')
        if (completed['status'] != 'training_completed_candidate_only' or completed['steps'] != 4000
                or completed['configured_steps'] != 4000 or completed['budget_override'] is not False):
            raise ValueError('Formal world PPO requires the completed 4000-step world')
        for key, path in (('world_checkpoint_sha256', wp), ('world_training_provenance_sha256', world_run / 'provenance.json'),
                          ('world_completion_sha256', world_run / 'completion.json')):
            if c[key] != sha(path): raise ValueError('World dependency binding mismatch: ' + key)
            artifacts.add(path)
        for key in ('forecast', 'normalization'):
            path = Path(bound[key]['path']).resolve()
            if sha(path) != bound[key]['sha256']: raise ValueError('World frozen dependency SHA mismatch')
            artifacts.add(path)
        for relative, expected in bound['source_sha256'].items():
            path = (P / relative).resolve(); path.relative_to(P)
            if sha(path) != expected: raise ValueError('World training source SHA mismatch')
            sources.add(path)
    hashes = {str(path.resolve()): sha(path) for path in sorted(sources | artifacts)}
    return dict(config=c, protocol_sha256=sha(R / 'protocol.json'), evaluator_sha256=sha(R / 'evaluate_candidates.py'),
                source_sha256={str(path.resolve()): hashes[str(path.resolve())] for path in sorted(sources)},
                bound_file_sha256=hashes)


def verify_frozen(binding):
    changed = [path for path, digest in binding['bound_file_sha256'].items()
               if not Path(path).is_file() or sha(path) != digest]
    if changed: raise ValueError('Startup-bound source/config/dependency changed: ' + repr(changed))


def training_health(run):
    failure = run / 'failure.json'
    if failure.exists(): raise RuntimeError('Training explicitly reported failure: ' + str(failure))
    completion = run / 'completion.json'
    state = dict(completion_present=completion.exists(), failure_present=False, activity_unknown=True)
    if completion.exists():
        try: value = read(completion)
        except json.JSONDecodeError:
            state['completion_write_in_progress_or_invalid'] = True; return state
        if value.get('status') not in ('completed', 'completed_candidate_only') or value.get('iterations') != 40:
            raise RuntimeError('Training completion metadata does not report the fixed 40-iteration budget')
        state.update(training_completed=True, completion_sha256=sha(completion), activity_unknown=False)
    # Silence, stale progress, and a missing completion file are NOT evidence of failure.
    return state


def completed_checkpoint(run, iteration):
    history = run / 'history.jsonl'; checkpoint = run / ('policy_iter%02d.pt' % iteration)
    if not history.exists(): return None, 'waiting_for_history'
    prefix = []
    with history.open('rb') as stream:
        for expected in range(1, iteration + 1):
            line = stream.readline()
            if not line.endswith(b'\n'): return None, 'waiting_for_complete_history_prefix'
            row = json.loads(line)
            if type(row['iteration']) is not int or row['iteration'] != expected:
                raise ValueError('Training-history prefix is not contiguous')
            prefix.append(line)
    if not checkpoint.exists(): return None, 'waiting_for_checkpoint_file'
    if (not Path(row['checkpoint']).is_absolute() or Path(row['checkpoint']).resolve() != checkpoint
            or sha(checkpoint) != row['checkpoint_sha256']):
        raise ValueError('Completed checkpoint/history SHA binding mismatch')
    raw = b''.join(prefix)
    return dict(iteration=iteration, checkpoint=str(checkpoint), checkpoint_sha256=row['checkpoint_sha256'],
                history_prefix_sha256=hashlib.sha256(raw).hexdigest(), history_prefix_bytes=len(raw)), 'ready'


def evaluation_command(args, run, iteration):
    name = '%s%02d_r1' % (args.name_prefix, iteration)
    return name, [str(P / '.venv/bin/python'), '-u', str(R / 'evaluate_candidates.py'),
                  '--kind', args.kind, '--name', name, '--split', 'development',
                  '--config', str(run / 'config.json'), '--checkpoint', str(run / ('policy_iter%02d.pt' % iteration)),
                  '--batch-size', str(args.batch_size)]


def inspect_evaluation(out, kind, checkpoint_binding, startup):
    manifest_path = out / 'manifest.json'; summary_path = out / 'summary.json'
    result = dict(manifest_sha256=sha(manifest_path) if manifest_path.exists() else None,
                  summary_sha256=sha(summary_path) if summary_path.exists() else None)
    if not manifest_path.exists() or not summary_path.exists():
        raise RuntimeError('Evaluator did not leave a complete manifest/summary pair')
    manifest = read(manifest_path); summary = read(summary_path)
    if (manifest['kind'] != kind or manifest['split'] != 'development' or manifest['smoke'] is not False
            or manifest.get('confirmation') is not False or manifest['checkpoint_sha256'] != checkpoint_binding['checkpoint_sha256']
            or summary['kind'] != kind or summary['split'] != 'development' or summary['smoke'] is not False
            or summary['status'] != 'completed' or summary.get('error') is not None
            or summary['count'] != 60 or summary['planned_count'] != 60
            or any(item.get('technical_failure_reason') is not None for item in summary['episodes'])):
        raise RuntimeError('Formal development evaluation failed or its protocol/binding changed')
    actual = manifest['policy_evaluation_binding']
    for key in ('iteration', 'checkpoint_sha256', 'history_prefix_sha256', 'history_prefix_bytes'):
        if actual[key] != checkpoint_binding[key]: raise ValueError('Evaluator checkpoint/history binding differs from launch')
    run = Path(checkpoint_binding['checkpoint']).parent
    for key, filename in (('config_sha256', 'config.json'), ('training_provenance_sha256', 'provenance.json')):
        if actual[key] != startup['bound_file_sha256'][str(run / filename)]:
            raise ValueError('Evaluator training metadata differs from startup: ' + key)
    if manifest['config_sha256'] != actual['config_sha256']:
        raise ValueError('Evaluator manifest/config binding mismatch')
    if manifest['artifact_sha256'].get(str((R / 'protocol.json').relative_to(P))) != startup['protocol_sha256']:
        raise ValueError('Evaluator protocol binding differs from startup')
    for relative, digest in manifest['source_sha256'].items():
        path = str((P / relative).resolve())
        if startup['source_sha256'].get(path) != digest: raise ValueError('Evaluator used source outside its startup snapshot: ' + relative)
    result.update(status=summary['status'], count=summary['count'], failed_episode_count=summary['failed_episode_count'],
                  native_or_environment_episode_failures_do_not_stop_queue=True)
    return result


def stop_owned_evaluation(process):
    if process.poll() is not None: return
    try: os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError: return
    try: process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try: os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError: return
        process.wait(timeout=15)


def run_schedule(args):
    validate_request(args)
    run = (R / 'results' / args.run).resolve(); run.relative_to((R / 'results').resolve())
    audit = R / 'checks' / ('queue_%s_%s_%s_%s' % (args.kind, args.run, args.name_prefix,
                     '-'.join('%02d' % value for value in args.iterations)))
    audit.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); deadline = started + MAX_SECONDS; launched = []; process = None
    final = dict(status='started', kind=args.kind, run=str(run), iterations=args.iterations,
                 started_at=utc(), maximum_seconds=MAX_SECONDS, checkpoint_selection=False, confirmation=False)
    with (audit / 'events.jsonl').open('x', buffering=1) as events:
        def event(event_type, **values):
            item = dict(event=event_type, at=utc(), elapsed_seconds=time.monotonic() - started, **values)
            events.write(json.dumps(item, allow_nan=False) + '\n')
            if event_type != 'waiting': print(json.dumps(item, allow_nan=False), flush=True)
        try:
            startup = startup_binding(args.kind, run)
            planned = [evaluation_command(args, run, iteration) for iteration in args.iterations]
            if any((R / 'results' / name).exists() for name, _ in planned):
                raise FileExistsError('At least one planned evaluation directory already exists; no overwrite or skip')
            write_new(audit / 'manifest.json', dict(**final, startup=startup,
                scheduler_source_sha256=sha(Path(__file__).resolve()), poll_seconds=POLL_SECONDS,
                planned_evaluations=[dict(name=name, command=command) for name, command in planned]))
            event('queue_started', audit=str(audit), iterations=args.iterations)
            for iteration, (name, command) in zip(args.iterations, planned):
                while True:
                    if time.monotonic() >= deadline: raise TimeoutError('Eight-hour queue deadline reached')
                    health = training_health(run)
                    binding, waiting = completed_checkpoint(run, iteration)
                    if binding is not None: break
                    if health.get('training_completed'):
                        raise RuntimeError('Training reports completion but the requested checkpoint/history is missing')
                    event('waiting', iteration=iteration, reason=waiting, training=health,
                          checkpoint_exists=(run / ('policy_iter%02d.pt' % iteration)).exists())
                    time.sleep(min(POLL_SECONDS, max(0, deadline - time.monotonic())))
                verify_frozen(startup)
                training_health(run)
                if completed_checkpoint(run, iteration)[0] != binding: raise ValueError('Checkpoint binding changed before launch')
                if time.monotonic() >= deadline: raise TimeoutError('Eight-hour queue deadline reached before launch')
                out = R / 'results' / name
                if out.exists(): raise FileExistsError('Evaluation output appeared before launch: ' + str(out))
                row = dict(iteration=iteration, name=name, command=command, started_at=utc(), checkpoint_binding=binding)
                launched.append(row); event('launch', **row)
                log_path = audit / ('iter%02d.stdout_stderr.log' % iteration)
                try:
                    with log_path.open('x') as log:
                        process = subprocess.Popen(command, cwd=str(P), stdout=log, stderr=subprocess.STDOUT,
                                                   start_new_session=True)
                        row['pid'] = process.pid
                        while process.poll() is None:
                            if time.monotonic() >= deadline: raise TimeoutError('Eight-hour deadline reached during evaluation')
                            time.sleep(min(POLL_SECONDS, max(0, deadline - time.monotonic())))
                        row['exit_code'] = process.returncode
                    row['finished_at'] = utc(); row['log_sha256'] = sha(log_path)
                    for filename in ('manifest.json', 'summary.json', 'failure.json'):
                        path = out / filename
                        row[filename.replace('.', '_') + '_sha256'] = sha(path) if path.exists() else None
                    if row['exit_code'] != 0: raise RuntimeError('Formal evaluation exited nonzero; queue stopped')
                    row.update(inspect_evaluation(out, args.kind, binding, startup))
                    verify_frozen(startup)
                    if completed_checkpoint(run, iteration)[0] != binding: raise ValueError('Checkpoint/history changed during evaluation')
                    event('evaluation_completed', **row)
                except BaseException as error:
                    if process is not None: stop_owned_evaluation(process)
                    row.update(error=repr(error), finished_at=utc(), exit_code=None if process is None else process.poll())
                    row['log_sha256'] = sha(log_path) if log_path.exists() else None
                    for filename in ('manifest.json', 'summary.json', 'failure.json'):
                        path = out / filename
                        row[filename.replace('.', '_') + '_sha256'] = sha(path) if path.exists() else None
                    raise
                finally:
                    write_new(audit / ('iter%02d.json' % iteration), row)
                    process = None
            final['status'] = 'completed_fixed_queue'
        except BaseException as error:
            final.update(status='stopped', error=repr(error))
            event('queue_stopped', error=repr(error))
        final.update(finished_at=utc(), seconds=time.monotonic() - started, evaluations=launched)
        write_new(audit / 'summary.json', final)
        event('queue_finished', status=final['status'], summary_sha256=sha(audit / 'summary.json'))
    return audit, final['status'] == 'completed_fixed_queue'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('ppo', 'world_ppo'), required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--iterations', type=int, nargs='+', required=True)
    parser.add_argument('--name-prefix', required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args(); audit, passed = run_schedule(args)
    print(json.dumps(dict(completed=passed, audit=str(audit))), flush=True)
    if not passed: raise SystemExit(1)


if __name__ == '__main__':
    main()
