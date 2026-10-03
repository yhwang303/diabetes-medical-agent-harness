"""Conditional fixed three-epoch BC only; no PPO rollout, simulator or evaluation."""
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import argparse
import json
from pathlib import Path
import random
import shutil
import time

from planner_bc_common import (BC_CONFIG, PLAN_TO_ACTION, MULTIPLIERS, R, P, SEED,
    inspect_cache, state_sha, runtime_seed, verify_files, write_new, sha, agreement_gate)


def load_features(directory, manifest):
    import numpy as np
    chunks = {k: [] for k in ('features', 'teacher_action_index', 'teacher_plan_index',
                              'teacher_action_u_h', 'teacher_plan_cost', 'anchor_u_h', 'origin_minute')}
    parts = []; episodes = []
    for episode, item in enumerate(manifest['files']):
        path = directory / item['file']
        if sha(path) != item['sha256']: raise ValueError('Feature file changed')
        with np.load(path, allow_pickle=False) as packed:
            if set(packed.files) != set(chunks): raise ValueError('Unexpected feature cache fields')
            values = {k: packed[k] for k in chunks}
        n = item['count']
        for k, x in values.items():
            expected_shape = (n, 195) if k == 'features' else (n, 9) if k == 'teacher_plan_cost' else (n,)
            expected_type = np.int64 if k.endswith('_index') else np.float64 if k == 'teacher_plan_cost' else np.float32
            if x.shape != expected_shape or x.dtype != expected_type or not np.isfinite(x).all():
                raise ValueError('Cached array shape/dtype/value mismatch: ' + k)
        plan = values['teacher_plan_index']; label = values['teacher_action_index']; a = values['anchor_u_h']
        if (np.any((plan < 0) | (plan > 8)) or np.any((label < 0) | (label > 8))
                or np.any((a <= 0) | (a > 20)) or np.any(np.diff(values['origin_minute']) <= 0)
                or not np.array_equal(label, np.asarray(PLAN_TO_ACTION)[plan])
                or not np.array_equal(plan, values['teacher_plan_cost'].argmin(-1))
                or not np.array_equal(values['teacher_action_u_h'], np.clip(a * np.asarray(MULTIPLIERS, dtype=np.float32)[label], 0, 20))):
            raise ValueError('Cached teacher choice/action/origin contract mismatch')
        for k, x in values.items(): chunks[k].append(x)
        parts.extend([item['part'] == 'fit'] * n); episodes.extend([episode] * n)
    data = {k: np.concatenate(v) for k, v in chunks.items()}
    data['fit'] = np.asarray(parts, dtype=bool); data['episode_index'] = np.asarray(episodes, dtype=np.int64)
    if not data['fit'].any() or data['fit'].all(): raise ValueError('Need both whole-scene fit and internal-check states')
    return data


def logits_for(policy, features):
    import numpy as np
    import torch
    with torch.no_grad():
        output = [policy(torch.as_tensor(features[i:i + 512], device='cuda')).cpu().numpy()
                  for i in range(0, len(features), 512)]
    result = np.concatenate(output)
    if result.shape != (len(features), 9) or not np.isfinite(result).all(): raise ValueError('Invalid actor logits')
    return result


def internal_diagnostics(logits, data, files, indices):
    """Train-internal first-action agreement; neither BG validation nor calibration."""
    import numpy as np
    z = logits.astype(np.float64); z -= z.max(-1, keepdims=True)
    probability = np.exp(z); probability /= probability.sum(-1, keepdims=True)
    chosen = logits.argmax(-1); label = data['teacher_action_index'][indices]
    anchor = data['anchor_u_h'][indices]; teacher = data['teacher_action_u_h'][indices]
    action = np.clip(anchor * np.asarray(MULTIPLIERS, dtype=np.float32)[chosen], 0, 20)
    tol = BC_CONFIG['action_tolerance_u_h']
    matched = np.abs(action.astype(np.float64) - teacher) <= tol
    nonhold = np.abs(teacher.astype(np.float64) - anchor) > tol
    episodes = data['episode_index'][indices]
    nonhold_episodes = sorted(set(episodes[nonhold].tolist()))
    full_episodes = [i for i in nonhold_episodes if files[i]['full_source_episode']]
    recalls = [float((chosen[label == k] == k).mean()) if np.any(label == k) else None for k in range(9)]
    sorted_p = np.sort(probability, axis=1)
    overall = float(matched.mean()); nonhold_agreement = float(matched[nonhold].mean()) if nonhold.any() else None
    per_episode = []
    for i in sorted(set(episodes.tolist())):
        keep = episodes == i; subset = keep & nonhold
        per_episode.append(dict(file=files[i]['file'], states=int(keep.sum()), full_source_episode=files[i]['full_source_episode'],
            action_agreement=float(matched[keep].mean()), nonhold_states=int(subset.sum()),
            nonhold_agreement=float(matched[subset].mean()) if subset.any() else None))
    report = dict(scope='train_internal_103004_not_independent_world_validation', states=len(indices),
        overall_action_agreement=overall, class_index_agreement=float((chosen == label).mean()),
        nonhold_action_agreement=nonhold_agreement, nonhold_states=int(nonhold.sum()),
        nonhold_source_episodes=len(nonhold_episodes), full_nonhold_source_episodes=len(full_episodes),
        actual_action_mae_u_h=float(np.abs(action.astype(np.float64) - teacher).mean()),
        teacher_action_counts=np.bincount(label, minlength=9).tolist(), actor_action_counts=np.bincount(chosen, minlength=9).tolist(),
        teacher_plan_counts=np.bincount(data['teacher_plan_index'][indices], minlength=9).tolist(),
        class_index_recall=recalls, mean_hold_probability=float(probability[:, 4].mean()),
        mean_entropy=float(-(probability * np.log(np.maximum(probability, 1e-300))).sum(-1).mean()),
        mean_probability_margin=float((sorted_p[:, -1] - sorted_p[:, -2]).mean()),
        episodes=per_episode, gate=agreement_gate(overall, nonhold_agreement, int(nonhold.sum()), len(full_episodes)))
    return report, dict(logits=logits, probabilities=probability, actor_action_index=chosen,
        actor_action_u_h=action, teacher_action_index=label, teacher_action_u_h=teacher,
        sample_index=indices, episode_index=episodes, origin_minute=data['origin_minute'][indices])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    directory, prepared = inspect_cache(args.cache)
    c = prepared['binding']['policy_config']; out = Path(c['output_dir']).resolve()
    out.relative_to((R / 'results').resolve()); out.mkdir(parents=True, exist_ok=False)
    begin = time.time()
    try:
        write_new(out / 'config.json', c)  # Original Trainer contract, not an assertion that PPO ran.
        import numpy as np
        import torch
        from ppo_world_worker import Trainer
        trainer = Trainer((out / 'config.json').resolve())
        runtime_seed()
        initial = dict(world=state_sha(trainer.world), actor=state_sha(trainer.policy), value=state_sha(trainer.value))
        if (trainer.provenance != prepared['original_trainer_provenance']
                or initial['world'] != prepared['world_state_after']
                or initial['actor'] != prepared['actor_state_after'] or initial['value'] != prepared['value_state_after']):
            raise ValueError('Actual loaded model/runtime differs from frozen cache generator')
        data = load_features(directory, prepared)
        fit = np.flatnonzero(data['fit']); check = np.flatnonzero(~data['fit'])
        if any(p.requires_grad for p in trainer.world.parameters()): raise ValueError('World was not frozen')
        sources = prepared['binding']['source_sha256']
        provenance = dict(method='planner_behavior_cloning_only', ppo_iterations=0, simulator_transitions=0,
            config=c, bc_config=BC_CONFIG, training_seed=SEED, cache_path=str(directory), cache_manifest_sha256=sha(directory / 'manifest.json'),
            source_sha256=sources, frozen_state_initial=initial, fit_states=len(fit), internal_check_states=len(check),
            original_trainer_provenance=trainer.provenance, teacher_config=prepared['binding']['teacher_config'],
            model_imagined_or_real_rollouts_used_for_BC=False, source_histories_are_real_train_rollouts=True,
            label_source='frozen_risk_MPC_first_action_only', development_or_confirmation_loaded=False)
        write_new(out / 'provenance.json', provenance)
        optimizer = torch.optim.Adam(trainer.policy.parameters(), lr=BC_CONFIG['learning_rate'])
        updates = 0
        with (out / 'bc_history.jsonl').open('x', buffering=1) as log:
            for epoch in range(1, BC_CONFIG['epochs'] + 1):
                order = fit[torch.randperm(len(fit)).numpy()]; total = 0.; seen = 0
                for start in range(0, len(order), BC_CONFIG['batch_size']):
                    ix = order[start:start + BC_CONFIG['batch_size']]
                    x = torch.as_tensor(data['features'][ix], device='cuda')
                    label = torch.as_tensor(data['teacher_action_index'][ix], device='cuda')
                    logp = trainer.policy(x).log_softmax(-1)
                    eps = BC_CONFIG['label_smoothing']
                    loss = -(1 - eps) * logp.gather(1, label[:, None]).mean() - eps * logp.mean()
                    if not torch.isfinite(loss): raise ValueError('Nonfinite BC loss')
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainer.policy.parameters(), BC_CONFIG['gradient_clip'], error_if_nonfinite=True)
                    optimizer.step(); updates += 1; seen += len(ix); total += float(loss) * len(ix)
                if seen != len(fit): raise ValueError('Incomplete BC epoch')
                item = dict(epoch=epoch, updates=updates, seen=seen, mean_loss=total / seen,
                            actor_state_sha256=state_sha(trainer.policy), seconds=time.time() - begin)
                log.write(json.dumps(item, allow_nan=False) + '\n'); print(json.dumps(item), flush=True)
        trainer.policy.eval(); trainer.value.eval()
        final = dict(world=state_sha(trainer.world), actor=state_sha(trainer.policy), value=state_sha(trainer.value))
        if final['world'] != initial['world'] or final['value'] != initial['value'] or final['actor'] == initial['actor']:
            raise ValueError('Expected actor-only state change')
        if (trainer.buffer or trainer.count or trainer.iteration or trainer.opt.state or trainer.vo.state
                or any(p.grad is not None for p in trainer.world.parameters())):
            raise ValueError('BC contaminated frozen world or PPO state')
        logits = logits_for(trainer.policy, data['features'][check])
        diagnostics, predictions = internal_diagnostics(logits, data, prepared['files'], check)
        write_new(out / 'internal_check.json', diagnostics)
        with (out / 'internal_check_predictions.npz').open('xb') as stream: np.savez_compressed(stream, **predictions)
        # Reset exactly one seed for the future PPO handoff; retain fresh empty PPO Adam states.
        runtime_seed()
        state = dict(config=c, provenance=trainer.provenance, model=trainer.policy.state_dict(), value=trainer.value.state_dict(),
            opt=dict(actor=trainer.opt.state_dict(), value=trainer.vo.state_dict()), iteration=0, simulator_transitions=0,
            rng_python=random.getstate(), rng_numpy=np.random.get_state(), rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all(),
            bc_initialization=dict(method='planner_behavior_cloning_only', epochs=3, updates=updates,
                provenance_sha256=sha(out / 'provenance.json'), cache_manifest_sha256=sha(directory / 'manifest.json'),
                actual_state_sha256=final, internal_gate=diagnostics['gate'], true_RL_training_performed=False),
            bc_optimizer_state=optimizer.state_dict())
        candidate = out / 'bc_final_candidate.pt'
        with candidate.open('xb') as stream: torch.save(state, stream)
        # This is an actual original-Trainer load, not a key-name compatibility assertion.
        restored = Trainer((out / 'config.json').resolve(), candidate.resolve())
        restored.policy.eval(); restored.value.eval()
        reloaded = dict(world=state_sha(restored.world), actor=state_sha(restored.policy), value=state_sha(restored.value))
        replayed = logits_for(restored.policy, data['features'][check])
        max_error = float(np.max(np.abs(replayed.astype(np.float64) - logits)))
        if (reloaded != final or max_error > 1e-6 or restored.iteration != 0 or restored.count != 0
                or restored.opt.state or restored.vo.state or restored.buffer):
            raise ValueError('Original Trainer failed state/load/logit compatibility')
        compatibility = dict(actual_original_Trainer_load=True, all_model_states_exact=True,
            all_internal_check_logits_compared=True, max_abs_logit_error=max_error, tolerance=1e-6,
            original_PPO_optimizers_empty=True, iteration=0, simulator_transitions=0,
            existing_formal_evaluator_accepts_BC=False, automatic_PPO_start=False)
        write_new(out / 'compatibility.json', compatibility)
        verify_files(prepared['binding']['bound_file_sha256'])
        if sha(directory / 'manifest.json') != provenance['cache_manifest_sha256']: raise ValueError('Cache manifest changed')
        for item in prepared['files']:
            if sha(directory / item['file']) != item['sha256']: raise ValueError('Cache array changed during training')
        passed = diagnostics['gate']['passed']
        if passed:
            with candidate.open('rb') as src, (out / 'bc_init.pt').open('xb') as dest: shutil.copyfileobj(src, dest)
        write_new(out / 'completion.json', dict(status='BC_completed_internal_gate_passed' if passed else 'BC_completed_internal_gate_failed',
            training_method='behavior_cloning_only', epochs=3, updates=updates, ppo_iterations=0, simulator_transitions=0,
            candidate_sha256=sha(candidate), bc_init_sha256=sha(out / 'bc_init.pt') if passed else None,
            provenance_sha256=sha(out / 'provenance.json'), config_sha256=sha(out / 'config.json'),
            internal_check_sha256=sha(out / 'internal_check.json'), predictions_sha256=sha(out / 'internal_check_predictions.npz'),
            compatibility_sha256=sha(out / 'compatibility.json'), state_sha256=final,
            seconds=time.time() - begin, gate=diagnostics['gate'], clinical_validation=False))
        print(json.dumps(dict(status='BC_completed', gate_passed=passed, output=str(out), no_true_RL_run=True)), flush=True)
    except BaseException as error:
        write_new(out / 'failure.json', dict(error=repr(error), seconds=time.time() - begin))
        raise
    if not passed: raise SystemExit(2)


if __name__ == '__main__':
    main()
