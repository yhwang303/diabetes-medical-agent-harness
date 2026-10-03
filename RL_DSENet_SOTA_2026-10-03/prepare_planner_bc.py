"""Conditional train-only planner labels/features. Never starts a simulator or RL."""
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import argparse
import json
from pathlib import Path
import time

from planner_bc_common import (R, BC_CONFIG, FIT_SEEDS, inspect_inputs, observed_pack,
                               map_teacher, state_sha, verify_files, write_new, sha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--natural-train', type=Path, required=True)
    parser.add_argument('--world-checkpoint', type=Path, required=True)
    parser.add_argument('--name', required=True, help='New PPO_world_bc... cache and eventual BC run name')
    args = parser.parse_args()
    binding = inspect_inputs(args.natural_train, args.world_checkpoint, args.name)
    out = R / 'cache' / args.name
    if Path(binding['policy_config']['output_dir']).exists(): raise FileExistsError('Eventual BC run already exists')
    out.mkdir(parents=True, exist_ok=False)
    begin = time.time()
    try:
        write_new(out / 'policy_config.json', binding['policy_config'])
        write_new(out / 'input_binding.json', binding)
        import numpy as np
        import torch
        from ppo_world_worker import Trainer, INPUT_DIM, MULTIPLIERS
        from world_control_worker import candidate_plans, score_predictions
        from planner_bc_common import MULTIPLIERS as FIXED_MULTIPLIERS, runtime_seed
        if INPUT_DIM != 195 or tuple(MULTIPLIERS) != FIXED_MULTIPLIERS: raise ValueError('Original feature/action contract changed')
        trainer = Trainer((out / 'policy_config.json').resolve())
        runtime_seed()
        if any(p.requires_grad for p in trainer.world.parameters()): raise ValueError('World must remain frozen')
        before = dict(world_state_before=state_sha(trainer.world), actor_state_before=state_sha(trainer.policy),
                      value_state_before=state_sha(trainer.value))
        files = []
        for record in binding['records']:
            h, a, minutes, full = observed_pack(binding, record)
            features = []; labels = []; chosen_all = []; actual_all = []; costs = []
            for start in range(0, len(h), 64):
                hb, ab = h[start:start + 64], a[start:start + 64]
                with torch.no_grad():
                    # Reuse the original 195 features without changing its 60-minute plans.
                    x, anchors = trainer.features(hb, ab)
                    plans = candidate_plans(ab)  # Existing teacher's separate 60/120-minute plans.
                    cache = trainer.world.encode(torch.as_tensor(hb, device='cuda'), anchors)
                    prediction = trainer.world.predict(cache, torch.as_tensor(plans, device='cuda'))
                    convert = lambda key: prediction[key].detach().cpu().numpy()
                    actions, chosen, diagnostics = score_predictions(convert('cgm_quantiles_mgdl'),
                        convert('quantile_levels'), convert('p_bg_below70'), convert('p_bg_below54'),
                        plans, ab, 'risk')
                    label, executable = map_teacher(chosen, plans, ab)
                    if not np.array_equal(actions.astype(np.float32), executable): raise ValueError('Teacher action mismatch')
                    features.append(x.cpu().numpy()); labels.append(label); chosen_all.append(chosen)
                    actual_all.append(executable)
                    costs.append(np.asarray([d['candidate_total_cost'] for d in diagnostics], dtype=np.float64))
            path = out / record['file']
            arrays = dict(features=np.concatenate(features), teacher_action_index=np.concatenate(labels),
                teacher_plan_index=np.concatenate(chosen_all).astype(np.int64),
                teacher_action_u_h=np.concatenate(actual_all), teacher_plan_cost=np.concatenate(costs),
                anchor_u_h=a, origin_minute=minutes)
            with path.open('xb') as stream: np.savez_compressed(stream, **arrays)
            files.append(dict(file=path.name, sha256=sha(path), count=len(h), job=record['job'],
                part='fit' if record['job']['seed'] in FIT_SEEDS else 'internal_check',
                full_source_episode=full, source_npz_sha256=record['sha256'], source_raw_sha256=record['source_sha256'],
                teacher_action_counts=np.bincount(arrays['teacher_action_index'], minlength=9).tolist(),
                teacher_plan_counts=np.bincount(arrays['teacher_plan_index'], minlength=9).tolist()))
            print(json.dumps(dict(event='prepared_train_episode', file=path.name, count=len(h),
                                  episodes=len(files), seconds=time.time() - begin)), flush=True)
        after = dict(world_state_after=state_sha(trainer.world), actor_state_after=state_sha(trainer.policy),
                     value_state_after=state_sha(trainer.value))
        if any(before[k + '_before'] != after[k + '_after'] for k in ('world_state', 'actor_state', 'value_state')):
            raise ValueError('Preparation changed model state')
        if (trainer.buffer or trainer.iteration != 0 or trainer.count != 0 or trainer.opt.state or trainer.vo.state
                or any(p.grad is not None for model in (trainer.world, trainer.policy, trainer.value) for p in model.parameters())):
            raise ValueError('Preparation unexpectedly performed training')
        verify_files(binding['bound_file_sha256'])
        write_new(out / 'manifest.json', dict(status='prepared_no_training', binding=binding, bc_config=BC_CONFIG,
            original_trainer_provenance=trainer.provenance, files=files, input_binding_sha256=sha(out / 'input_binding.json'),
            policy_config_sha256=sha(out / 'policy_config.json'), source=sha(Path(__file__).resolve()),
            observations_verified_against_raw=True, future_label_arrays_accessed=False,
            actual_training_or_simulation=False, seconds=time.time() - begin, **before, **after))
        print(json.dumps(dict(status='prepared_no_training', cache=str(out), manifest_sha256=sha(out / 'manifest.json'))), flush=True)
    except BaseException as error:
        write_new(out / 'failure.json', dict(error=repr(error), seconds=time.time() - begin))
        raise


if __name__ == '__main__':
    main()
