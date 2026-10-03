"""Actual CUDA smoke-checkpoint checks; no performance or selection claim."""
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from iql_wide import IQL
from iql_wide_worker import Worker
from prepare_iql_replay import R, Replay, history_windows, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='IQL_wide_smoke')
    args = parser.parse_args()
    if Path(args.run).name != args.run:
        raise ValueError('Expected a simple run name')
    run = R/'results'/args.run
    checkpoint = run/'smoke_step4.pt'
    before = sha(checkpoint)
    state = torch.load(checkpoint, map_location='cpu')
    completion = json.loads((run/'completion.json').read_text())
    assert completion['status'] == 'smoke_completed' and completion['steps'] == 4
    assert state['step'] == 4 and state['smoke'] and not state['final']
    assert state['config']['seed'] == 260915
    torch.manual_seed(260915)
    initial = IQL(state['config']).state_dict()
    changed = {}
    for prefix in ('actor.', 'q1.', 'q2.', 'value.', 'q1_target.', 'q2_target.'):
        values = [not torch.equal(value, initial[name]) for name, value in state['model'].items()
                  if name.startswith(prefix)]
        assert values and any(values), prefix
        changed[prefix] = int(sum(values))
    assert all(torch.isfinite(value).all() for value in state['model'].values())
    optimizer_steps = {}
    for name in ('actor', 'q', 'value'):
        steps = sorted(set(int(s['step']) for s in state['optimizers'][name]['state'].values()))
        assert steps == [4], (name, steps)
        optimizer_steps[name] = steps
    assert state['optimizers']['actor_scheduler']['last_epoch'] == 4
    replay = Replay(R/state['config']['replay'], state['config'])
    indices = np.array([0, replay.size-1])
    histories = history_windows(replay.arrays['frames'], replay.arrays['start'][indices])
    anchors = replay.arrays['anchor'][indices]
    worker = Worker(checkpoint.resolve(), allow_smoke=True)
    one = worker.evaluate(histories, anchors)
    two = worker.evaluate(histories, anchors)
    single = worker.evaluate(histories[:1], anchors[:1])
    assert one['actions'] == two['actions']
    assert np.allclose(single['actions'], one['actions'][:1], atol=1e-6, rtol=0)
    actions = np.asarray(one['actions'])
    assert actions.shape == (2,) and np.isfinite(actions).all()
    assert ((actions >= 0) & (actions <= np.minimum(20, 2*anchors)+1e-8)).all()
    rejected = False
    try:
        worker.evaluate(histories[:, :-1], anchors)
    except ValueError:
        rejected = True
    assert rejected
    assert all(p.grad is None and not p.requires_grad for p in worker.agent.parameters())
    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    assert sha(checkpoint) == before
    result = dict(status='passed', run=args.run, checkpoint_sha256=before,
                  replay_manifest_sha256=sha(replay.directory/'manifest.json'),
                  transitions=replay.size, optimizer_steps=optimizer_steps,
                  changed_parameter_counts=changed, all_parameters_finite=True,
                  repeated_actions_exact=True, batched_actions_atol_u_h=1e-6,
                  invalid_shape_rejected=True, checkpoint_unchanged=True,
                  evaluation_frozen=True, TF32=False, actions=one['actions'],
                  source_sha256=sha(Path(__file__).resolve()), performance_claim=False)
    target = R/'checks'/('iql_mechanics_'+args.run+'.json')
    with target.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
