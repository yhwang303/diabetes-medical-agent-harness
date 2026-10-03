"""Evaluation-only probability-mean deployment of an existing PPO checkpoint.

This post-development mechanism ablation is NOT the original argmax policy or
retraining. It averages the nine executable actions, after each action's 20 U/h
cap. Original Trainer initialization strictly restores the original features,
actor/config/provenance; no Trainer act/update method is called. The evaluation
gate, not this adapter, authorizes formal checkpoints 8/16/32/40 and data splits.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

R = Path(__file__).resolve().parent
P = R.parent
RATIOS = [0., .25, .5, .75, 1., 1.25, 1.5, 1.75, 2.]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()


def checkpoint_binding(config, checkpoint):
    """Bind only the completed history prefix; later training appends are allowed."""
    run = config.parent.resolve(); run.relative_to((R / 'results').resolve())
    c = json.loads(config.read_text())
    if config.name != 'config.json' or checkpoint.parent.resolve() != run or c['name'] != run.name:
        raise ValueError('Use the checkpoint and resolved config.json from the same PPO run')
    number = checkpoint.stem[len('policy_iter'):]
    if (not checkpoint.stem.startswith('policy_iter') or not number.isdigit()
            or checkpoint.name != 'policy_iter%02d.pt' % int(number)):
        raise ValueError('Expected a completed policy_iterNN.pt checkpoint')
    iteration = int(number)
    provenance_path = run / 'provenance.json'
    training = json.loads(provenance_path.read_text())
    if (c['seed'] != 260915 or training['training_seed'] != 260915 or training['config'] != c
            or not 1 <= iteration <= c['iterations']):
        raise ValueError('Training config/provenance/iteration binding mismatch')
    prefix = []
    with (run / 'history.jsonl').open('rb') as stream:
        for expected in range(1, iteration + 1):
            line = stream.readline()
            if not line.endswith(b'\n'): raise ValueError('Incomplete checkpoint history prefix')
            record = json.loads(line)
            if type(record['iteration']) is not int or record['iteration'] != expected:
                raise ValueError('Training history is not a contiguous completed prefix')
            prefix.append(line)
    if (not Path(record['checkpoint']).is_absolute() or Path(record['checkpoint']).resolve() != checkpoint
            or record['checkpoint_sha256'] != sha(checkpoint)):
        raise ValueError('Checkpoint bytes are not bound to the completed training update')
    raw = b''.join(prefix)
    return dict(iteration=iteration, checkpoint_sha256=record['checkpoint_sha256'],
                config_sha256=sha(config), training_provenance_sha256=sha(provenance_path),
                history_prefix_sha256=hashlib.sha256(raw).hexdigest(), history_prefix_bytes=len(raw))


def validate_request(request):
    if (not isinstance(request, dict) or set(request) != {'op', 'history', 'anchors'}
            or request['op'] != 'evaluate'):
        raise ValueError('Only op=evaluate with history and anchors is accepted; no training, labels or hidden fields')


def distribution_summary(logits, anchors, ratios):
    """FP64 scalar softmax/expectation of original FP32 actor logits, no sampling.

    Nine values per history make CPU audit aggregation negligible. This pure
    function can be checked without Torch; the live actor still runs unchanged.
    """
    if ratios != RATIOS or not logits or len(logits) != len(anchors):
        raise ValueError('Expected a nonempty batch and the unchanged nine-action grid')
    result = {key: [] for key in (
        'actions', 'action_mean', 'probabilities', 'actual_action_grid_u_h', 'expected_ratio',
        'raw_ratio_mean_rate', 'argmax_indices', 'argmax_actions_u_h', 'argmax_probabilities',
        'hold_probability', 'probability_margin_top2', 'entropy', 'mean_minus_argmax_u_h',
        'arithmetic_boundary_adjustment_u_h')}
    for row, anchor in zip(logits, anchors):
        if (len(row) != 9 or not all(math.isfinite(value) for value in row)
                or not math.isfinite(anchor) or not 0 < anchor <= 20):
            raise ValueError('Expected finite nine logits and a positive observed anchor <=20 U/h')
        high = max(row); exponentials = [math.exp(value - high) for value in row]
        denominator = math.fsum(exponentials)
        probabilities = [value / denominator for value in exponentials]
        actual = [min(20., anchor * ratio) for ratio in ratios]
        expected_ratio = math.fsum(p * ratio for p, ratio in zip(probabilities, ratios))
        mean = math.fsum(p * action for p, action in zip(probabilities, actual))
        # Floating-point convex sums can exceed an endpoint by a few ulps only.
        action = min(20., max(0., mean))
        if abs(action - mean) > 1e-12: raise ValueError('Mean escaped the executable convex action range')
        index = max(range(9), key=lambda k: row[k])  # Same first-index tie rule as logits.argmax.
        ordered = sorted(probabilities, reverse=True)
        values = dict(actions=action, action_mean=action, probabilities=probabilities,
            actual_action_grid_u_h=actual, expected_ratio=expected_ratio,
            raw_ratio_mean_rate=anchor * expected_ratio, argmax_indices=index,
            argmax_actions_u_h=actual[index], argmax_probabilities=probabilities[index],
            hold_probability=probabilities[4], probability_margin_top2=ordered[0] - ordered[1],
            entropy=-math.fsum(p * math.log(p) for p in probabilities if p > 0),
            mean_minus_argmax_u_h=action - actual[index], arithmetic_boundary_adjustment_u_h=action - mean)
        for key, value in values.items(): result[key].append(value)
    return result


class MeanController:
    def __init__(self, family, config, checkpoint):
        if not config.is_absolute() or not checkpoint.is_absolute():
            raise ValueError('Configuration and checkpoint paths must be absolute')
        config = config.resolve(); checkpoint = checkpoint.resolve()
        binding = checkpoint_binding(config, checkpoint)
        import torch
        if family == 'wide':
            from ppo_wide_worker import Trainer
        elif family == 'world':
            from ppo_world_worker import Trainer
        else:
            raise ValueError('Unknown original PPO family')
        # Preserve the family's original precision/backend choices; do not introduce a TF32 ablation.
        trainer = Trainer(config, checkpoint)
        if trainer.iteration != binding['iteration'] or checkpoint_binding(config, checkpoint) != binding:
            raise ValueError('Checkpoint/config/history binding changed while loading the original Trainer')
        if (trainer.policy[0].in_features != trainer.provenance['input_dim']
                or trainer.policy[-1].out_features != 9 or trainer.multipliers.cpu().tolist() != RATIOS):
            raise ValueError('Original actor head/input/action-grid contract mismatch')
        if trainer.buffer: raise ValueError('Original Trainer unexpectedly created rollout entries during restoration')
        for module in (trainer.policy, trainer.value, trainer.world):
            module.eval().requires_grad_(False)
        # Strict restoration above necessarily constructs these, but deployment retains no optimizer/buffer.
        del trainer.opt, trainer.vo, trainer.buffer
        self.trainer = trainer; self.torch = torch
        self.ratios = trainer.multipliers.cpu().tolist()
        worker = R / ('ppo_wide_worker.py' if family == 'wide' else 'ppo_world_worker.py')
        self.ready = dict(ready=True, kind='ppo_mean', family=family, **binding,
            feature_dim=trainer.policy[0].in_features, provenance=trainer.provenance,
            deployment_rule='probability_mean_of_capped_executable_grid',
            original_deployment_rule='argmax', same_checkpoint_new_deployment_ablation=True,
            proposed_after_ppo08_development_results=True, original_preregistration=False,
            training_enabled=False, rollout_buffer_retained=False, random_action_sampling=False,
            source_sha256={str(path.relative_to(P)): sha(path) for path in (Path(__file__).resolve(), worker)},
            runtime=dict(torch=torch.__version__, torch_threads=torch.get_num_threads(),
                         actor_forward='original FP32 feature/actor computation', softmax_and_mean='Python float64',
                         matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                         cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                         cudnn_deterministic=torch.backends.cudnn.deterministic))

    def evaluate(self, request):
        validate_request(request)
        with self.torch.no_grad():
            features, anchor = self.trainer.features(request['history'], request['anchors'])
            logits = self.trainer.policy(features)
            if logits.ndim != 2 or logits.shape != (len(anchor), 9) or not self.torch.isfinite(logits).all():
                raise ValueError('Original actor returned invalid logits')
            return distribution_summary(logits.double().cpu().tolist(), anchor.double().cpu().tolist(), self.ratios)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=('wide', 'world'), required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    controller = MeanController(args.family, args.config, args.checkpoint)
    print(json.dumps(controller.ready, allow_nan=False), flush=True)
    for line in sys.stdin:
        try:
            result = controller.evaluate(json.loads(line))
        except Exception as error:
            print(json.dumps(dict(error=repr(error))), flush=True)
            raise
        print(json.dumps(result, allow_nan=False), flush=True)
    # EOF is the only shutdown operation; no act/update/close request is accepted.


if __name__ == '__main__':
    main()
