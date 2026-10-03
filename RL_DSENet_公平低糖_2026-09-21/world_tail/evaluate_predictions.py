"""Apply the unchanged exposed-development event scorer to old/new predictions."""
import csv
import importlib.util
from common import R, P, B, sha, load_world, np, torch, json
from policy_bounded import Policy


def main():
    previous = P/'RL_DSENet_局部风险_2026-09-21'
    data = np.load(previous/'analysis/point_predictions.npz')
    c = json.loads((R/'configs/train.json').read_text())
    scorer = previous/'analyze_forecasts.py'; scorer_hash = sha(scorer)
    x = torch.tensor(data['history'], device='cuda')
    anchor = torch.tensor(data['anchor'], device='cuda')
    plans = torch.tensor(data['plans'], device='cuda')
    ck = torch.load(B/'results/D06_selected_policy/policy.pt', map_location='cpu')
    actor = Policy().cuda().eval(); actor.load_state_dict(ck['policy'])
    torch.set_num_threads(2); torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    summary = {}
    for name, checkpoint in [('old', P/c['initial_world']), ('new', R/'results/W01_tail/world.pt')]:
        world, _ = load_world(checkpoint)
        with torch.no_grad():
            z, f = world.encode(x); rate = world.reference_rate(x)
            ref = world.reference(z, f, rate)
            pred = world.trajectories(z, f, rate, plans)
            q = world.utility(pred); logits = actor(z, ref, rate, anchor)
        assert torch.isfinite(pred).all()
        folder = R/'checks'/('development_'+name); (folder/'analysis').mkdir(parents=True, exist_ok=False)
        (folder/'results').mkdir()
        (folder/'results/P1_branches').symlink_to(previous/'results/P1_branches', target_is_directory=True)
        np.savez_compressed(folder/'analysis/point_predictions.npz', point_ids=data['point_ids'],
                            candidate_cgm_mg_dl=pred.cpu().numpy()*18, q=q.cpu().numpy(), logits=logits.cpu().numpy())
        spec = importlib.util.spec_from_file_location('unchanged_event_scorer', scorer)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); module.R = folder; module.main()
        assert sha(scorer) == scorer_hash
        event = json.loads((folder/'analysis/forecast_events.json').read_text())
        with (folder/'analysis/forecast_events.csv').open() as f:
            rows = list(csv.DictReader(f))
        regression = {}
        for label in ['severe', 'mild', 'negative']:
            subset = [r for r in rows if r['label'] == label]
            regression[label] = dict(arms=len(subset), mean_cgm_mae_mg_dl=float(np.mean([float(r['cgm_mae']) for r in subset])),
                                    mean_min_cgm_bias_mg_dl=float(np.mean([float(r['min_cgm_bias']) for r in subset])),
                                    predicted_low54=sum(float(r['predicted_min_cgm']) < 54 for r in subset),
                                    predicted_low70=sum(float(r['predicted_min_cgm']) < 70 for r in subset),
                                    false_positive54_cgm=sum(float(r['predicted_min_cgm']) < 54 <= float(r['actual_min_cgm']) for r in subset),
                                    false_positive70_cgm=sum(float(r['predicted_min_cgm']) < 70 <= float(r['actual_min_cgm']) for r in subset))
        summary[name] = dict(world_sha256=sha(checkpoint), event_summary=event['summary'], regression=regression)
        if name == 'old':
            summary['old_prediction_max_abs_delta_mg_dl'] = float(np.abs(pred.cpu().numpy()*18-data['candidate_cgm_mg_dl']).max())
            assert np.allclose(pred.cpu().numpy()*18, data['candidate_cgm_mg_dl'], atol=1e-4, rtol=1e-5)
    summary.update(scoring_source_sha256=scorer_hash,
                   original_development_predictions_sha256=sha(previous/'analysis/point_predictions.npz'),
                   evaluation='42 exposed event-enriched starts x 7 plans; CGM target and separate BG outcomes; correlated arms/windows, not independent population rate',
                   training_uses_these_labels=False, fixed_final_step=3000,
                   actor_retrained=False, clinical_safety_proven=False)
    (R/'checks/prediction_comparison.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__': main()
