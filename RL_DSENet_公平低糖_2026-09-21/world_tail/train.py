"""Fixed-budget adapter/response repair. No development outcomes are read."""
import time
from common import R, P, sha, load_world, gpu, np, torch, json


def main():
    c = json.loads((R/'configs/train.json').read_text())
    manifest = json.loads((R/'cache/manifest.json').read_text())
    assert manifest['config_sha256'] == sha(R/'configs/train.json')
    for name, digest in manifest['cache_sha256'].items(): assert sha(R/'cache'/name) == digest
    assert sha(P/c['initial_world']) == c['initial_sha256']
    torch.manual_seed(c['seed']); torch.set_num_threads(2)
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    rng = np.random.default_rng(c['seed']); world, old = load_world(P/c['initial_world'])
    trainable = [n for n, p in world.named_parameters() if p.requires_grad]
    assert all(n.split('.')[0] in c['trainable'] for n in trainable)
    frozen = {k: v.detach().cpu().clone() for k, v in world.state_dict().items() if k.startswith(('forecast.', 'history_encoder.'))}
    a = gpu(dict(np.load(R/'cache/s1.npz'))); b = gpu(dict(np.load(R/'cache/s2.npz')))
    tail = np.flatnonzero(b['tail'].cpu().numpy()); n = len(b['z'])
    out = R/'results/W01_tail'; out.mkdir(exist_ok=False)
    (out/'config.json').write_text(json.dumps(c, indent=2))
    provenance = dict(training_cache_manifest=manifest, source_sha256={p.name: sha(p) for p in [R/'common.py', R/'prepare.py', R/'train.py']}, initial_sha256=sha(P/c['initial_world']))
    (out/'provenance.json').write_text(json.dumps(provenance, indent=2))
    opt = torch.optim.AdamW([p for p in world.parameters() if p.requires_grad], lr=c['lr'], weight_decay=c['weight_decay'])
    start = time.time(); logs = []; arm_count = 0; step = 0
    try:
        with (out/'history.jsonl').open('w', buffering=1) as log:
            for step in range(1, c['steps']+1):
                ix = rng.integers(len(a['z']), size=c['paired_groups_per_step'])
                ref = world.reference(a['z'][ix], a['forecast'][ix], a['rate'][ix])
                prediction = world.response(a['z'][ix], a['actions'][ix, 1:]-a['actions'][ix, 0:1])
                mask = a['mask'][ix, 0]; joint = a['mask'][ix, 1:] & a['mask'][ix, 0:1]
                ref_loss = ((ref-a['targets'][ix, 0]).square()*mask).sum()/mask.sum()
                effect_loss = ((prediction-(a['targets'][ix, 1:]-a['targets'][ix, 0:1])).square()*joint).sum()/joint.sum()
                jx = np.concatenate([rng.choice(tail, size=c['ppo_tail_per_step']), rng.integers(n, size=c['ppo_origins_per_step']-c['ppo_tail_per_step'])])
                predicted = world.trajectories(b['z'][jx], b['forecast'][jx], b['rate'][jx], b['actions'][jx, None])[:, 0]
                weight = torch.where(b['targets'][jx]*18 < 70, c['low_point_weight'], 1.)
                factual_loss = ((predicted-b['targets'][jx]).square()*weight).sum()/weight.sum()
                loss = ref_loss+effect_loss+factual_loss
                assert torch.isfinite(loss)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in world.parameters() if p.requires_grad], c['gradient_clip'], error_if_nonfinite=True)
                opt.step(); logs.append([float(ref_loss), float(effect_loss), float(factual_loss)])
                arm_count += int(a['counts'][ix].sum())
                if step % 100 == 0:
                    row = dict(step=step, loss_reference=float(np.mean(logs, 0)[0]), loss_effect=float(np.mean(logs, 0)[1]),
                               loss_ppo_factual=float(np.mean(logs, 0)[2]), s1_unique_arms_sampled=arm_count,
                               s2_origins_sampled=step*c['ppo_origins_per_step'], seconds=time.time()-start)
                    logs=[]; log.write(json.dumps(row)+'\n'); print(json.dumps(row), flush=True)
        for k, value in frozen.items(): assert torch.equal(value, world.state_dict()[k].cpu()), k
        z = a['z'][:8]; same = torch.zeros((8, 1, 48), device='cuda')
        with torch.no_grad():
            assert torch.equal(world.response(z, same), same)
            changed = same.clone(); changed[:, :, 24:] = .25
            response = world.response(z, changed)
            assert torch.equal(response[:, :, :24], same[:, :, :24])
            assert torch.isfinite(response).all() and response[:, :, 24:].abs().max() > 0
        config = dict(old['config']); config['tail_repair'] = c; config['tail_provenance'] = provenance
        torch.save(dict(model=world.state_dict(), config=config, step=step), out/'world.pt')
        done = dict(agent_proposed_status='trained_evaluation_pending', steps=step, seconds=time.time()-start,
                    frozen_backbones_exact=True, zero_effect_exact=True, future_causality_exact=True,
                    world_sha256=sha(out/'world.pt'), s1_arms_sampled=arm_count, s2_origins_sampled=step*c['ppo_origins_per_step'])
        (out/'completion.json').write_text(json.dumps(done, indent=2)); print(json.dumps(done), flush=True)
    except Exception as e:
        (out/'failure.json').write_text(json.dumps(dict(step=step, error=repr(e)), indent=2)); raise


if __name__ == '__main__': main()
