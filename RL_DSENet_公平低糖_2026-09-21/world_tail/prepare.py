"""Training data only: merged S1 arms and complete S2 factual future windows."""
import time
from common import R, P, OLD, sha, history_clock, encode, load_world, np, torch, json


def main():
    c = json.loads((R/'configs/train.json').read_text())
    assert sha(P/c['initial_world']) == c['initial_sha256']
    out = R/'cache'; out.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    world, _ = load_world(P/c['initial_world'])
    groups = {}; source_hashes = {}; input_groups = input_arms = 0
    for folder in c['s1']:
        directory = OLD/folder
        manifest = json.loads((directory/'manifest.json').read_text())
        assert manifest['status'] == 'complete'
        assert manifest['contract']['scenario_seeds'] == [40101, 40102]
        source_hashes[str((directory/'manifest.json').relative_to(P))] = sha(directory/'manifest.json')
        for item in manifest['scenarios']:
            path = directory/item['file']; assert sha(path) == item['sha256']
            source_hashes[str(path.relative_to(P))] = sha(path)
            with np.load(path) as d:
                for i in range(len(d['features'])):
                    hist = d['features'][i, 0, :72]
                    assert np.array_equal(d['features'][i, :, :72], np.broadcast_to(hist, d['features'][i, :, :72].shape))
                    ref = [d[k][i, 0] for k in ['action', 'target', 'mask']]
                    import hashlib
                    key = hashlib.sha256(hist.tobytes()+b''.join(v.tobytes() for v in ref)).hexdigest()
                    if key not in groups: groups[key] = dict(history=hist.copy(), arms={})
                    g = groups[key]; input_groups += 1
                    for arm in range(d['action'].shape[1]):
                        values = [d[k][i, arm].copy() for k in ['action', 'target', 'mask']]
                        assert np.array_equal(values[2], np.arange(48) < d['length'][i, arm])
                        armkey = hashlib.sha256(b''.join(v.tobytes() for v in values)).hexdigest()
                        g['arms'].setdefault(armkey, values); input_arms += 1
    gs = list(groups.values()); max_arms = max(len(g['arms']) for g in gs)
    actions = np.zeros((len(gs), max_arms, 48), dtype='float32')
    targets = np.zeros_like(actions); masks = np.zeros_like(actions, dtype=bool)
    counts = []
    for i, g in enumerate(gs):
        counts.append(len(g['arms']))
        for j, values in enumerate(g['arms'].values()):
            actions[i, j], targets[i, j], masks[i, j] = values
    enc = encode(world, history_clock(np.stack([g['history'] for g in gs])))
    assert np.allclose(actions[:, 0][masks[:, 0]], np.broadcast_to(enc['rate'][:, None], actions[:, 0].shape)[masks[:, 0]], atol=1e-5)
    np.savez_compressed(out/'s1.npz', **enc, actions=actions, targets=targets, mask=masks, counts=np.array(counts))

    replay = R.parent/'shared_replay'
    manifest = json.loads((replay/'manifest.json').read_text()); assert manifest['status'] == 'complete'
    for name, digest in manifest['array_sha256'].items(): assert sha(replay/name) == digest, name
    assert sha(replay/'episodes.json') == manifest['episodes_sha256']
    episodes = json.loads((replay/'episodes.json').read_text())
    a = {k: np.load(replay/(k+'.npy'), mmap_mode='r') for k in ['features', 'start', 'action_u_h', 'cgm_mmol_l', 'episode_remaining', 'episode_id']}
    rows = []
    for ep in episodes:
        sources = [s for s in ep['sources'] if s['role'] == 'PPO_real_rewards']
        if not sources: continue
        assert all(1 <= s['iteration'] <= 8 and 930001 <= s['scenario_seed'] <= 930016 for s in sources)
        indices = np.arange(ep['transition_offset'], ep['transition_offset']+ep['length'])
        rows.extend(indices[a['episode_remaining'][indices] >= 48].tolist())
    indices = np.array(rows, dtype='int64'); assert len(indices) > 0
    positions = indices[:, None]+np.arange(48)[None]
    assert np.array_equal(a['episode_id'][positions], np.broadcast_to(a['episode_id'][indices, None], positions.shape))
    targets = np.asarray(a['cgm_mmol_l'][positions]); actions = np.asarray(a['action_u_h'][positions])
    zparts = []; fparts = []; rparts = []; current = []; start = time.time()
    norm = json.loads((P/'Loop数据集/训练管线_v2/prepared/normalization.json').read_text())['cgm_mmol_l']
    for i in range(0, len(indices), 128):
        starts = a['start'][indices[i:i+128]]
        hist = history_clock(a['features'][starts[:, None]+np.arange(-71, 1)[None]])
        assert (hist[:, -1, 5] > .5).all()
        current.extend(((hist[:, -1, 0]*norm['scale']+norm['mean'])*18).tolist())
        enc = encode(world, hist)
        zparts.append(enc['z']); fparts.append(enc['forecast']); rparts.append(enc['rate'])
        if i % 12800 == 0: print(json.dumps(dict(encoded=i, total=len(indices), seconds=time.time()-start)), flush=True)
    minimum = targets.min(-1)*18
    tail = (minimum < c['tail_threshold_mg_dl']) | (np.array(current)-minimum >= c['tail_drop_mg_dl'])
    assert tail.any()
    np.savez_compressed(out/'s2.npz', z=np.concatenate(zparts), forecast=np.concatenate(fparts), rate=np.concatenate(rparts),
                        actions=actions, targets=targets, indices=indices, tail=tail, current_cgm_mg_dl=np.array(current))
    report = dict(s1_input_groups=input_groups, s1_input_arms=input_arms, s1_unique_groups=len(gs),
                  s1_unique_arms=sum(counts), s1_max_unique_arms=max_arms,
                  s1_deduplicated_reference_groups=input_groups-len(gs),
                  s2_origins=len(indices), s2_tail=int(tail.sum()), s2_min54=int((minimum < 54).sum()),
                  s2_min70=int((minimum < 70).sum()), all_s2_labels_real_48step=True,
                  s1_source_sha256=source_hashes, shared_manifest_sha256=sha(replay/'manifest.json'),
                  config_sha256=sha(R/'configs/train.json'), initial_world_sha256=sha(P/c['initial_world']),
                  cache_sha256={p.name: sha(p) for p in out.glob('*.npz')}, source_sha256=sha(__file__))
    (out/'manifest.json').write_text(json.dumps(report, indent=2)); print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
