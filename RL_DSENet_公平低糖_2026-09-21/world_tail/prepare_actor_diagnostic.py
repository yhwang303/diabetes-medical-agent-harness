"""Fixed training-state selection and exact NumPy2 persistent-anchor recovery."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

R=Path(__file__).resolve().parent
P=R.parent.parent

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    assert np.__version__ == '2.4.6', 'Use the original PPO collection .venv runtime'
    c=json.loads((R/'configs/actor_diagnostic.json').read_text())
    replay=R.parent/'shared_replay'; m=json.loads((replay/'manifest.json').read_text())
    assert m['status']=='complete' and m['ppo_episodes']==160
    assert sha(replay/'episodes.json')==m['episodes_sha256']
    arrays={}
    for name in ['features','start','episode_remaining','episode_id','cgm_mmol_l']:
        p=replay/(name+'.npy'); assert sha(p)==m['array_sha256'][p.name]
        arrays[name]=np.load(p,mmap_mode='r')
    episodes=json.loads((replay/'episodes.json').read_text());rows=[];anchors=[];mins=[];records=[]
    for ep in episodes:
        sources=[s for s in ep['sources'] if s['role']=='PPO_real_rewards']
        if not sources: continue
        assert len(sources)==1
        source=sources[0]
        assert 1<=source['iteration']<=8 and 930001<=source['scenario_seed']<=930016
        rawpath=P/source['path']; assert sha(rawpath)==m['source_sha256'][source['path']]
        raw=json.loads(rawpath.read_text()); assert raw['records'][71]['minute']==360
        first=ep['transition_offset'];start=arrays['start'][first]
        x=arrays['features'][start]; assert x[6]>.5 and x[5]>.5
        # This is exactly train_ppo.py's operation in its NumPy2 collection runtime.
        anchor=float((x[1]*.14462788945609448+.09945811581924525)*12)
        actual_warmup=raw['records'][71]['delivered_basal_u_h']
        assert abs(anchor-actual_warmup)<1e-5
        offsets=np.linspace(0,ep['length']-48,c['states_per_trajectory']).astype('int64')
        selected=first+offsets
        assert len(np.unique(selected))==c['states_per_trajectory']
        assert np.all(arrays['episode_remaining'][selected]>=48)
        assert np.all(arrays['episode_id'][selected]==ep['episode_id'])
        future=arrays['cgm_mmol_l'][selected[:,None]+np.arange(12)]*18
        rows.extend(selected.tolist()); anchors.extend([anchor]*len(selected));mins.extend(future.min(-1).tolist())
        records.append(dict(episode_id=ep['episode_id'],source=source,selected_indices=selected.tolist(),
                            anchor=anchor,actual_warmup_delivered=actual_warmup,raw_sha256=sha(rawpath)))
    assert len(records)==160 and len(rows)==c['total_states']
    out=R/'checks/actor_diagnostic';out.mkdir(exist_ok=False)
    np.savez_compressed(out/'selection.npz',indices=np.array(rows,dtype='int64'),anchors=np.array(anchors,dtype='float32'),
                        future60min_cgm_min_mg_dl=np.array(mins,dtype='float32'))
    (out/'episodes.json').write_text(json.dumps(records,indent=2))
    manifest=dict(contract=c,contract_sha256=sha(R/'configs/actor_diagnostic.json'),shared_manifest_sha256=sha(replay/'manifest.json'),
                  selection_sha256=sha(out/'selection.npz'),episodes_sha256=sha(out/'episodes.json'),
                  source_sha256=sha(__file__),python=sys.version,numpy=np.__version__,states=len(rows),trajectories=len(records))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(states=len(rows),trajectories=len(records),future60_low70=int((np.array(mins)<70).sum()),
                         future60_low54=int((np.array(mins)<54).sum()),selection_sha256=manifest['selection_sha256'])),flush=True)

if __name__=='__main__':main()
