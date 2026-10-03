"""Inference-only diagnostic on fixed S2 states with persistent real anchors."""
import sys
from common import R,P,B,sha,load_world,np,torch,json
sys.path.insert(0,str(R.parent))
from ppo_worker import physical
from policy_bounded import Policy

def main():
    out=R/'checks/actor_diagnostic'; m=json.loads((out/'manifest.json').read_text())
    assert sha(out/'selection.npz')==m['selection_sha256']
    selected=np.load(out/'selection.npz');cache=np.load(R/'cache/s2.npz')
    assert sha(R/'cache/s2.npz')==json.loads((R/'cache/manifest.json').read_text())['cache_sha256']['s2.npz']
    positions=np.searchsorted(cache['indices'],selected['indices'])
    assert np.array_equal(cache['indices'][positions],selected['indices'])
    world,_=load_world(B/'results/D05_selected_world/world.pt')
    torch.set_num_threads(2)
    z=torch.tensor(cache['z'][positions],device='cuda');f=torch.tensor(cache['forecast'][positions],device='cuda')
    rate=torch.tensor(cache['rate'][positions],device='cuda');anchor=torch.tensor(selected['anchors'],device='cuda')
    assert (rate-anchor).abs().max()<=.26
    with torch.no_grad():ref=world.reference(z,f,rate)
    checkpoints={'D06':B/'results/D06_selected_policy/policy.pt'}
    history=[json.loads(line) for line in (R.parent/'results/PPO_real_rewards/history.jsonl').read_text().splitlines()]
    for iteration in [2,4,8]:
        p=R.parent/'results/PPO_real_rewards'/('policy_iter%02d.pt'%iteration)
        assert sha(p)==history[iteration-1]['checkpoint_sha256'];checkpoints['PPO%02d'%iteration]=p
    actions={};classes={};probabilities={};hashes={};ties={}
    for name,path in checkpoints.items():
        ck=torch.load(path,map_location='cpu');actor=Policy().cuda().eval();actor.load_state_dict(ck['policy'])
        assert ck['config']['world_sha256']==sha(B/'results/D05_selected_world/world.pt')
        with torch.no_grad():
            logits=actor(z,ref,rate,anchor);physical_logits=physical(logits);choice=physical_logits.argmax(-1)
            action=(anchor+torch.tensor([0.,-.25,.25],device='cuda')[choice]).clamp(0,20)
            original_map=torch.tensor([0,1,2,0,0,0,0],device='cuda')[logits.argmax(-1)]
        classes[name]=choice.cpu().numpy();actions[name]=action.cpu().numpy()
        probabilities[name]=physical_logits.softmax(-1).cpu().numpy();hashes[name]=sha(path)
        ties[name]=int((original_map!=choice).sum())
    summaries={};target=selected['future60min_cgm_min_mg_dl']
    for label,mask in [('all',np.ones(len(target),dtype=bool)),('future60_cgm_lt70',target<70),('future60_cgm_lt54',target<54)]:
        group=dict(states=int(mask.sum()),D06_class_counts=np.bincount(classes['D06'][mask],minlength=3).tolist(),models={})
        for name in ['PPO02','PPO04','PPO08']:
            delta=actions[name][mask]-actions['D06'][mask]; changed=delta!=0
            matrix=np.zeros((3,3),dtype=int)
            np.add.at(matrix,(classes['D06'][mask],classes[name][mask]),1)
            group['models'][name]=dict(changed=int(changed.sum()),changed_fraction=float(changed.mean()) if len(delta) else None,
                    dose_increased=int((delta>0).sum()),dose_decreased=int((delta<0).sum()),
                    mean_delta_u_h=float(delta.mean()) if len(delta) else None,
                    class_counts=np.bincount(classes[name][mask],minlength=3).tolist(),transition_matrix=matrix.tolist(),
                    mean_probability_total_variation=float((.5*np.abs(probabilities[name][mask]-probabilities['D06'][mask]).sum(-1)).mean()) if len(delta) else None)
        summaries[label]=group
    np.savez_compressed(out/'predictions.npz',indices=selected['indices'],anchor=selected['anchors'],rate=rate.cpu().numpy(),
                        future60min_cgm_min_mg_dl=target,**{k+'_actions':v for k,v in actions.items()},
                        **{k+'_physical_probabilities':v for k,v in probabilities.items()})
    result=dict(groups=summaries,class_order=['hold','reduce_0.25','increase_0.25'],checkpoint_sha256=hashes,
                selection_manifest_sha256=sha(out/'manifest.json'),source_sha256={str(p.relative_to(P)):sha(p) for p in [R/'actor_diagnostic.py',R.parent/'ppo_worker.py']},
                predictions_sha256=sha(out/'predictions.npz'),max_rate_anchor_difference=float((rate-anchor).abs().max()),
                physical_vs_original7_tie_disagreements=ties,association_only=True,confirmation_read=False,
                limitation='Actual future outcomes are from generating stochastic PPO trajectory, not from rerunning each candidate; repeated correlated training states.')
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
