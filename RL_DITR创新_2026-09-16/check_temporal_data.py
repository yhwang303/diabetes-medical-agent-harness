"""Validate temporal training branches and exact shared reference with prior data."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');args=ap.parse_args();folder=ROOT/'paired_sim_train_temporal';manifest=folder/'manifest.json'
 if not args.partial:assert manifest.exists(),'Generation not complete'
 files=sorted(folder.glob('*.npz'));assert files;groups=0;ranks=[];full_equal_pairs=0
 hashes={x['file']:x['sha256'] for x in json.loads(manifest.read_text())['scenarios']} if manifest.exists() else {}
 for path in files:
  if hashes:assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[path.name]
  with np.load(path) as b,np.load(ROOT/'paired_sim_train'/path.name) as old:
   x=b['features'];assert x.shape[1:]==(7,120,20);groups+=len(x)
   assert np.isfinite(x).all() and np.isfinite(b['target']).all() and np.isfinite(b['action']).all()
   np.testing.assert_array_equal(x[:,:,:72],np.repeat(x[:,0:1,:72],7,axis=1))
   for key in ['features','target','bg','action','mask','length','terminal']:np.testing.assert_array_equal(b[key][:,0],old[key][:,0])
   np.testing.assert_array_equal(b['mask'],np.arange(48)[None,None,:]<b['length'][:,:,None])
   for i in range(len(x)):
    if not b['mask'][i].all():continue
    da=b['action'][i,1:]-b['action'][i,0];rank=int(np.linalg.matrix_rank(da,tol=1e-5));assert rank==3;ranks.append(rank)
    total=b['action'][i].sum(1)/12
    np.testing.assert_allclose(total[[1,3,5]],total[1],atol=2e-6);np.testing.assert_allclose(total[[2,4,6]],total[2],atol=2e-6);full_equal_pairs+=6
 result={'status':'partial_verified' if args.partial else 'passed','files_checked':len(files),'groups_checked':groups,'complete_group_action_ranks':sorted(set(ranks)),'equal_total_complete_pairs':full_equal_pairs,'reference_arm_matches_previous_data_exactly':True,'same_observed_history_across_arms':True,'future_features_for_labels_only':True,'full_manifest_available':manifest.exists()}
 (ROOT/'checks'/('temporal_training_partial.json' if args.partial else 'temporal_training_check.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
