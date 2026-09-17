"""Training-only library of jointly logged future events, conditional on observable history summaries."""
import json,hashlib
import numpy as np
import torch
from dynamics_data import Dynamics,ROOT
from model_guidance import context_features

def main():
 torch.set_num_threads(2);d=Dynamics('train',device='cpu');generator=torch.Generator().manual_seed(61791);eligible=torch.nonzero(d.extra['length']>=12).flatten();selected=eligible[torch.randperm(len(eligible),generator=generator)[:8192]];contexts=[];events=[]
 for offset in range(0,len(selected),256):
  b=d.batch(selected[offset:offset+256]);contexts.append(context_features(b['state']).numpy());events.append(b['events'].numpy())
 p=ROOT/'event_bank_v1.npz';np.savez(p,context=np.concatenate(contexts),events=np.concatenate(events),origin_index=selected.numpy());meta={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'seed':61791,'source_split':'train','library_size':len(selected),'source_complete60_origins':len(eligible),'training_actor_origins_unchanged':d.size,'context':['normalized current CGM','30min CGM change','last-hour mean normalized basal','last-hour bolus record count /3','last-hour carb record count /3','last-hour exercise record count /3'],'events':'joint next-60min normalized bolus/carb/exercise values and observed masks; logged absence remains unknown','limitations':'Empirical conditional event scenarios from training observations, not a calibrated generative or causal model; no actual future evaluation events read','clinical_ready':False};p.with_suffix('.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
