"""No-training reward-consistency ablation of the unchanged H02 transition."""
from ditr_model import PatientModel,status_score

class PointRewardPatient(PatientModel):
 def step(self,z,action,memory,features=None):
  next_z,_,features=super().step(z,action,memory,features)
  return next_z,status_score(self.glucose(next_z).squeeze(-1))/12,features
 def rollout(self,history,actions,memory=None):
  result=super().rollout(history,actions,memory=memory)
  result['reward']=status_score(result['glucose'])/12
  return result
