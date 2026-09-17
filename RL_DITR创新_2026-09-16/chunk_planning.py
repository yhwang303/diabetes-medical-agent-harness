"""Action-block beam retaining every five-minute risk prediction, not endpoints only."""
import torch

@torch.no_grad()
def plan_chunks(agent,history,horizon=48,block_steps=16,beam_size=10,gamma=.9**(1/12)):
 if horizon<1 or block_steps<1 or horizon>440:raise ValueError('Invalid planning horizon')
 memory=agent.patient.encode(history);batch=len(history);z=memory[:,-1];paths=z.new_zeros(batch,1,0);beam=1
 for start in range(0,horizon,block_steps):
  length=min(block_steps,horizon-start);candidates=agent.policy.candidates(z,beam_size);branches=candidates.shape[1];a=candidates.reshape(batch,beam*branches)
  expanded=torch.cat([paths.repeat_interleave(branches,1),a[:,:,None].expand(-1,-1,length)],-1);steps=expanded.shape[-1];flat_actions=expanded.reshape(-1,steps)
  histories=history[:,None].expand(-1,beam*branches,-1,-1).reshape(-1,*history.shape[1:]);mem=memory[:,None].expand(-1,beam*branches,-1,-1).reshape(-1,*memory.shape[1:])
  prediction=agent.patient.rollout(histories,flat_actions,memory=mem)
  discount=torch.pow(z.new_tensor(gamma),torch.arange(steps,device=z.device));reward=(prediction['reward']*discount).sum(1);last=prediction['states'][:,-1]
  score=(reward+gamma**steps*agent.patient.value(last).squeeze(-1)).reshape(batch,-1)
  count=min(beam_size,score.shape[1]);take=score.topk(count,dim=1).indices;flat=(torch.arange(batch,device=z.device)[:,None]*score.shape[1]+take).reshape(-1)
  z=last[flat];paths=expanded.gather(1,take[:,:,None].expand(-1,-1,steps));beam=count
 return paths[:,0,0],{'planned_actions_u_h':paths[:,0].cpu().tolist(),'plan_value':score.gather(1,take)[:,0].cpu().tolist(),'horizon':horizon,'block_steps':block_steps,'beam_size':beam_size,'score_includes_all_5min_rewards':True,'execution':'first5min only, then reobserve and replan'}
