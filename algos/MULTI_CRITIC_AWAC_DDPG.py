import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from algos.MULTI_CRITIC import MULTICRI_DDPG
from torch.distributions.normal import Normal


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiCri_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(MultiCri_Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 32)
        self.l2 = nn.Linear(32, 32)
        self.l3 = nn.Linear(32, action_dim)
        
        self.max_action = max_action

    
    def forward(self, state):
        a = F.relu(self.l1(state))
        #a = F.relu(self.l2(a))
        return self.max_action * torch.sigmoid(self.l3(a))


class MultiCri_Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(MultiCri_Critic, self).__init__()

        self.l1_0 = nn.Linear(state_dim, 32)
        self.l2_0 = nn.Linear(32 + action_dim, 32)
        self.l3_0 = nn.Linear(32, 1)

        self.l1_1 = nn.Linear(state_dim, 32)
        self.l2_1 = nn.Linear(32 + action_dim, 32)
        self.l3_1 = nn.Linear(32, 1)

        self.l1_2 = nn.Linear(state_dim, 32)
        self.l2_2 = nn.Linear(32 + action_dim, 32)
        self.l3_2 = nn.Linear(32, 1)

        self.l1_3 = nn.Linear(state_dim, 32)
        self.l2_3 = nn.Linear(32 + action_dim, 32)
        self.l3_3 = nn.Linear(32, 1)

        self.l1_4 = nn.Linear(state_dim, 32)
        self.l2_4 = nn.Linear(32 + action_dim, 32)
        self.l3_4 = nn.Linear(32, 1)

        self.l1_5 = nn.Linear(state_dim, 32)
        self.l2_5 = nn.Linear(32 + action_dim, 32)
        self.l3_5 = nn.Linear(32, 1)

        self.l1_6 = nn.Linear(state_dim, 32)
        self.l2_6 = nn.Linear(32 + action_dim, 32)
        self.l3_6 = nn.Linear(32, 1)

        self.l1_7 = nn.Linear(state_dim, 32)
        self.l2_7 = nn.Linear(32 + action_dim, 32)
        self.l3_7 = nn.Linear(32, 1)

    def forward(self, state, action):
        action = action.reshape(-1, 1)
        q0 = F.relu(self.l1_0(state))
        q0 = F.relu(self.l2_0(torch.cat([q0, action], 1)))

        q1 = F.relu(self.l1_1(state))
        q1 = F.relu(self.l2_1(torch.cat([q1, action], 1)))

        q2 = F.relu(self.l1_2(state))
        q2 = F.relu(self.l2_2(torch.cat([q2, action], 1)))

        q3 = F.relu(self.l1_3(state))
        q3 = F.relu(self.l2_3(torch.cat([q3, action], 1)))

        q4 = F.relu(self.l1_4(state))
        q4 = F.relu(self.l2_4(torch.cat([q4, action], 1)))

        q5 = F.relu(self.l1_5(state))
        q5 = F.relu(self.l2_5(torch.cat([q5, action], 1)))

        q6 = F.relu(self.l1_6(state))
        q6 = F.relu(self.l2_6(torch.cat([q6, action], 1)))
        
        q7 = F.relu(self.l1_7(state))
        q7 = F.relu(self.l2_7(torch.cat([q7, action], 1)))

        return [self.l3_0(q0), self.l3_1(q1), self.l3_2(q2), self.l3_3(q3), self.l3_4(q4), self.l3_5(q5), self.l3_6(q6), self.l3_7(q7)]   


def get_path():
    import os
    return os.getcwd()

class MULTI_CRITIC_AWAC_DDPG(object):
    def __init__(self, args):
        state_dim, action_dim, max_action, discount, tau=args.state_dim, args.action_dim, args.max_action, args.discount, args.tau
        self.actor = MultiCri_Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = MultiCri_Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-2)

        self.discount = discount
        self.tau = tau
        self.reward_dim = args.reward_dim
        self.weight = [args.reward_click, args.reward_like,args.reward_follow, args.reward_comment, args.reward_forward,args.reward_hate, args.reward_play_time]

        self.constrained_policy = MULTICRI_DDPG(args)
        self.constrained_policy.load(get_path() + args.constrained_policy_model_path)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        #return self.actor(state).cpu().data.numpy().flatten()
        return self.actor(state)


    def train(self, replay_buffer, args, batch_size=64):
        # Sample replay buffer 
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        
        # no need to process reward beforehand, 
        # process each dimension one by one
        #reward=self.process_reward(reward, args)
        
        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        
        for i in range(self.reward_dim):
            #print("reward:",  reward.shape, "reward i :",  reward[:, i].shape )
            target_Q[i] = reward[:, i].reshape([-1, 1]) + (not_done * self.discount * target_Q[i]).detach()
            

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = 0 
        for i in range(self.reward_dim):
            critic_loss += F.mse_loss(current_Q[i], target_Q[i])
            
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        # policy loss+ bc loss
        qValue = self.critic(state, self.actor(state))
        tmp_loss = 0
        for i in range(self.reward_dim):
            tmp_loss += -qValue[i] *self.weight[i]
            #print("tmp_loss shape: ", tmp_loss.shape,"qValue shape: ", qValue[i].shape )
        constrained_policy_action = self.constrained_policy.actor(state)
        constrained_dist = Normal(constrained_policy_action, args.sigma)
        current_action = self.actor(state)
        current_dist = Normal(current_action, args.sigma)
        if args.new_ratio:
            ratio = torch.exp(constrained_dist.log_prob(current_action)).clamp(0,1).detach()
        else:
            ratio = torch.exp(constrained_dist.log_prob(action) - current_dist.log_prob(action)).clamp(0,1).detach()

        bc_loss = F.mse_loss(self.actor(state), action)
        ref_loss = F.mse_loss(self.actor(state), constrained_policy_action.detach())

        actor_loss = args.awac_loss_coef * torch.mean(ratio * tmp_loss) + args.bc_loss_coeff * bc_loss + args.kl_loss_coeff * ref_loss


        # Optimize the actor 
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        return actor_loss, critic_loss, self.actor(state)
        
    def process_reward(self, reward, args):
        reward= args.reward_click * reward[:,0] + args.reward_like * reward[:,1] + args.reward_follow * reward[:,2] + args.reward_comment * reward[:,3] + args.reward_forward * reward[:,4] + args.reward_hate * reward[:,5] + args.reward_play_time * reward[:,6]
        #print("reward:", reward)
        return reward

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic", map_location=torch.device('cpu')))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer", map_location=torch.device('cpu')))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor", map_location=torch.device('cpu')))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", map_location=torch.device('cpu')))
        self.actor_target = copy.deepcopy(self.actor)
        
