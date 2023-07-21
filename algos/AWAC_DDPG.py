import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from algos.DDPG import DDPG
from algos.MULTI_CRITIC import MULTICRI_DDPG
from torch.distributions.normal import Normal


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def get_path():
	import os
	return os.getcwd()

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 32)
		self.l2 = nn.Linear(32, 32)
		self.l3 = nn.Linear(32, action_dim)
		
		self.max_action = max_action

	
	def forward(self, state):
		a = F.relu(self.l1(state))
		#a = F.relu(self.l2(a))
		return self.max_action * torch.sigmoid(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim, 32)
		self.l2 = nn.Linear(32 + action_dim, 32)
		self.l3 = nn.Linear(32, 1)


	def forward(self, state, action):
		q = F.relu(self.l1(state))
		q = F.relu(self.l2(torch.cat([q, action], 1)))
		return self.l3(q)


class AWAC_DDPG(object):
	def __init__(self, args):
		state_dim, action_dim, max_action, discount, tau=args.state_dim, args.action_dim, args.max_action, args.discount, args.tau
		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-2)

		self.discount = discount
		self.tau = tau

		self.constrained_policy = DDPG(args)
		self.constrained_policy.load(get_path() + args.constrained_policy_model_path)


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		#return self.actor(state).cpu().data.numpy().flatten()
		return self.actor(state)


	def train(self, replay_buffer, args, batch_size=64):
		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
		
		reward=self.process_reward(reward, args)
		
		# Compute the target Q value
		target_Q = self.critic_target(next_state, self.actor_target(next_state))
		target_Q = reward + (not_done * self.discount * target_Q).detach()

		# Get current Q estimate
		current_Q = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Compute actor loss
		# policy loss+ bc loss
		constrained_policy_action = self.constrained_policy.actor(state)
		constrained_dist = Normal(constrained_policy_action, args.sigma)
		current_action = self.actor(state)
		current_dist = Normal(current_action, args.sigma)

		ratio = torch.exp(constrained_dist.log_prob(action) - current_dist.log_prob(action)).clamp(0,10).detach()

		actor_loss = -torch.mean(ratio * self.critic(state, self.actor(state))) + args.bc_loss_coeff * F.mse_loss(self.actor(state), action)
		
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
		reward= args.reward_service * reward[:,0] + args.reward_business * reward[:,1] + args.reward_cleanliness * reward[:,2] + args.reward_check_in * reward[:,3] + args.reward_value * reward[:,4] + args.reward_rooms * reward[:,5] + args.reward_location * reward[:,6] + args.reward_overall * reward[:,7]
		#print("reward:", reward)
		reward = torch.reshape(reward, [-1,1])
		return reward

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
		