import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from algos.LAYER import FeaturesLinear,FeaturesEmbedding,MultiLayerPerceptron,FactorizationMachine
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_filed_max_dict = {}
feature_filed_min_dict = {}

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        feature_size = 1044
        embed_dim = 4
        mlp_dims = [32,32]
        dropout = 0.2
        # self.feature_filed_max_list = [149.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 149.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 149.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        # self.feature_filed_min_list = [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0]
        self.feature_filed_max_list = [5] * 1044
        self.feature_filed_min_list = [-2] * 1044

        field_dims = [int(self.feature_filed_max_list[i] - self.feature_filed_min_list[i] + 2) for i in range(feature_size)]
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.max_action = max_action

    def _feature_hash(self, state):
        state_np = state.numpy()
        for i in range(len(state_np)):
            for j in range(len(state_np[0])):
                if state_np[i][j] > self.feature_filed_max_list[j]:
                    state_np[i][j] = self.feature_filed_max_list[j] - self.feature_filed_min_list[j] + 1
                else:
                    state_np[i][j] = state_np[i][j] - self.feature_filed_min_list[j]
        return torch.from_numpy(state_np)

    def forward(self, state):
        state = self._feature_hash(state)
        embed_x = self.embedding(state.long())
        x = self.linear(state.long()) + self.mlp(embed_x.view(-1, self.embed_output_dim)) 
        return self.max_action *  torch.sigmoid(x.squeeze(1))

class WIDE_DEEP(object):
    def __init__(self, args):
        state_dim, action_dim, max_action, discount, tau=args.state_dim, args.action_dim, args.max_action, args.discount, args.tau
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
    
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        #return self.actor(state).cpu().data.numpy().flatten()
        return self.actor(state)

    def train(self, replay_buffer, args, batch_size=64):
                # Sample replay buffer 
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        reward=self._process_reward(reward, args)
        
        # Compute actor loss
        # policy loss+ bc loss
        actor_loss = torch.mean(reward.detach() * torch.pow(self.actor(state) - action, 2))
        
        # Optimize the actor 
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss, actor_loss, self.actor(state)

    def _process_reward(self, reward, args):
        # reward= args.reward_service * reward[:,0] + args.reward_business * reward[:,1] + args.reward_cleanliness * reward[:,2] + args.reward_check_in * reward[:,3] + args.reward_value * reward[:,4] + args.reward_rooms * reward[:,5] + args.reward_location * reward[:,6]
        reward= args.reward_click * reward[:,0] + args.reward_like * reward[:,1] + args.reward_follow * reward[:,2] + args.reward_comment * reward[:,3] + args.reward_forward * reward[:,4] + args.reward_hate * reward[:,5] + args.reward_play_time * reward[:,6]
        #  + args.reward_overall * reward[:,7]
        #print("reward:", reward)
        reward = torch.reshape(reward, [-1,1])
        # if reward_max - reward_min < 1e-4:
        #     reward = reward * 0.0
        # else:
        #     reward = (reward - reward_min) / (reward_max - reward_min + 1e-4)
        return reward

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
         
    def load(self, filename): 
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
                
 
