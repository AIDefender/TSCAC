import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l0 = nn.Linear(state_dim, 512)
        self.l1 = nn.Linear(512, 256)
        self.l2 = nn.Linear(256, 64)
        self.l3 = nn.Linear(64, action_dim)
        
        self.max_action = max_action

    
    def forward(self, state):
        a = F.relu(self.l0(state))
        # print(a)
        a = F.relu(self.l1(a))
        a = F.relu(self.l2(a))
        return self.max_action * torch.sigmoid(self.l3(a))

class Actor_onehot(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor_onehot, self).__init__()
        self.l0 = nn.Linear(state_dim, 512)
        self.l1 = nn.Linear(512, 256) 
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, max_action)
        
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l0(state))
        a = F.relu(self.l1(a))
        a = F.relu(self.l2(a))
        return self.l3(a)

class SL_onehot(object):
    def __init__(self, args):
        state_dim, action_dim, max_action=args.state_dim, args.action_dim, args.max_action
        self.actor = Actor_onehot(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer =torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.max_action=max_action

    def select_action(self, state):
        if len(state.shape)== 1:
            state = state.reshape(1, -1)
        state = torch.FloatTensor(state).to(device)
        out = self.actor(state)
        return torch.argmax(out, axis=1).double()


    def train(self, replay_buffer, batch_size=64):
        # Sample replay buffer 
        state, action, _,_,_= replay_buffer.sample(batch_size)

        predict_action=self.actor(state)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss=loss_fn(predict_action,action.long())

        # Optimize the actor 
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return loss

    def evaluate(self, replay_buffer, batch_size=32768):
        state, action, _,_,_= replay_buffer.sample(batch_size)
        predict_action=self.select_action(state)
        loss=F.mse_loss(predict_action,action)
        return loss.item(), torch.mean(predict_action), torch.std(predict_action)

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor", map_location="cpu"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", map_location="cpu"))

class SL(object):
    def __init__(self, args):
        state_dim, action_dim, max_action=args.state_dim, args.action_dim, args.max_action
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer =torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.max_action=max_action

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        #return self.actor(state).cpu().data.numpy().flatten()
        return self.actor(state)


    def train(self, replay_buffer, batch_size=64):
        # Sample replay buffer 
        state, action, _,_,_= replay_buffer.sample(batch_size)

        predict_action=self.actor(state)
        action=action.clamp(-self.max_action, self.max_action)
        loss=F.mse_loss(predict_action,action)

        # Optimize the actor 
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return loss

    def evaluate(self, replay_buffer, batch_size=32768):
        state, action, _,_,_= replay_buffer.sample(batch_size)
        predict_action=self.actor(state)
        action=action.clamp(-self.max_action, self.max_action)
        #print("predict_action: ", predict_action)
        #print("action: ", action)
        loss=F.mse_loss(predict_action,action)

        return loss.item()

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor", map_location="cpu"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", map_location="cpu"))
    
        
