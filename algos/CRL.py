import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class L_Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))

class L_Encoder(nn.Module):
	def __init__(self, state_dim, hidden_dim):
		super(L_Encoder, self).__init__()
		self.encoder = nn.Sequential(nn.Linear(state_dim, 256),
									nn.ReLU(inplace=True),
									nn.Linear(256, hidden_dim)
									)

	def forward(self, l_state):
		return self.encoder(l_state)

class H_Actor(nn.Module):
	def __init__(self, state_dim, hidden_dim):
		super(H_Actor, self).__init__()
		self.encoder = nn.Sequential(nn.Linear(state_dim, 256),
									nn.ReLU(inplace=True),
									nn.Linear(256, 256),
									nn.ReLU(inplace=True))  
        self.fc_mu  = nn.Linear(256, hidden_dim) 
        self.fc_std = nn.Linear(256, hidden_dim)

	def encode(self, x):
        x = self.encoder(x)
        return self.fc_mu(x), F.softplus(self.fc_std(x), beta=1)
    
    def reparameterise(self, mu, std):
        """
        mu : [batch_size,z_dim]
        std : [batch_size,z_dim]        
        """        
        # get epsilon from standard normal
        eps = torch.randn_like(std)
        return mu + std*eps
    
    def forward(self, h_state):
		# x_shape: (batch_size, state_dim)
        mu, std = self.encode(h_state)
        z = self.reparameterise(mu, std)
        return z, mu, std 

class CRL(object):
    def __init__(self, args):
        self.context_mask=args.context_mask

        self.h_actor=H_Actor(args.h_state_dim, args.hidden_dim).to(device)
        self.l_actor=L_Actor(args.hidden_dim*2,action_dim, max_action).to(device)
		self.l_encoder=[L_Encoder(args.l_state_dim, args.hidden_dim).to(device) for _ in range(args.n_l_response)] # 7 types of low-level responses
        
        self.target
        
        self.optimizer

		self.critic

	def select_action(self, state):
		h_state,l_state=state
		z, mu, std=self.h_actor(h_state) # (batch, hidden_dim)
		l_features=[l_encoder(l_state) for l_encoder in self.l_encoder] 
	
	
	def train(self, replay_buffer, batch_size=64):
