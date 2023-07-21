import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, args):
        
        self.max_size = args.max_buffer_size
        self.ptr_state = 0
        self.ptr_action = 0
        self.ptr_next_state = 0
        self.ptr_response = 0
        self.ptr_not_done = 0
        self.size = 0

        self.state = np.zeros((self.max_size, args.state_dim))
        self.action = np.zeros((self.max_size, args.action_dim))
        self.next_state = np.zeros((self.max_size, args.state_dim))
        self.response = np.zeros((self.max_size, args.response_dim))
        self.not_done = np.zeros((self.max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, state, action, next_state, response, done, state_dim, action_dim, response_dim):

        #end_ptr_state=self.ptr_state+state_dim
        #self.state[self.ptr_state:end_ptr_state] = state
        #print("state: ", state, self.state)
        self.state[self.size] = state

        #end_ptr_action=self.ptr_action+action_dim
        #self.action[self.ptr_action:end_ptr_action] = action
        #print("action: ", action, self.action)
        self.action[self.size] = action

        #end_ptr_next_state=self.ptr_next_state+state_dim
        #self.next_state[self.ptr_next_state:end_ptr_next_state] = next_state
        #print("next_state: ", next_state, self.next_state)
        self.next_state[self.size] = next_state

        #end_ptr_response=self.ptr_response+response_dim
        #self.response[self.ptr_response:end_ptr_response] = response
        #print("response: ", response, self.response)
        self.response[self.size] = response

        #end_ptr_not_done=self.ptr_not_done+1
        #self.not_done[self.ptr_not_done:end_ptr_not_done] = 1. - np.array(done)
        #print("not_done ", done, self.not_done)
        self.not_done[self.size] = 1. - np.array(done)

        #self.ptr_state = (self.ptr_state + state_dim) % (self.max_size * state_dim)
        #self.ptr_action = (self.ptr_action + action_dim) % (self.max_size * action_dim)
        #self.ptr_next_state = (self.ptr_next_state + state_dim) % (self.max_size * state_dim)
        #self.ptr_response = (self.ptr_response + response_dim) % (self.max_size * response_dim)
        #self.ptr_not_done = (self.ptr_not_done + 1) % (self.max_size*1)

        self.size = min(self.size + 1, self.max_size)
        # if self.size==self.max_size:
        #     print("Replay buffer is full!")
        print("self.size: ",self.size,"\n")
        assert self.size<=self.max_size

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.response[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def normalize_states(self, eps = 1e-3):
        mean = self.state.mean(0,keepdims=True)
        std = self.state.std(0,keepdims=True) + eps
        self.state = (self.state - mean)/std
        self.next_state = (self.next_state - mean)/std
        return mean, std
    
    def stat_actions(self, eps= 1e-3):
        mean = self.action.mean(0,keepdims=True)
        std= self.action.std(0,keepdims=True) + eps
        max_a=self.action.max(axis=0)
        min_a=self.action.min(axis=0)
        return mean.tolist(), std.tolist(), max_a.tolist(), min_a.tolist()

    def normal_actions(self, eps= 1e-3, mean=None, std=None):
        if mean is None:
            mean = self.action.mean(0,keepdims=True)
        if std is None:
            std= self.action.std(0,keepdims=True) + eps
        self.action=(self.action-mean)/std

    def save(self, path):
        np.savez(path, state=self.state[:self.size], action=self.action[:self.size], next_state=self.next_state[:self.size], response=self.response[:self.size], not_done=self.not_done[:self.size])

    def load(self, path, allow_pickle=False):
        stored_array=np.load(path, allow_pickle=allow_pickle)
        self.size=len(stored_array["state"])
        print("shape: ", stored_array["state"].shape)
        print("load_size: ", self.size)
        self.state = stored_array["state"]
        self.action = stored_array["action"]
        self.next_state = stored_array["next_state"]
        self.response = stored_array["response"]
        self.not_done = stored_array["not_done"]

        print("res: ", self.response)
        
    def load_kuai_data(self, path):
        with open(path+'states.npy', 'rb') as f:
            self.state = np.load(f, allow_pickle=True)
            np.nan_to_num(self.state, copy=False)
        with open(path+'actions.npy', 'rb') as f:
            self.action = np.load(f, allow_pickle=True)
        with open(path+'next_states.npy', 'rb') as f:
            self.next_state = np.load(f, allow_pickle=True)
            np.nan_to_num(self.next_state, copy=False)
        with open(path+'rewards.npy', 'rb') as f:
            self.response = np.load(f, allow_pickle=True)
        with open(path+'dones.npy', 'rb') as f:
            self.not_done = np.logical_not(np.load(f, allow_pickle=True))

        self.size = len(self.state)
