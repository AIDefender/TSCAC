import csv  
import yaml
from types import SimpleNamespace as SN
from buffer import *

def get_path():
	import os
	return os.getcwd()

with open(get_path() + "/config/buffer.yaml", 'r') as f:
    args=yaml.safe_load(f)
    args=SN(**args)   
    buffer=ReplayBuffer(args)
    #buffer.save(os.path.join(save_path,f"{i}-fold"))
    with open(get_path() + '/data/tripadvisor_data/tripadvisor_test_data.csv','r') as myFile:  
        lines=csv.reader(myFile)  
        row = 0
        test_buffer = []
        for line in lines:  
            #print(line)
            row = row + 1
            if row>=2:
                #'reward_service', 'reward_business', 'reward_cleanliness', 'reward_check_in', 'reward_value', 'reward_rooms', 'reward_location', 'reward_overall', 'action',
                reward_service = line[3]
                reward_business = line[4]
                reward_cleanliness = line[5]
                reward_check_in = line[6]
                reward_value = line[7]
                reward_rooms = line[8]
                reward_location = line[9]
                reward_overall = line[10]
                reward = line[3:11]
                action = line[11]
                next_state = line[12:39]
                state = line[39:66]
                done = 0
                sample = []
                sample.append(state)
                sample.append(action)
                sample.append(next_state)
                sample.append(reward)
                sample.append(done)
                #print("sample:",sample, "\n")
                #print("state:",state, "\n")
                #print("action:",action, "\n")
                #print("next_state:",next_state, "\n")
                #print("reward:",reward, "\n")
                #print("done:",done, "\n")
                test_buffer.append(sample)

                #add to replay buffer
                #print("state_dim:", args.state_dim)
                #print("action_dim:", args.action_dim)
                #print("response_dim:", args.response_dim)
                buffer.add(state,action,next_state,reward,done,args.state_dim,args.action_dim,args.response_dim)
            #if row>=100:
            #    break

        #print("test_buffer:", test_buffer, "\n")
        buffer.save(get_path() + "/data/tripadvisor_data/test_buffer_fulldata")

load_buffer = ReplayBuffer(args)
load_buffer.load(get_path() + "/data/tripadvisor_data/test_buffer_fulldata.npz")
#print("buffer_state:",load_buffer.state, "\n")
#print("buffer_action:",load_buffer.action, "\n")
#print("buffer_next_state:",load_buffer.next_state, "\n")
#print("buffer_reward:",load_buffer.response, "\n")
#print("buffer_done:",load_buffer.not_done, "\n")
