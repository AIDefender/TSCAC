import os
import json
from torch.distributions.normal import Normal
from data_process import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_config_file(params):
    config_file = "ddpg"
    for i, v in enumerate(params):
        print(v)
        if v.split("=")[0] == "--config":
            config_file = v.split("=")[1]
            del params[i]
            break
    return config_file


def offline_ab(policy, behavior_policy, replay_buffer):
    # calculate sequentially, could be calculate in batch as well.
    total_ratio=0
    estimated_rewards = np.zeros(8)
    with torch.no_grad():
        state = replay_buffer.state
        action = replay_buffer.action
        reward = replay_buffer.response
        size = replay_buffer.size
        for t, s in enumerate(state):
            s=np.array(s)
            policy_action=policy.select_action(s)
            behavior_policy_action=behavior_policy.select_action(s)
            dist=Normal(policy_action,25)
            dist_b=Normal(behavior_policy_action, 25)

            a=torch.tensor(action[t]).to(device)
            #print("action: ",a)
            #print("policy_action: ",policy_action)
            #print("behavior_action: ",behavior_policy_action)
            log_prob=dist.log_prob(a).sum(axis=-1)
            log_prob_b=dist_b.log_prob(a).sum(axis=-1)
            ratio=torch.exp(log_prob-log_prob_b).clamp(0,10)
            total_ratio = total_ratio + ratio
            #print("ratio:", ratio)
            #print("total_ratio:", total_ratio)
            #print("\n")
        
        total_ratio = total_ratio
        print("total_ratio:", total_ratio)

        for t, s in enumerate(state):
            s=np.array(s)
            policy_action=policy.select_action(s)
            behavior_policy_action=behavior_policy.select_action(s)
            dist=Normal(policy_action,25)
            dist_b=Normal(behavior_policy_action, 25)

            a=torch.tensor(action[t]).to(device)
            print("action: ",a)
            print("policy_action: ",policy_action)
            print("behavior_action: ",behavior_policy_action)
            log_prob=dist.log_prob(a).sum(axis=-1)
            log_prob_b=dist_b.log_prob(a).sum(axis=-1)
            #ratio=torch.exp(log_prob-log_prob_b).clamp(0.1,10)
            ratio=torch.exp(log_prob-log_prob_b).clamp(0,10)
            ratio = ratio/total_ratio

            R = np.array(reward[t])
            ratio = ratio.item()
            print("estimated_rewards: ", R)
            R = ratio * R
            print("ratio:", ratio)
            print("estimated_rewards: ", R)
            print("\n")

            estimated_rewards = estimated_rewards + R

        print("estimated_rewards:", estimated_rewards)

    return estimated_rewards

def normalize_action(a):
    a=np.array(a)
    mu=0
    var=1
    a=(a-mu)/var
    return a

def get_path():
	import os
	return os.getcwd()

if __name__ == "__main__":
    from types import SimpleNamespace as SN
    from algos import *
    from algos import SL
    import yaml
    import copy

    model_name = "deep_fm"
    model_version = "1"
    with open(get_path() + "/config/" + model_name + ".yaml","r") as f:
        config=yaml.safe_load(f)
    args=SN(**config)

    policy=DEEP_FM.DEEP_FM(args)
    model_list = os.listdir(get_path() + "/results/" + model_name + "/")
    model_list.remove('_sources')
    model_list = sorted(model_list)
    policy.load(get_path() + "/results/" + model_name + "/" + model_version + "/models/") #load trained DDPG policy
    
    replay_buffer=ReplayBuffer(args)
    replay_buffer.load(get_path() + "/data/tripadvisor_data/test_buffer_fulldata.npz") #load test buffer

    behavior_policy=SL.SL(args)
    #train behavior_policy
    #for e in range(int(args.max_steps)):
    #    loss=behavior_policy.train(replay_buffer, args.batch_size)
    #    print("step: ", e)
    #    print(loss)
    #behavior_policy.save(get_path() + "/behavior_model/sl")

    #load behavior_policy
    behavior_policy.load(get_path() + "/behavior_model/sl")
    
    offline_ab(policy, behavior_policy,replay_buffer)