import json
from torch.distributions.normal import Normal
from data_process import *
def get_path():
	import os
	return os.getcwd()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            #print("action: ",a)
            #print("policy_action: ",policy_action)
            #print("behavior_action: ",behavior_policy_action)
            log_prob=dist.log_prob(a).sum(axis=-1)
            log_prob_b=dist_b.log_prob(a).sum(axis=-1)
            #ratio=torch.exp(log_prob-log_prob_b).clamp(0.1,10)
            ratio=torch.exp(log_prob-log_prob_b).clamp(0,10)
            ratio = ratio/total_ratio

            R = np.array(reward[t])
            ratio = ratio.item()
            R = ratio * R
            if t%1000 ==0:
                print("t:", t)
                print("ratio:", ratio)
                print("estimated_rewards: ", R)
                print("\n")

            estimated_rewards = estimated_rewards + R

    print("final estimated_rewards:", estimated_rewards)
    print("total ratio:", total_ratio)
    print( "without total_ratio", torch.tensor(estimated_rewards).to(device)/total_ratio)

    return estimated_rewards

def normalize_action(a):
    a=np.array(a)
    mu=0
    var=1
    a=(a-mu)/var
    return a

if __name__ == "__main__":
    from types import SimpleNamespace as SN
    from algos import *
    import yaml
    import copy

#2w no.41
#16w n0.43
#30w(20w) no.44
#16w(overeall 100) no.46
#2w(overeall 100) no.47 
#24w(overeall 100) no.52

#25.2w(location 100) no.53
#25.2w(location 100， lr=1e-6) no.54
#15w(overall 100， lr=0) no.55
#8w(overall 100， lr=1e-4) no.56, root=weight^2
#4.5w(overall 100， lr=1e-4) no.57
#20.4w(overall 100， lr=1e-10) no.58
#24w(overall 100， lr=1e-10) no.59
#24w(overall 100， lr=1e-10， bc_loss_coeff=1e-4) no.60

    with open(get_path() + "/config/pareto_critic.yaml","r") as f:
        config=yaml.safe_load(f)
    args=SN(**config)
    
    policy=MULTI_CRITIC.MULTICRI_DDPG(args)
    policy.load(get_path() + "/results/pareto_critic/60/models/") #load trained DDPG policy
 
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