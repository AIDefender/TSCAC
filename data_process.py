import os, random
from buffer import *
import json
from tqdm import tqdm


def get_path():
	import os
	return os.getcwd()
 
action_dim=8

def parse_json(trajactory): 
    state=[]
    h_state=[]
    action=[]
    response=[]
    h_response=[]

    for session in trajactory:
        for i, request in enumerate(session["request"]):
            state.append(request["state"])
            h_state.append(request["h_state"])
            assert len(request["action"])>=8 and len(request["action"])<=12
            action.append(request["action"][:action_dim])
            response.append(request["response"])
            h_response.append(request["h_response"])

    next_state=state[1:]
    next_state.append(state[-1])

    next_h_state=h_state[1:]
    next_h_state.append(h_state[-1])

    done=np.zeros_like(list(range(len(state))))
    done[-1]=1  
    done=np.expand_dims(done, axis=1)
    return state,h_state,action,next_state,next_h_state, response,h_response,done

if __name__ == "__main__":
    import yaml
    from types import SimpleNamespace as SN


