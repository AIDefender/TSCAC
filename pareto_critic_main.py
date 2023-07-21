import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch
import logging
import random
import yaml
from types import SimpleNamespace as SN
import pprint
from algos import *
from utils import *
from buffer import *
from data_process import *

def get_path():
	import os
	return os.getcwd()


root_path = get_path()

def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        '[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('DEBUG')
    return logger


# set to "no" if you want to see stdout/stderr in console
SETTINGS['CAPTURE_MODE'] = "fd"
logger = get_logger()

ex = Experiment()
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(root_path, "results")


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    random.seed(_config["seed"])
    np.random.seed(_config["seed"])
    torch.manual_seed(_config["seed"])
    # run the framework
    run(_run, _config, _log)


def parse_config_file(params):
    config_file = "pareto_critic"
    for i, v in enumerate(params):
        print(v)
        if v.split("=")[0] == "--config":
            config_file = v.split("=")[1]
            del params[i]
            break
    return config_file

def Create_Policy(args):
    if args.algo== "pareto_critic":
        return PARETO_CRITIC.PARETO_DDPG(args)
    if args.algo== "multi_critic":
        return MULTI_CRITIC.MULTICRI_DDPG(args)
    if args.algo=="ddpg":
        return DDPG.DDPG(args)
    if args.algo=="td3":
        return TD3.TD3(args)
    if args.algo=="td3_bc":
        return TD3_BC.TD3_BC(args)


def run(_run, _config, _log):
    args = SN(**_config)
    args.ex_results_path = os.path.join(args.ex_results_path, str(_run._id))
    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                    indent=4,
                                    width=1)
    _log.info("\n\n" + experiment_params + "\n")

    if args.use_tensorboard:
        logger.setup_tb(args.ex_results_path)

    # sacred is on by default
    logger.setup_sacred(_run)

    start_time = time.time()
    last_time = start_time
    logger.console_logger.info("Beginning training for {} steps".format(args.max_steps))
    
    last_train_e=0
    last_test_e=-args.test_every-1
    last_save_e=0
    last_log_e=0

    replay_buffer=ReplayBuffer(args)
    #sample_user(replay_buffer, number=1,folder_path=get_path() + "/data/training")
    #loading from the training buffer

    replay_buffer.load(get_path() + "/data/tripadvisor_data/train_buffer_fulldata.npz")
    policy=Create_Policy(args)

    for e in range(int(args.max_steps)):
        loss=policy.train(replay_buffer, args, args.batch_size)
        logger.log_stat("actor_loss", loss[0].item(), e)
        logger.log_stat("critic_loss", loss[1].item(), e)
        #logger.log_stat("predict_action", loss[2].item(), e)
        #print(loss)
        if (e-last_test_e) / args.test_every >= 1.0: #testing
            pass
        if args.save_model and (e-last_save_e) / args.save_every >= 1.0: #saving
            save_path = os.path.join(args.ex_results_path, "models/")
            os.makedirs(save_path, exist_ok=True)
            policy.save(save_path)
            last_save_e = e
        if e % args.print_every == 0:
            print("iter:", e)
            print("actor_loss", loss[0])
            print("critic_loss", loss[1])
            print("weight", loss[3])
        if (e-last_log_e) / args.log_every >= 1.0:
            pass

if __name__ == '__main__':
    params = deepcopy(sys.argv)
    config_file = parse_config_file(params)

    ex.add_config(get_path() + '/config/{}.yaml'.format(config_file))

    logger.info(
        f"Saving to FileStorageObserver in {root_path}/results/{config_file}.")
    file_obs_path = os.path.join(results_path, config_file)
    ex.add_config(name=config_file)
    ex.add_config(ex_results_path=file_obs_path)
    ex.observers.append(FileStorageObserver.create(file_obs_path))
    ex.run_commandline(params)
 