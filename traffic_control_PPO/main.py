import os
import sys
import argparse
import time
from logger import logger
from traffic_env import traffic_env
from model import PPO
import shutil
import numpy as np
import traci

## 主要是导入一些常用的shell命令
## for linux
os.environ['SUMO_HOME'] = "/myfile/sumo-0.32.0/"
sys.path.append("/myfile/sumo-0.32.0/tools")
## for windows
# os.environ['SUMO_HOME'] = 'E:/sumo-0.32.0/'
# sys.path.append('E:/sumo-0.32.0/')
# sys.path.append(os.path.join('E:/sumo-0.32.0/', 'tools'))



parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='traffic')
parser.add_argument('--model_name', type=str, default='model')
parser.add_argument('--render', action='store_true')
parser.add_argument('--mpc_horizon', type=int, default=15)
parser.add_argument('--num_random_action_selection', type=int, default=4096)
parser.add_argument('--nn_layers', type=int, default=1)
args = parser.parse_args()

data_dir = 'DATA'
exp_dir = os.path.join(data_dir, args.exp_name)
mod_dir = os.path.join(data_dir, args.model_name)
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir, exist_ok=True)
else:
    shutil.rmtree(exp_dir)
    os.makedirs(exp_dir, exist_ok=True)
logger.setup(exp_dir, os.path.join(exp_dir, 'log.txt'), 'debug')
def traffic_train():
    num_episodes = 2000
    max_epLength = 1000
    total_step = 0
    env = traffic_env()
    ppo = PPO(n_actions=env.n_act, n_features=env.n_sta, saving_loading=True)
    ppo.load_model(path=mod_dir)
    total_reward = []
    start_step = 30
    for ep in range(num_episodes):
        env.generateTrips(max_epLength*2)
        env.reset(env.sumoCmd)
        reward = []
        for si in range(start_step):
            traci.simulationStep()
        s = env.start()
        for t in range(max_epLength - start_step):
            a = ppo.choose_action(s)
            r, s_ = env.step(a)
            ppo.store_transition(s,a,r,s_)
            if total_step > ppo.memory_size and t % 10 == 0:
                ppo.learn()
            reward.append(r)
            s = s_
            total_step += 1
        env.end()
        logger.debug(' train ' + str(ep) + ' totalreward : ' + str(sum(reward)) + ' ReturnAvg : ' + str(np.mean(reward)) + ' ReturnStd : ' + str(np.std(reward)))
        total_reward.append(sum(reward))
        if ppo.saving_or_loading == True and total_step > 15000 and sum(reward) == max(total_reward):
            ppo.saver.save(sess=ppo.sess, save_path=ppo.save_path)
    for t in range(20):
        env.reset(env.sumoCmd_gui)
        for i in range(max_epLength):
            traci.simulationStep()
            time.sleep(0.1)
        env.end()
def run_main():
    traffic_train()


if __name__ == "__main__":
    run_main()







