import os
import sys
sys.path.append("..")

from stable_baselines3 import PPO
from envs.thread_allocation_env import *
from envs.basic_class import *
import numpy as np

def make_server_env():
    server = Server(0,capacity=20)
    threads = []
    for i in range(3):
        length = np.random.randint(80,120)
        alpha = np.random.uniform(0.1,0.9)
        thread = Thread(i,length,alpha)
        threads.append(thread)
        server.threads.append(thread)
    return ResourceAllocationEnv(server,threads)
env = make_server_env()
model = PPO("MlpPolicy",env,verbose=1)
model.learn(total_timesteps=100_000)
model.save("D:\\codes\\RL_paper\\models\\thread_alloc")