import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import gym
import numpy as np
import networkx as nx
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from envs.thread_allocation_env import *
# 创建环境实例
env = GraphRLServerEnv(num_threads=10, num_servers=3, server_capacity=60, optimize='makespan')

# 检查环境是否符合 Gym 接口
check_env(env)

# 创建 DQN 模型
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.001,
    buffer_size=10000,
    learning_starts=500,
    batch_size=32,
    target_update_interval=500,
    exploration_fraction=0.8,
    exploration_final_eps=0.05,
    tensorboard_log="./dqn_graph_server_logs/"
)

# 训练模型
model.learn(total_timesteps=100000, log_interval=4)

# 保存模型
model.save(model,"dqn_graph_server")

# 测试模型
obs = env.reset()
for _ in range(10):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(f"Action: {action}, Reward: {rewards}, Done: {dones}")