import gymnasium as gym
import numpy as np
from gymnasium import spaces
from basic_class import *

class ResourceAllocationEnv(gym.Env):
    """底层环境：资源分配（动作：连续向量）"""
    def __init__(self, server: Server, threads: list):
        super().__init__()
        self.server = server
        self.threads = threads
        self.n = len(threads)
        self.total_capacity = server.capacity

        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n * 2,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.n,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        obs = []
        for t in self.threads:
            obs.append(t.length / 200)
            obs.append(t.alpha / 2.0)
        return np.array(obs, dtype=np.float32), {}
    def step(self, action):
        # 确保 action 无零值
        action = np.clip(action, 1e-3, 1.0)  # 限制最小值
        normed_allocs = action / (np.sum(action) + 1e-8)
        allocs = normed_allocs * self.total_capacity

        for i, t in enumerate(self.threads):
            t.resource = max(allocs[i], 1e-3)  # 确保 resource >= 1e-3
            speed = t.alpha * np.log(1 + t.resource)
            t.exce_time = t.length / speed

        # 处理可能的数值异常
        reward = -max(np.clip([t.exce_time for t in self.threads], -1e6, 1e6))
        done = True
        return np.array([], dtype=np.float32), reward, done, False, {}
    