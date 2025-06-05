import gymnasium as gym
from gymnasium import spaces
import numpy as np
from basic_class import *

class ThreadToServerEnv(gym.Env):
    """高层环境：线程分配到服务器（动作：离散）"""
    metadata = {"render_modes": []}

    def __init__(self, num_threads=10, num_servers=3, server_capacity=60):
        super().__init__()
        self.num_threads = num_threads
        self.num_servers = num_servers
        self.server_capacity = server_capacity

        self.max_length = 200
        self.max_alpha = 2.0

        self.action_space = spaces.Discrete(self.num_servers)
        self.observation_space = spaces.Box(low=0, high=1, shape=(4 + self.num_servers * 2,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        self.threads = [
            Thread(i, np.random.randint(100, self.max_length), np.random.uniform(0.5, self.max_alpha))
            for i in range(self.num_threads)
        ]
        self.servers = [Server(i, self.server_capacity) for i in range(self.num_servers)]
        self.current_idx = 0
        return self._get_obs(), {}

    def _get_obs(self):
        t = self.threads[self.current_idx]
        thread_feats = np.array([
            t.length / self.max_length,
            t.alpha / self.max_alpha,
            self.current_idx / self.num_threads,
            len(self.threads) / 100
        ])
        server_feats = []
        for s in self.servers:
            server_feats.append(s.remaining / s.capacity)
            server_feats.append(len(s.threads) / self.num_threads)
        return np.concatenate([thread_feats, np.array(server_feats, dtype=np.float32)], dtype=np.float32)

    def step(self, action):
        assert self.action_space.contains(action)
        thread = self.threads[self.current_idx]
        server = self.servers[action]

        # 暂时不给资源，由底层分配
        server.threads.append(thread)
        self.current_idx += 1

        done = self.current_idx >= self.num_threads
        reward = 0
        if done:
            # 完成后调用底层策略进行资源分配（或模拟），得到 makespan
            reward = -max(s.current_max_time() for s in self.servers)
        return self._get_obs() if not done else np.zeros_like(self._get_obs()), reward, done, False, {}
