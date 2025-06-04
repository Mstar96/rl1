import gym
import numpy as np
from gym import spaces
import networkx as nx

class Thread:
    def __init__(self,id,length,alpha):
        self.id = id
        self.length = length
        self.alpha = alpha
        self.resource = 0
        self.exce_time = 0.0

class Server:
    def __init__(self,id,capacity):
        self.id = id
        self.capacity = capacity
        self.remaining = capacity
        self.threads = []
    
    def assign_thread(self,thread:Thread,resource):
        resource = min(resource,self.remaining)
        thread.resource = resource
        speed = thread.alpha * np.log(1+resource)
        thread.exce_time = thread.length / speed
        self.remaining -= resource
        self.threads.append(thread)
    
    def current_max_time(self):
        if not self.threads:
            return 0.0
        return max(t.exce_time for t in self.threads)
    
    def current_total_time(self):
        return sum(t.exce_time for t in self.threads)

class GraphRLServerEnv(gym.Env):
    "构建图结构"
    def __init__(self,num_threads=10,num_servers=3,server_capacity=60,optimize='makespan'):
        super().__init__()
        self.num_threads = num_threads
        self.num_servers = num_servers
        self.server_capacity = server_capacity
        self.optimize = optimize 

        self.max_length = 200
        self.max_alpha = 2.0
        self.min_alloc = 0.1
        
        #动作：线程t_i -> 服务器 S_j
        self.action_space = spaces.Discrete(self.num_servers)
        self.reset()
        
    def reset(self, *, seed = None, options = None):
        self.threads = []
        for i in range(self.num_threads):
            length = np.random.randint(100,self.max_length)
            alpha = np.random.uniform(0.5,self.max_alpha)
            self.threads.append(Thread(i,length,alpha))
        self.servers = [Server(i,self.server_capacity)for i in range(self.num_servers)]
        self.current_idx = 0
        return self._get_graph()

    def _get_graph(self):
        G = nx.Graph()
        node_features = {}
        
        for t in self.threads:
            G.add_node(f"t{t.id}",type='thread')
            node_features[f"t{t.id}"] = [
                t.length / self.max_length,
                t.alpha / self.max_alpha,
                0.0 if t.exce_time == 0 else 1.0,
                t.exce_time / 100 if t.exce_time > 0 else 0.0
            ]
        
        for s in self.servers:
            G.add_node(f"s{s.id}",type='server')
            node_features[f"s{s.id}"] = [
                s.remaining / s.capacity,
                len(s.threads) / self.num_threads,
                s.current_max_time() / 100.0,
                s.current_total_time() / 200.0
            ]
        
        for t in self.threads:
            for s in self.servers:
                G.add_edge(f"t{t.id}",f"s{s.id}")
        return G,node_features

    def step(self, server_id):
        assert self.action_space.contains(server_id)
        thread = self.threads[self.current_idx]
        server = self.servers[server_id]
        #平均分配一下
        if len(server.threads) == 0:
            alloc = server.remaining
        else:
            alloc = server.remaining / (len(server.threads) + 1)
        alloc = max(self.min_alloc,alloc)
        
        server.assign_thread(thread,alloc)
        self.current_idx += 1
        done = self.current_idx >= self.num_threads
        
        if not done:
            reward = 0
            obs = self._get_graph()
        else:
            if self.optimize == 'makespan':
                reward = -max(s.current_max_time() for s in self.servers)
            else:
                reward = -sum(s.current_total_time() for s in self.servers)
            obs = (None,None)
        return obs,reward,done,{}

#test
if __name__ == '__main__':
    env = GraphRLServerEnv()
    G,feats = env.reset()
    print("初始化图结构，节点数：",len(G.nodes))
    print("线程节点示例：",{k:v for k,v in feats.items() if k.startswith('t')}["t0"])
