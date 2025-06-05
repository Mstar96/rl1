import numpy as np

class Thread:
    def __init__(self, tid, length, alpha):
        self.id = tid
        self.length = length
        self.alpha = alpha
        self.resource = 0
        self.exce_time = 0.0

class Server:
    def __init__(self, sid, capacity):
        self.id = sid
        self.capacity = capacity
        self.remaining = capacity
        self.threads = []

    def assign_thread(self, thread: Thread, resource):
        resource = min(resource, self.remaining)
        thread.resource = resource
        speed = thread.alpha * np.log(1 + resource)
        thread.exce_time = thread.length / speed
        self.remaining -= resource
        self.threads.append(thread)

    def current_max_time(self):
        return max([t.exce_time for t in self.threads], default=0)

