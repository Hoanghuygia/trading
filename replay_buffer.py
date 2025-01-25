import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = []
        self.priorites = []
        self.capacity = capacity
        
    def add(self, experience, priority):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
            self.priorites.append(priority)
        else:
            idx = np.argmin(self.priorites)
            self.memory[idx] = experience
            self.priorites[idx] = priority