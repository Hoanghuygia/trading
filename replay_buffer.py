import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = []
        self.priorites = []
        self.capacity = capacity
               
    def add(self, experience, priority):
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
            self.priorites.pop(0)
        self.memory.append(experience)
        self.priorites.append(priority)
