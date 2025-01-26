import numpy as np
from replay_buffer import ReplayBuffer
from dqn import DeepQNetwork

class Agent:
    def __init__(self, state_size, action_size, gamma, memory_size, epsilon_min, epsilon_decay, epsilon= 1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = DeepQNetwork(state_size, action_size)       
    
    def boltzman_exploration(self, q_values, temperature=1.0):
        scaled_q_values = q_values / max(temperature, 1e-6)  
        exp_values = np.exp(scaled_q_values - np.max(scaled_q_values)) 
        probabilities = exp_values / np.sum(exp_values)
        return np.random.choice(len(q_values), p=probabilities)

    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return self.boltman_exploration(np.zeros(self.action_size))
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        if len(self.memory.memory) < batch_size:
            return
        minibatch = np.random.choice(len(self.memory.memory), batch_size, replace= False)
        for i in minibatch:
            state, action, reward, next_state, done = self.memory.memory[i]
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
                
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay