import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from trading_env import TradingEvironment
from agent import Agent

def preprocess_data(data):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])
    return data_scaled, scaler

def train_trading_model(data, initial_balance= 10000, episodes= 1000, batch_size= 32):
    data_scaled, scaler = preprocess_data(data)
    train_data, test_data = train_test_split(data_scaled, test_size= .2, shuffle= False)
    
    env = TradingEvironment(train_data, initial_balance)
    state_size = env.data.shape[1]
    action_size = 3
    agent = Agent(state_size, action_size, gamma= 0.95, memory_size= 2000, epsilon_min= 0.01, epsilon_decay= 0.995)
    
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        
        for time in range(len(train_data)):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            
            agent.memory.add((state, action, reward, next_state, done), priority= 1.0)
            
            state = next_state
            total_reward += reward
            
            if done: 
                print(f"Episode: {e + 1}/ {episodes}, total reward: {total_reward}")
                break
        
        if(len(agent.memory.memory) > batch_size):
            agent.replay(batch_size)
    return agent, scaler
            