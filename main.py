import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from trading_env import TradingEvironment
from agent import Agent

def preprocess_data(data):
    data['Volume'] = data['Volume'].str.replace(',', '').astype(float)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])
    return data_scaled, scaler

def train_trading_model(data, initial_balance=10000, episodes=1000, batch_size=32):
    data_scaled, scaler = preprocess_data(data)
    train_data, test_data = train_test_split(data_scaled, test_size=0.2, shuffle=False)
    
    env = TradingEvironment(train_data, initial_balance)
    state_size = env.data.shape[1]
    action_size = 3
    agent = Agent(state_size, action_size, gamma=0.95, memory_size=2000, epsilon_min=0.01, epsilon_decay=0.995)
    
    rewards = []

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        
        for time in range(len(train_data)):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            
            agent.memory.add((state, action, reward, next_state, done), priority=1.0)
            
            state = next_state
            total_reward += reward
            
            if done: 
                print(f"Episode: {e + 1}/{episodes}, Total Reward: {total_reward}")
                rewards.append(total_reward)
                break
        
        if len(agent.memory.memory) > batch_size:
            agent.replay(batch_size)
    
    return agent, scaler, rewards

def plot_training_progress(rewards):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label='Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig("training_progress.png")
    plt.show()

if __name__ == "__main__":
    data_path = "gmedata.csv"
    
    try:
        data = pd.read_csv(data_path)
        print("Data loaded successfully!")
    except FileNotFoundError:
        print(f"File not found: {data_path}")
        exit()

    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in data.columns for col in required_columns):
        print(f"Missing required columns in the data. Expected: {required_columns}")
        exit()

    print("Starting training...")
    initial_balance = 10000
    episodes = 1000
    batch_size = 32
    checkpoint_interval = 10 

    agent, scaler, rewards = None, None, []
    try:
        for e in range(1, episodes + 1):
            if agent is None:
                data_scaled, scaler = preprocess_data(data)
                train_data, test_data = train_test_split(data_scaled, test_size=0.2, shuffle=False)
                env = TradingEvironment(train_data, initial_balance)
                state_size = env.data.shape[1]
                action_size = 3
                agent = Agent(
                    state_size=state_size,
                    action_size=action_size,
                    gamma=0.95,
                    memory_size=2000,
                    epsilon_min=0.01,
                    epsilon_decay=0.995,
                )

            state = env.reset()
            state = np.reshape(state, [1, state_size])
            total_reward = 0

            for time in range(len(train_data)):
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                next_state = np.reshape(next_state, [1, state_size])
                agent.memory.add((state, action, reward, next_state, done), priority=1.0)
                state = next_state
                total_reward += reward
                if done:
                    print(f"Episode: {e}/{episodes}, Total Reward: {total_reward}")
                    rewards.append(total_reward)
                    break

            if len(agent.memory.memory) > batch_size:
                agent.replay(batch_size)

            if e % checkpoint_interval == 0:
                agent.model.save_checkpoint()
                print(f"Checkpoint saved at episode {e}")

    except KeyboardInterrupt:
        print("Training interrupted. Saving progress...")

    os.makedirs("models", exist_ok=True)
    agent.model.save_checkpoint()
    print(f"Final model saved to models/trained_dqn_model")

    plot_training_progress(rewards)
