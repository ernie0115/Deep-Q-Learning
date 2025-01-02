# import numpy as np # import random # import torch # import torch.nn as nn # import torch.optim as optim # import QNetwork 
from DQNAgent import DQNAgent # import Environment_Easy 

# Define your 2D environment and the training loop 
def train_dqn(num_episodes, env): 
    state_size = 27 # Adjust according to your state representation (9 eyes * 3 info). 
    action_size = 4 # Number of possible actions (4 kinds of movements). 
    total_reward = 0 

    agent = DQNAgent(state_size, action_size) 

    for episode in range(num_episodes): 
        state = env.initial_state # Initialize the environment state (3 * 9 array, elements: 10.0). 
        episode_total_reward = 0 
        done = False 
        
        print(f"Episode: {episode + 1}") 

        while not done: 
            action = agent.select_action(state) 
            next_state, reward, done, action_string, agent_position = env.step(action)  # Use the environment you defined. 
            # next_state, reward, done, action_string, agent_position, red_apple_positions, green_poisonous_thing_positions = env.step(action)  # Use the environment you defined. 

            agent.train(state, action, reward, next_state, done) 
            state = next_state 
            episode_total_reward += reward 
            print(f"Action: {action_string}, Done: {done}, Reward: {reward}, Agent Position: {agent_position}") 
            
        print(f"Episode {episode + 1} Total Reward: {episode_total_reward}\n") 
        total_reward += episode_total_reward 

    print(f"Total Reward: {total_reward}") 