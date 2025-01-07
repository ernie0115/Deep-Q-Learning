# import numpy as np # import random # import torch # import torch.nn as nn # import torch.optim as optim # import QNetwork 
# from DQNAgent import DQNAgent # import Environment_Easy 
# from DQNAgent_TF import DQNAgent 
from DQNAgent_TF_ER import DQNAgent 
import numpy as np 
import matplotlib.pyplot as plt 
import torch 

# Define your 2D environment and the training loop 
def train_dqn(num_episodes, env): 
    state_size = 27 # Adjust according to your state representation (9 eyes * 3 info). 
    action_size = 4 # Number of possible actions (4 kinds of movements). 
    total_reward = 0 

    agent = DQNAgent(state_size, action_size) 

    rewards_per_episode = []  # List to store average rewards per episode. 
    max_q_values_per_episode = []  # List to store average maximum Q-values. 

    for episode in range(num_episodes): 
        state = env.initial_state # Initialize the environment state (3 * 9 array, elements: 10.0). 
        episode_total_reward = 0 
        done = False 

        episode_max_q_values = []  # List to store max Q-values during the episode. 
        
        print(f"Episode: {episode + 1}") 

        while not done: 
            action = agent.select_action(state) 
            next_state, reward, done, action_string, agent_position = env.step(action)  # Use the environment you defined. 
            # next_state, reward, done, action_string, agent_position, red_apple_positions, green_poisonous_thing_positions = env.step(action)  # Use the environment you defined. 

            agent.train(state, action, reward, next_state, done) 
            state = next_state 
            episode_total_reward += reward 
            print(f"Action: {action_string}, Done: {done}, Reward: {reward}, Agent Position: {agent_position}") 

            # Calculate average maximum predicted action-value. 
            state = torch.tensor(state.flatten(), dtype = torch.float32) 

            with torch.no_grad(): 
                q_values = agent.online_network(state) 
                max_q_value = torch.max(q_values).item() 
                episode_max_q_values.append(max_q_value) # Store max Q-value for this step. 

        rewards_per_episode.append(episode_total_reward) 

        # Calculate average maximum predicted action-value for the episode. 
        average_max_q_value = np.mean(episode_max_q_values) 
        max_q_values_per_episode.append(average_max_q_value) 

        print(f"Episode {episode + 1} Total Reward: {episode_total_reward}\n") 
        total_reward += episode_total_reward 

    print(f"Total Reward: {total_reward}") 

    # Quantification (Total Award vs Episode) 
    # Plot the performance metrics. 
    plt.figure(figsize = (12, 5)) 

    # Plot average rewards per episode. 
    plt.subplot(1, 2, 1) 
    plt.plot(rewards_per_episode) 
    plt.title("Average Reward per Episode") 
    plt.xlabel("Episode") 
    plt.ylabel("Total Reward") 

    # Plot average maximum predicted action-value. 
    plt.subplot(1, 2, 2) 
    plt.plot(max_q_values_per_episode) 
    plt.title("Average Maximum Predicted Action-Value") 
    plt.xlabel("Episode") 
    plt.ylabel("Max Q-Value") 

    plt.tight_layout() 
    plt.show() 