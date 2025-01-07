import numpy as np 
import random 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from QNetwork import QNetwork 
from ReplayBuffer import ReplayBuffer 

# Define the Deep Q-Learning agent. 
class DQNAgent: 
    def __init__(self, state_size, action_size, gamma = 0.99, epsilon = 1.0, epsilon_decay = 0.995, epsilon_min = 0.01, target_update_frequency = 10, replay_buffer_size = 1000): 
        self.state_size = state_size # state_size: 27 (9 eyes * 3 info) 
        self.action_size = action_size # action_size: 4 kinds of movements 
        self.gamma = gamma # Discount factor for future rewards 
        self.epsilon = epsilon # Exploration rate 
        self.epsilon_decay = epsilon_decay # Decay rate for exploration 
        self.epsilon_min = epsilon_min # Minimum exploration rate 

        # Update Target Network More Frequently. 
        self.target_update_frequency = target_update_frequency  # New parameter for target update frequency. 

        # Initialize replay buffer with a specified size. 
        self.replay_buffer = ReplayBuffer(replay_buffer_size)  # Initialize replay buffer. 
        
        # Create the online network and target network. # state_size, action_size: 27 (9 eyes * 3 info), 4 kinds of movements. 
        # Online Network: for selecting actions 
        # Target Network: for stable Q-value targets 
        self.online_network = QNetwork(state_size, action_size) 
        self.target_network = QNetwork(state_size, action_size) 
        self.target_network.load_state_dict(self.online_network.state_dict()) 
        self.target_network.eval()  # Freeze the target network's parameters. 

        # The Adam optimizer is used to update the weights of the online_network. 
        self.optimizer = optim.Adam(self.online_network.parameters(), lr = 0.001) 

        # Track the number of training steps. 
        self.training_step = 0 
    
    # The select_action method of the DQNAgent class returns the index of the action with the highest Q-value. 
    def select_action(self, state): 
        if np.random.rand() < self.epsilon: 
            action = random.randrange(self.action_size) # Explore: Select a random action. 
        else: 
            with torch.no_grad(): 
                # Flatten the state to a 1D tensor of size 27. 
                state = torch.tensor(state.flatten(), dtype = torch.float32) 
                q_values = self.online_network(state) # Get Q-values for the current state. 
                action = int(np.argmax(q_values.numpy())) # Exploit: Select the action with the highest Q-value. 
        return action 
    
    # Method to train the agent 
    def train(self, state, action, reward, next_state, done): 
        # Add the experience to the replay buffer. 
        self.replay_buffer.add((state, action, reward, next_state, done)) 

        # Sample a batch from the replay buffer. 
        if self.replay_buffer.size() > 32:  # Ensure enough samples for training. 
            batch = self.replay_buffer.sample(32) # Sample a batch of experiences. 
            for state, action, reward, next_state, done in batch: 

                # Convert state to tensor and flatten it to 1D. 
                state = torch.tensor(state.flatten(), dtype = torch.float32) 
                # Convert next state to tensor and flatten it to 1D. 
                next_state = torch.tensor(next_state.flatten(), dtype = torch.float32) 
                
                # 1. Clear the gradient. 
                self.optimizer.zero_grad() 

                # 2. Input the data into the model. 
                q_values = self.online_network(state)  # Get Q-values for the current state. 
                next_q_values = self.target_network(next_state) # Get Q-values for the next state. 

                target = q_values.clone() # Clone the Q-values for updating. 

                # Update the target for the action taken. 
                target[action] = reward + self.gamma * torch.max(next_q_values) * (not done) 

                # Calculate loss and update the model. 
                loss = nn.MSELoss()(q_values, target) # 3. Calculate loss. 
                loss.backward() # 4. Calculate the gradient. [Backpropogation] 
                self.optimizer.step() # 5. Do gradient Descent. [Update the parameter.] 

        # Update the target network at specified intervals. 
        self.training_step += 1 
        if self.training_step % self.target_update_frequency == 0: 
            self.target_network.load_state_dict(self.online_network.state_dict()) 
            self.target_network.eval()  # Freeze the target network's parameters. 

        if self.epsilon > self.epsilon_min: 
            self.epsilon *= self.epsilon_decay # Decay exploration rate. 