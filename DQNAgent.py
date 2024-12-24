import numpy as np 
import random 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from QNetwork import QNetwork 

# Define the Deep Q-Learning agent. 
class DQNAgent: 
    def __init__(self, state_size, action_size, gamma = 0.99, epsilon = 1.0, epsilon_decay = 0.995, epsilon_min = 0.01): 
        self.state_size = state_size # state_size: 9 eyes 
        self.action_size = action_size # action_size: 4 kinds of movements 
        self.gamma = gamma # Discount factor for future rewards 
        self.epsilon = epsilon # Exploration rate 
        self.epsilon_decay = epsilon_decay # Decay rate for exploration 
        self.epsilon_min = epsilon_min # Minimum exploration rate 
        
        # Create the online network and target network. # state_size, action_size: 9 eyes, 4 kinds of movements. 
        self.online_network = QNetwork(state_size, action_size) 
        self.target_network = QNetwork(state_size, action_size) 
        self.target_network.load_state_dict(self.online_network.state_dict()) 
        self.target_network.eval()  # Freeze the target network's parameters. 

        self.optimizer = optim.Adam(self.online_network.parameters(), lr = 0.001) 
    
    # The select_action method of the DQNAgent class returns the index of the action with the highest Q-value. 
    def select_action(self, state): 
        if np.random.rand() < self.epsilon: 
            action = random.randrange(self.action_size) # Explore: Select a random action. 
        else: 
            with torch.no_grad(): 
                state = torch.tensor(state, dtype = torch.float32) 
                q_values = self.online_network(state) 
                action = int(np.argmax(q_values.numpy())) # Exploit: Select the action with the highest Q-value. 

        # Ensure action is within bounds. 
        while action < 0 or action >= self.action_size: 
            # Re-select the action if it is out of bounds. 
            action = random.randrange(self.action_size) # Explore: Select a random action. 
            
        # Ensure action is within bounds. 
        return max(0, min(action, self.action_size - 1)) 
    
    # Method to train the agent 
    def train(self, state, action, reward, next_state, done): 
        state = torch.tensor(state, dtype = torch.float32) # Convert state to tensor. 
        next_state = torch.tensor(next_state, dtype = torch.float32) # Convert next state to tensor. 
        
        self.optimizer.zero_grad() # 1. Clear the gradient. 

        q_values = self.online_network(state)  # Get Q-values for the current state. # 2. Input the data into the model. 
        next_q_values = self.target_network(next_state) # Get Q-values for the next state. 

        target = q_values.clone() # Clone the Q-values for updating. 

        # target[0][action] = reward + self.gamma * torch.max(next_q_values) 
        # Update the target for the action taken. 
        target[0][action] = reward + self.gamma * torch.max(next_q_values) * (not done) 

        # Calculate loss and update the model. 
        loss = nn.MSELoss()(q_values, target) # 3. Calculate loss. 
        loss.backward() # 4. Calculate the gradient. [Backpropogation] 
        self.optimizer.step() # 5. Do gradient Descent. [Update the parameter.] 

        if self.epsilon > self.epsilon_min: 
            self.epsilon *= self.epsilon_decay 
