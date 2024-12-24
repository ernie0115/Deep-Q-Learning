import numpy as np 
import random 
import torch 
import torch.nn as nn 
import torch.optim as optim # import QNetwork # import DQNAgent 
from Environment_Easy import Environment_Easy 
from train_dqn import train_dqn 

# Create an instance of your 2D environment. 
# 351: # of red apples 
# 289: # of green poisonous things 
env = Environment_Easy(351, 289) 

# Main training loop 
if __name__ == "__main__": 
    num_episodes = 1 # Adjust the number of episodes as needed. 
    train_dqn(num_episodes, env) 