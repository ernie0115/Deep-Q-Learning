import numpy as np 
import random # import torch # import torch.nn as nn # import torch.optim as optim # import QNetwork # import DQNAgent 

# Define your 2D environment class. 
class Environment_Easy: 
    def __init__(self, num_red_apples, num_green_poisonous_things): 
        
        # Define the number of eyes and the number of values each eye senses. 
        self.num_eyes = 9 
        self.num_values_per_eye = 3 
        self.max_visibility_distance = 10.0 # Adjust as needed. 
        
        # Define the boundaries of the environment. 
        self.x_min = 0 
        self.x_max = 100 # Adjust as needed. 
        self.y_min = 0 
        self.y_max = 100 # Adjust as needed. 
        
        # Initialize the environment state with maximum visibility distances. 
        # self.initial_state is a 3 * 9 array with 10.0 in each element. 
        self.initial_state = np.full((self.num_values_per_eye, self.num_eyes), self.max_visibility_distance) 

        # Define the number of red apples and green poisonous things. 
        self.num_red_apples = num_red_apples 
        self.num_green_poisonous_things = num_green_poisonous_things 
        
        # Place the red apples and green poisonous things randomly.
        self.red_apple_positions = self._generate_random_positions(self.num_red_apples, [])
        self.green_poisonous_thing_positions = self._generate_random_positions(self.num_green_poisonous_things, self.red_apple_positions)

        # Initialize the agent's position. 
        self.agent_position = [50, 50]  # Starting position of the agent 
        
    def _generate_random_positions(self, num_positions, exclude_positions): 
        positions = [] 
        for i in range(num_positions): 
            position = [random.randint(self.x_min, self.x_max), random.randint(self.y_min, self.y_max)] 
            while position in positions or position in exclude_positions: 
                position = [random.randint(self.x_min, self.x_max), random.randint(self.y_min, self.y_max)] 
            positions.append(position) 
        return positions 
    
    def _get_agent_positions(self): 
        # Return the current position of the agent. 
        return [self.agent_position] 
    
    def _update_state(self): 
        # Initialize a state array with maximum visibility distances. 
        state = np.full((self.num_values_per_eye, self.num_eyes), self.max_visibility_distance) 

        # For each red apple, calculate the distance and update the state. 
        for i, apple_position in enumerate(self.red_apple_positions): 
            if apple_position is not None: 
                distance = np.linalg.norm(np.array(self.agent_position) - np.array(apple_position)) 
                if i < self.num_eyes: 
                    state[0][i] = distance  # Update the distance to the red apple. 

        # For each green poisonous thing, calculate the distance and update the state. 
        for i, poison_position in enumerate(self.green_poisonous_thing_positions): 
            if poison_position is not None: 
                distance = np.linalg.norm(np.array(self.agent_position) - np.array(poison_position)) 
                if i < self.num_eyes: 
                    state[1][i] = distance  # Update the distance to the green poisonous thing. 

        return state 
        
    def step(self, action): 
        
        # Define the mapping between action indices and action strings. 
        action_mapping = { 
            0: "move_left", 
            1: "move_right", 
            2: "move_up", 
            3: "move_down" 
        } 
        
        # Implement the step function to update the environment based on the chosen action. 

        # # Update the state. 
        # next_state = self._update_state() 

        # 1. Calculate the reward. 
        reward = 0 
        
        # Check if the agent collects a red apple. 
        for i in range(self.num_red_apples): 
            if self.red_apple_positions[i] in self._get_agent_positions(): 
                reward = reward + 1.0 # Add a positive reward for collecting a red apple. 
                self.red_apple_positions[i] = None # Remove the collected red apple from the environment. 
                break # Exit the loop once a red apple is collected. 
                
        # Check if the agent collides with a green poisonous thing. 
        for i in range(self.num_green_poisonous_things): 
            if self.green_poisonous_thing_positions[i] in self._get_agent_positions(): 
                reward = reward - 1.0 # Add a negative reward for colliding with a green poisonous thing. 
                self.green_poisonous_thing_positions[i] = None # Remove the collected green things from the environment. 
                done = True # Terminate the episode. 
                break # Exit the loop once a green poisonous thing is collided with. 

        # 2. Determine if the episode is done. 
        done = False 

        # Update the agent's position based on the chosen action. 
        if action == 0 and self.agent_position[0] > self.x_min: 
            self.agent_position[0] -= 1 # Move left. 
            
        elif action == 1 and self.agent_position[0] < self.x_max: 
            self.agent_position[0] += 1 # Move right. 
            
        elif action == 2 and self.agent_position[1] < self.y_max: 
            self.agent_position[1] += 1 # Move up. 
            
        elif action == 3 and self.agent_position[1] > self.y_min: 
            self.agent_position[1] -= 1 # Move down. 
        
        # Check if the action takes the agent outside the boundaries. 
        if action == 0 and self.agent_position[0] <= self.x_min: 
            done = True # Terminate the episode. 
            
        elif action == 1 and self.agent_position[0] >= self.x_max: 
            done = True # Terminate the episode. 
            
        elif action == 2 and self.agent_position[1] >= self.y_max: 
            done = True # Terminate the episode. 
            
        elif action == 3 and self.agent_position[1] <= self.y_min: 
            done = True # Terminate the episode. 
            
        # Get the corresponding action string from the mapping. 
        action_string = action_mapping.get(action, "unknown") 

        # Update the state based on the new position of the agent. 
        next_state = self._update_state() 
        
        # Return the next_state, reward, and whether the episode is done. 
        return next_state, reward, done, action_string, self.agent_position 
        # return next_state, reward, done, action_string, self.agent_position, self.red_apple_positions, self.green_poisonous_thing_positions 