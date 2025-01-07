import random 
from collections import deque 

class ReplayBuffer: 
    def __init__(self, max_size): 
        # Initialize a deque (double-ended queue) with a maximum size. 
        self.buffer = deque(maxlen = max_size) 

    def add(self, experience): 
        # Add a new experience to the buffer. 
        self.buffer.append(experience) 

    def sample(self, batch_size): 
        # Randomly sample a batch of experiences from the buffer. 
        return random.sample(self.buffer, batch_size) 

    def size(self): 
        # Return the current size of the buffer. 
        return len(self.buffer) 