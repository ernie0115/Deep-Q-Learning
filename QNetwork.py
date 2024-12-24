import torch 
import torch.nn as nn 

# Define your neural network architecture. 
# In this case, input_size = 9 (9 eyes), output_size = 4 (4 kinds of movements). 
class QNetwork(nn.Module): 
    def __init__(self, input_size, output_size): 
        super().__init__() 
        self.fc1 = nn.Linear(input_size, 128) 
        self.fc2 = nn.Linear(128, 64) 
        self.fc3 = nn.Linear(64, output_size) 

    # This line defines the forward method for the QNetwork class, 
    # which specifies how data flows through the network during the forward pass. 
    def forward(self, x): 
        # ReLU: Rectified Linear Unit 
        # It applies ReLU activation function to the output of the 1st fully connected layer (fc1). 
        x = torch.relu(self.fc1(x)) # torch.nn.functional.relu 
        # It applies ReLU activation function to the output of the 2nd fully connected layer (fc2). 
        x = torch.relu(self.fc2(x)) # torch.nn.functional.relu 
        # It returns the output of the 3rd fully connected layer (fc3) without using activation function. 
        return self.fc3(x) 
    