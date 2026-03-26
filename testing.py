#-------------------------- Neural Network Project ----------------------------#

### Testing Code Module ###

import torch
import torch.nn as nn

class HeartNet(nn.Module): #creating a neural network class
    
    def __init__ (self):
        super().__init__() #calls the constructor of the parent class nn.Module
        # nn sequential stack layers in order
        self.network = nn.Sequential (
            nn.Linear(13, 16),       # layer 1 (Input: 13 features; Output: 16 neurons) 
            nn.ReLU(),
            nn.Linear(16, 8),        # layer 2 (narrowing down)
            nn.ReLU(),
            nn.Linear(8, 1),         # output layer
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# create module
model = HeartNet()

# Print model structure
print(model)

# Fake batch of 32 patients, 13 features each
fake_batch = torch.randn(32, 13)

# Forward pass — no training yet, weights are random
output = model(fake_batch)

print(output.shape)    # torch.Size([32, 1]) ← one probability per patient
print(output[:5])      # 5 random probabilities between 0 and 1
