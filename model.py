#-------------------------- Neural Network Project ----------------------------#

###--------------Model instanciation------------------------

import torch
import torch.nn as nn

class HeartNet(nn.Module): #creating a neural network class
    
    def __init__ (self, dropout=0.3):
        super().__init__() #calls the constructor of the parent class nn.Module
        # nn sequential stack layers in order
        self.network = nn.Sequential (
            nn.Linear(13, 16),       # layer 1 (Input: 13 features; Output: 16 neurons) 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 8),        # layer 2 (narrowing down)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(8, 1),         # output layer
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

