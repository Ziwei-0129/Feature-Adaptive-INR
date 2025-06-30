import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class tSoftMax(nn.Module):
    def __init__(self, temperature, dim=-1, trainable=False):
        super().__init__()
        if trainable:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.temperature = temperature
        self.dim = dim
        self.activation = nn.Softmax(dim=dim)

    def forward(self, x):
        return self.activation(x / self.temperature)
    
    



class Manager(nn.Module):
    def __init__(self, resolution=16, n_experts=-1, hidden_dim=64):
        super().__init__()
        self.resolution = resolution
        self.n_experts = n_experts
        self.grid_shape = [resolution, resolution, resolution]
        self.num_feat_grid = 16
        
        self.grid = torch.nn.Parameter(
            torch.Tensor(1, self.num_feat_grid, *reversed(self.grid_shape)),
            requires_grad=True
        )
        # Initialize:
        nn.init.uniform_(self.grid, a=-0.001, b=0.001)
        
        self.linear = nn.Linear(self.num_feat_grid, self.n_experts)
        # Initialize:
        nn.init.orthogonal_(self.linear.weight.data)
        self.linear.bias.data.fill_(0.0)
        
        # self.norm = nn.LayerNorm(self.num_feat_grid)
        
        
    def forward(self, coords):
        batch_size = coords.shape[0]
        sample_grid = coords.view(1, batch_size, 1, 1, 3)

        # Sample single feature vector per coordinate
        sampled = F.grid_sample(self.grid, sample_grid, mode='bilinear', align_corners=True)
        sampled = sampled.reshape(self.num_feat_grid, batch_size).T  # [batch_size, num_feat_grid]

        # Produce logits directly
        logits = self.linear(sampled)  # [batch_size, n_experts]
        # logits = self.linear(self.norm(sampled))  # [batch_size, n_experts]
        return logits
        
        
