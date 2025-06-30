import numpy as np
import torch


import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs, include_input=True):
        """
        num_freqs: L (number of frequency bands)
        include_input: whether to include the original input p as part of the encoding
        """
        super(PositionalEncoding, self).__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input

    def forward(self, x):
        """
        x: tensor of shape (..., D)  # e.g. D=3 for 3D coordinates
        returns: encoded tensor of shape (..., encoded_dim)
        """
        # x is assumed to be in [B, ..., D] shape (any number of leading dims, then D)
        out = []

        if self.include_input:
            out.append(x)

        # For each frequency i in 0, 1, ..., L-1
        for i in range(self.num_freqs):
            freq = 2.0 ** i
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        
        # Concatenate along the last dimension
        return torch.cat(out, dim=-1)
        
        

def ReadScalarBinary(filename):
    data = np.fromfile(filename, dtype=np.float32)
    data = np.log10(data)
    return data

def ReadMPASOScalar(filename):
    try:
        data = np.load(filename)
    except:
        print(filename)
    return data

def pytorch_device_config(gpu_id=0):
    #  configuring device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        print('Running on the GPU')
    else:
        device = torch.device('cpu')
        print('Running on the CPU')
    return device

def ReadScalarSubdominBinary(filename, startidx:int, numItems:int):
    data = np.fromfile(filename, count= numItems, offset=startidx*4, dtype=np.float32)
    data = np.log10(data)
    return data



