import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from itertools import combinations
import numpy as np
from utils import *

from moe import *
from models_moe.modules import *


class PosEncoding(torch.nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, num_frequencies=10):
        super().__init__()
        self.num_frequencies = num_frequencies

    def forward(self, coords):
        # Ultra-optimized vectorized implementation
        coords_pos_enc = coords
        
        # Create frequency bands and compute all encodings in one go
        freqs = 2.0 ** torch.arange(self.num_frequencies, device=coords.device, dtype=coords.dtype)
        
        # Reshape for broadcasting: [1, 1, num_freqs]
        freqs = freqs.view(1, 1, -1)
        
        # Compute all sin/cos values at once: [batch, ..., features, num_freqs]
        coords_scaled = coords.unsqueeze(-1) * freqs * np.pi
        
        # Stack sin and cos together, then reshape
        sin_cos = torch.stack([torch.sin(coords_scaled), torch.cos(coords_scaled)], dim=-1)
        encodings = sin_cos.view(*coords.shape[:-1], -1)  # Flatten last two dims
        
        return torch.cat([coords_pos_enc, encodings], dim=-1)


class Sine(nn.Module):
    def __init__(self, freq=30, trainable=False):
        super().__init__()
        if trainable:
            self.freq = nn.Parameter(torch.tensor(freq))
        else:
            self.freq = freq
    def forward(self, input):
        # See SIREN paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(self.freq * input)

    

class SingleHeadCrossAttention(nn.Module):
    def __init__(self, feature_dim, feature_dim_1d):
        super().__init__()
        self.feature_dim = feature_dim
        self.W_q = nn.Linear(feature_dim, feature_dim, bias=False)
        self.W_k = nn.Linear(feature_dim, feature_dim, bias=False)
        self.W_v = nn.Linear(feature_dim, feature_dim, bias=False)
        
        ######### Attention value adapter #########
        self.value_adapter = nn.Sequential(
            nn.Linear(feature_dim+feature_dim_1d, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, feature_dim),
            nn.LayerNorm(feature_dim),
        )
        # Adapter initialization:
        self.initialize_adapter()
        
    def initialize_adapter(self):
        for layer in self.value_adapter:
            if isinstance(layer, nn.Linear):
                init.xavier_normal_(layer.weight)
                init.normal_(layer.bias, mean=0, std=0.001)
              
                
    def forward(self, query, keys, values, top_k=16, chunk_size=128, param_feats=None):
        """
        Optimized version: No chunking, fully vectorized for speed.
        query: (batch_size, feature_dim)
        keys: (num_kv_pairs, feature_dim)
        values: (num_kv_pairs, feature_dim)
        """
        if len(query.shape) == 1:
            query = query.unsqueeze(0)
        Q = self.W_q(query).unsqueeze(1)  # (batch_size, 1, feature_dim)
        scale = self.feature_dim ** 0.5

        # Full attention over all keys at once
        scores = torch.matmul(Q, keys.t())  # (batch_size, 1, num_keys)
        final_scores, final_top_k_indices = torch.topk(scores, top_k, dim=-1)  # (batch_size, 1, top_k)
        final_top_k_indices = final_top_k_indices.squeeze(1)  # (batch_size, top_k)

        K_top = keys[final_top_k_indices]        # (batch_size, top_k, feature_dim)
        V_top = values[final_top_k_indices]      # (batch_size, top_k, feature_dim)

        # Adapter with residual
        batch_size, top_k, feat_dim = V_top.shape
        param_feats_expanded = param_feats.unsqueeze(1).expand(-1, top_k, -1)
        V_adapter_input = torch.cat([V_top, param_feats_expanded], dim=-1)  # (batch_size, top_k, feature_dim + feature_dim_1d)
        adapted_V_top = V_top + self.value_adapter(V_adapter_input)

        # Compute attention weights for top-k
        attn_scores_top_k = torch.bmm(Q, K_top.transpose(-2, -1)) / scale   # (batch_size, 1, top_k)
        attn_weights = F.softmax(attn_scores_top_k, dim=-1)                 # (batch_size, 1, top_k)

        # Weighted sum of values
        attended_features = torch.bmm(attn_weights, adapted_V_top)  # (batch_size, 1, feature_dim)
        return attended_features.squeeze(1)  # (batch_size, feature_dim)


class KVMemoryModel(nn.Module):
    def __init__(self, feat_shapes, num_entries=1024, key_dim=3, feature_dim_3d=64, feature_dim_1d=64, 
                    top_K=8, chunk_size=512, num_hidden_layers=-1, mlp_encoder_dim=128, 
                    mlp_hidden_dim=128, out_features=1, n_experts=-1, manager_net=None):
        super().__init__()
        self.feature_dim_3d = feature_dim_3d
        self.pe = PosEncoding()
        self.top_K = top_K
        self.chunk_size = chunk_size
        
        self.n_experts = n_experts
        self.manager_net = manager_net

        self.mlp_encoder_dim = mlp_encoder_dim
        self.num_hidden_layers = num_hidden_layers
        
        #  MLP:
        num_feat = feature_dim_3d
        self.encoder_mlp_list = nn.ModuleList()
        for _ in range(self.n_experts):
            encoder_mlp = FullyConnectedNN(
                in_features=63,
                out_features=num_feat, 
                num_hidden_layers=self.num_hidden_layers,
                hidden_features=self.mlp_encoder_dim,
                outermost_linear=True,
                nonlinearity='sine',
                init_type='siren',
                module_name='.encoder_mlp'
            )
            self.encoder_mlp_list.append(encoder_mlp)

        # Encoder MLP initialization
        self._initialize_weights()
            
        # Layer norm:
        self.layer_norm = nn.LayerNorm(feature_dim_3d)
        
        # Cross attention:
        self.cross_attn_experts = nn.ModuleList([
            SingleHeadCrossAttention(feature_dim_3d, feature_dim_1d) for _ in range(self.n_experts)
        ])
        
        self.memory_keys_list = nn.Parameter(torch.randn(self.n_experts, num_entries, key_dim))
        self.memory_values_list = nn.Parameter(torch.randn(self.n_experts, num_entries, feature_dim_3d))

        nn.init.kaiming_uniform_(self.memory_keys_list, nonlinearity='relu')
        nn.init.uniform_(self.memory_values_list, a=-0.001, b=0.001)

        # Simulation parameters: 
        self.line_dimid = list(range(3, 3+len(feat_shapes)))
        self.line_dims = feat_shapes
        
        # Pre-calculate initialization bounds for efficiency
        num_lines = len(self.line_dimid)
        if num_lines > 0:
            a_val = (0.01) ** (1.0 / num_lines)
            b_val = (0.02) ** (1.0 / num_lines)
        else:
            a_val, b_val = 0.01, 0.02
        
        self.lines = nn.Parameter(
            torch.empty(len(self.line_dims), feature_dim_1d, self.line_dims[0])
        )
        nn.init.uniform_(self.lines, a=a_val, b=b_val)
        
        # MLP:
        self.sigmoid = torch.nn.Sigmoid()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim_3d, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, out_features),
        )
        self.mlp = torch.jit.script(self.mlp)
        
        # Values initialization:
        self.initialize_mlp()


    def _initialize_weights(self):
        for encoder_mlp in self.encoder_mlp_list:
            for m in encoder_mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)   
                    nn.init.zeros_(m.bias)            

    def initialize_mlp(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                init.xavier_normal_(layer.weight)
                init.normal_(layer.bias, mean=0, std=0.001)
                

    def forward(self, x, tau=1.0, top_k=2):
        if self.n_experts == 1:
            top_k = 1
        
        # Embed queries:
        coords = x[..., :3]
        batch_size = x.size(0)
        
        # Embed simulation parameters: 
        param_feats = torch.ones(batch_size, device=x.device, dtype=x.dtype)
        
        for i, dimids in enumerate(self.line_dimid):
            p1d = x[:, dimids]
            p1dn = p1d * (self.line_dims[i] - 1)
            p1d_f = torch.floor(p1dn)
            weights = p1dn - p1d_f
            
            # Optimized indexing and clamping
            idx_low = p1d_f.to(torch.long)
            idx_high = torch.clamp(p1d_f + 1.0, min=0.0, max=self.line_dims[i] - 1).to(torch.long)
            
            # Vectorized linear interpolation
            f1d = torch.lerp(
                self.lines[i][:, idx_low], 
                self.lines[i][:, idx_high], 
                weights
            )
            
            if f1d.dim() > 1:
                f1d = f1d.squeeze()
            param_feats = param_feats * f1d
            
        param_feats = param_feats.T
        
        # Precompute projected keys and values for each expert
        precomputed_keys = [self.cross_attn_experts[eid].W_k(self.memory_keys_list[eid]) for eid in range(self.n_experts)]
        precomputed_values = [self.cross_attn_experts[eid].W_v(self.memory_values_list[eid]) for eid in range(self.n_experts)]
        
        # Gating network
        gate_inputs = coords
        raw_q = self.manager_net(gate_inputs)
        gating_probs = torch.nn.functional.softmax(raw_q, dim=-1)

        # Get top-k expert indices and values  
        gating_probs = torch.clamp(gating_probs, min=1e-8)
        topk_vals, topk_indices = torch.topk(gating_probs, k=top_k, dim=-1)
        
        '''normalize weights so that they sum to 1 (only if K > 1)'''
        topk_vals = topk_vals / torch.sum(topk_vals, dim=-1).unsqueeze(-1)

        batch_size = x.size(0)
        spatial_feats = coords.new_zeros((batch_size, self.feature_dim_3d))

        for i in range(top_k):
            expert_idx = topk_indices[:, i]  
            expert_weight = topk_vals[:, i]
            for eid in torch.unique(expert_idx):
                eid = eid.item()
                selected = (expert_idx == eid).nonzero(as_tuple=True)[0]
                if selected.numel() == 0:
                    continue
                coords_subset = coords[selected]
                param_feats_subset = param_feats[selected]
                
                mlp_feats = self.encoder_mlp_list[eid](self.pe(coords_subset))
                query_subset = self.layer_norm(mlp_feats)

                out = self.cross_attn_experts[eid](
                    query_subset, precomputed_keys[eid], precomputed_values[eid], self.top_K, self.chunk_size, 
                    param_feats_subset
                )
                out = out + mlp_feats

                weights = expert_weight[selected].unsqueeze(-1)
                spatial_feats[selected] += out * weights
        
        # Decoder
        refined_features = self.mlp(spatial_feats)
        refined_features = self.sigmoid(refined_features)
        return refined_features, raw_q
    
    
