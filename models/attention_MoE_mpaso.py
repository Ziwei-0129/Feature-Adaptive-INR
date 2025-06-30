import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from itertools import combinations
import numpy as np
from utils import *

from models.manager_mpaso import *
from models.modules import *


class PosEncoding(torch.nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, num_frequencies=10):
        super().__init__()
        self.num_frequencies = num_frequencies

    def forward(self, coords):
        coords_pos_enc = coords
        in_features = coords.shape[-1]

        for i in range(self.num_frequencies):
            for j in range(in_features):
                c = coords[..., j]
                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)
                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)
        return coords_pos_enc


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
    """Single-head scaled dot-product cross-attention without W_k and W_v."""
    def __init__(self, feature_dim, feature_dim_1d):
        super().__init__()
        self.feature_dim = feature_dim
        self.W_q = nn.Linear(feature_dim, feature_dim, bias=False)
        self.W_k = nn.Linear(feature_dim, feature_dim, bias=False)
        self.W_v = nn.Linear(feature_dim, feature_dim, bias=False)
        
        ######### Attention value adapter #########
        self.value_adapter = nn.Sequential(
            nn.Linear(feature_dim+feature_dim_1d, 64),
            nn.ReLU(),
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
        if len(query.shape) == 1:
            query = query.unsqueeze(0)
        Q = self.W_q(query).unsqueeze(1)
        keys = self.W_k(keys)
        values = self.W_v(values)
        scale = self.feature_dim ** 0.5

        attn_scores_topk_list = []
        top_k_indices_list = []

        for start in range(0, keys.shape[0], chunk_size):
            end = min(start + chunk_size, keys.shape[0])
            K_chunk = keys[start:end]
            partial_scores = torch.bmm(Q, K_chunk.t().unsqueeze(0).expand(Q.size(0), -1, -1))

            # Top-k selection within the chunk
            partial_topk_scores, partial_topk_indices = torch.topk(partial_scores, top_k, dim=-1)  # (batch_size, 1, top_k)
            partial_topk_indices += start                                     # Adjust indices to reflect global key indices

            attn_scores_topk_list.append(partial_topk_scores)
            top_k_indices_list.append(partial_topk_indices)

        attn_scores = torch.cat(attn_scores_topk_list, dim=-1)                # (batch_size, 1, top_k * num_chunks)
        top_k_indices = torch.cat(top_k_indices_list, dim=-1) 

        final_scores, final_indices = torch.topk(attn_scores, top_k, dim=-1)  # (batch_size, 1, top_k)
        final_top_k_indices = torch.gather(top_k_indices, 2, final_indices)   # (batch_size, 1, top_k)

        # Efficient gathering without expand
        K_top = keys[final_top_k_indices.squeeze(1)]        # (batch_size, top_k, feature_dim)
        V_top = values[final_top_k_indices.squeeze(1)]      # (batch_size, top_k, feature_dim)
        
        ##############  Adapter with residual  ##############
        batch_size, top_k, feat_dim = V_top.shape
        
        param_feats_expanded = param_feats.unsqueeze(1).expand(-1, top_k, -1)
        V_adapter_input = torch.cat([V_top, param_feats_expanded], dim=-1)  # (batch_size, top_k, feature_dim + feature_dim_1d)
        
        # Pass through adapter and add residual
        adapted_V_top = V_top + self.value_adapter(V_adapter_input)
    
        # Compute attention weights for final top-k selection
        attn_scores_top_k = torch.bmm(Q, K_top.transpose(-2, -1)) / scale   # (batch_size, 1, top_k)
        attn_weights = F.softmax(attn_scores_top_k, dim=-1)                 # (batch_size, 1, top_k)

        # Compute weighted sum of values
        attended_features = torch.bmm(attn_weights, adapted_V_top)  # (batch_size, 1, feature_dim)
        return attended_features.squeeze(1)  # (batch_size, feature_dim)



class KVMemoryModel(nn.Module):
    def __init__(self, feat_shapes, num_entries=1024, key_dim=3, feature_dim_3d=64, feature_dim_1d=64, 
                    top_K=16, chunk_size=512, num_hidden_layers=-1, mlp_encoder_dim=128, 
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
        num_feat = feature_dim_3d #16
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
                    module_name='.encoder_mlp')
            self.encoder_mlp_list.append(encoder_mlp)
        
        # Encoder MLP initialization
        self._initialize_weights()
            
        # Layer norm:
        self.layer_norm = nn.LayerNorm(feature_dim_3d)
        
        # Cross attention:
        self.cross_attn_experts = nn.ModuleList([
            SingleHeadCrossAttention(feature_dim_3d, feature_dim_1d) for _ in range(self.n_experts)
        ])
        self.memory_keys_list = torch.nn.ParameterList([
            nn.Parameter(torch.randn(num_entries, key_dim), requires_grad=True) for _ in range(self.n_experts)
        ])
        self.memory_values_list = torch.nn.ParameterList([
            nn.Parameter(torch.randn(num_entries, feature_dim_3d), requires_grad=True) for _ in range(self.n_experts)
        ])
        
        # Values initialization:
        for j in range(self.n_experts):
            torch.nn.init.kaiming_uniform_(self.memory_keys_list[j], nonlinearity='relu')
            torch.nn.init.uniform_(self.memory_values_list[j], a=-0.001, b=0.001)
        
        # Simulation parameters: 
        self.line_dimid = list(range(3, 3+len(feat_shapes)))
        self.line_dims = feat_shapes
        self.lines = []
        for i, dim in enumerate(self.line_dims):
            line = torch.nn.Parameter(
                torch.Tensor(feature_dim_1d, dim),
                requires_grad=True
            )
            # Values initialization:
            torch.nn.init.uniform_(line, a=(0.01)**(1/len(self.line_dimid)), b=(0.02)**(1/len(self.line_dimid)))
            self.lines.append(line)
        self.lines = torch.nn.ParameterList(self.lines)
  
        # MLP:
        self.sigmoid = torch.nn.Sigmoid()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim_3d, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, out_features),
        )
        # Values initialization:
        self.initialize_mlp()

    def _initialize_weights(self):
        for encoder_mlp in self.encoder_mlp_list:
            for m in encoder_mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)  # Xavier uniform initialization
                    nn.init.zeros_(m.bias)             # Set bias to zero

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
        
        # Embed simulation parameters: 
        param_feats = 1.
        for i, dimids in enumerate(self.line_dimid):
            p1d = x[:,dimids]
            p1dn = p1d*(self.line_dims[i]-1)
            p1d_f = torch.floor(p1dn)
            weights = p1dn-p1d_f
            f1d = torch.lerp(self.lines[i][:,p1d_f.to(torch.long)], self.lines[i][:,torch.clamp(p1d_f+1.0, min=0.0, max=self.line_dims[i]-1).to(torch.long)], weights)
            f1d = f1d.squeeze()
            param_feats = param_feats * f1d
            
        param_feats = param_feats.T
        
        # Gating network
        gate_inputs = coords
        raw_q = self.manager_net(gate_inputs)
        gating_probs = torch.nn.functional.softmax(raw_q, dim=-1)
        gating_probs = torch.clamp(gating_probs, 0.0)
        topk_vals, topk_indices = torch.topk(gating_probs, k=top_k, dim=-1)
        
        '''normalize weights so that they sum to 1 (only if K > 1)'''
        topk_vals = topk_vals / torch.sum(topk_vals, dim=-1).unsqueeze(-1)

        batch_size = x.size(0)
        spatial_feats = coords.new_zeros((batch_size, self.feature_dim_3d))

        for i in range(top_k):
            expert_idx = topk_indices[:, i]  # shape: [batch_size]
            expert_weight = topk_vals[:, i]  # shape: [batch_size]

            for eid in torch.unique(expert_idx):
                eid = eid.item()
                selected = (expert_idx == eid).nonzero(as_tuple=True)[0]
                if selected.numel() == 0:
                    continue

                coords_subset = coords[selected]
                param_feats_subset = param_feats[selected]
                
                mlp_feats = self.encoder_mlp_list[eid](self.pe(coords_subset))
                query_subset = self.layer_norm(mlp_feats)

                memory_keys = self.memory_keys_list[eid]
                memory_values = self.memory_values_list[eid]

                out = self.cross_attn_experts[eid](
                    query_subset, memory_keys, memory_values, self.top_K, self.chunk_size, 
                    param_feats_subset
                )
                out = out + mlp_feats  # residual

                weights = expert_weight[selected].unsqueeze(-1)
                spatial_feats[selected] += out * weights
        
        # Decoder
        refined_features = self.mlp(spatial_feats)
        refined_features = self.sigmoid(refined_features)
        return refined_features, raw_q


    def forward_top1(self, x, tau=1.0, top_k=1):
        # Embed queries:
        coords = x[..., :3]
        
        # Embed simulation parameters: 
        param_feats = 1.
        for i, dimids in enumerate(self.line_dimid):
            p1d = x[:,dimids]
            p1dn = p1d*(self.line_dims[i]-1)
            p1d_f = torch.floor(p1dn)
            weights = p1dn-p1d_f
            f1d = torch.lerp(
                self.lines[i][:,p1d_f.to(torch.long)], 
                self.lines[i][:,torch.clamp(p1d_f+1.0, min=0.0, max=self.line_dims[i]-1).to(torch.long)], 
                weights)
            f1d = f1d.squeeze()
            param_feats = param_feats * f1d
            
        param_feats = param_feats.T
        
        # Gating network
        gate_inputs = coords
        raw_q = self.manager_net(gate_inputs)
        gating_probs = torch.nn.functional.softmax(raw_q, dim=-1)

        # Get top-k expert indices (non-differentiable) and values (differentiable)
        gating_probs = torch.clamp(gating_probs, 0.0)
        topk_vals, topk_indices = torch.topk(gating_probs, k=top_k, dim=-1)

        # Create a differentiable gating mask
        gating_mask = torch.zeros_like(gating_probs).scatter_(
            dim=1, index=topk_indices, src=topk_vals
        )

        # Compute expert outputs
        spatial_feats = coords.new_zeros((coords.size(0), self.feature_dim_3d))

        for expert_idx in range(self.n_experts):
            # Extract gating weight for current expert
            expert_gate_weights = gating_mask[:, expert_idx]  # [batch_size]

            selected_idx = (expert_gate_weights > 0).nonzero(as_tuple=True)[0]
            if selected_idx.numel() == 0:
                continue  # if no data points routed to this expert

            coords_subset = coords[selected_idx]
            param_feats_subset = param_feats[selected_idx]

            # Forward pass for selected expert
            mlp_feats = self.encoder_mlp_list[expert_idx](self.pe(coords_subset))
            query_subset = self.layer_norm(mlp_feats)

            memory_keys = self.memory_keys_list[expert_idx]
            memory_values = self.memory_values_list[expert_idx]

            out = self.cross_attn_experts[expert_idx](
                query_subset, memory_keys, memory_values, self.top_K, self.chunk_size, 
                param_feats_subset,
            )
            out = out + mlp_feats    # residual

            spatial_feats[selected_idx] += out * expert_gate_weights[selected_idx].unsqueeze(-1)

        # Decoder
        refined_features = self.mlp(spatial_feats)
        refined_features = self.sigmoid(refined_features)
        return refined_features, raw_q
    