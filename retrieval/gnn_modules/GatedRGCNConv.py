"""
This script defines ResidualGatedRGCNConv, a GNN layer that combines
a direct residual connection with aggregated neighborhood messages via a
GRU-like gating mechanism.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Union, Tuple, Optional

# Import for L2 normalization
import torch.nn.functional as F

import torch_sparse

from torch_geometric.nn import RGCNConv
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor, SparseTensor


def masked_edge_index(edge_index: Adj, edge_mask: Tensor) -> Adj:
    if isinstance(edge_index, Tensor):
        return edge_index[:, edge_mask]
    return torch_sparse.masked_select_nnz(edge_index, edge_mask, layout='coo')


class ResidualGatedRGCNConv(RGCNConv):
    """
    A Gated Relational Graph Convolutional Operator with a Direct Residual Path.
    
    This architecture is initialized to act as a near-perfect identity function
    by initializing the gate's weights to zero and its bias to a negative value.
    This ensures a stable starting point for training.
    """
    def __init__(
        self,
        channels: int,
        num_relations: int,
        gate_bias_init: float = -5.0,
        **kwargs,
    ):
        in_channels = out_channels = channels
        kwargs['root_weight'] = False
        kwargs['bias'] = False
        self.gate_bias_init = gate_bias_init

        super().__init__(in_channels, out_channels, num_relations, **kwargs)

        self.gate_nn = nn.Linear(channels * 2, channels)
        self.update_nn = nn.Linear(channels * 2, channels)

        self.reset_gate_parameters()

    def reset_gate_parameters(self):
        # Initialize gate weights to ZERO to eliminate random variance.
        torch.nn.init.zeros_(self.gate_nn.weight)
        # Initialize gate bias to a negative value.
        torch.nn.init.constant_(self.gate_nn.bias, self.gate_bias_init)
        
        # The update network can be initialized normally.
        glorot(self.update_nn.weight)
        zeros(self.update_nn.bias)

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, 'gate_nn'):
            self.reset_gate_parameters()

    def forward(self, x: Tensor, edge_index: Adj, edge_type: OptTensor = None):
        assert torch.is_floating_point(x), \
            "ResidualGatedRGCNConv requires floating point features."
        if self.in_channels != self.out_channels:
            raise ValueError("in_channels must equal out_channels for ResidualGatedRGCNConv")

        x_r = x
        x_l = x
        
        size = (x_l.size(0), x_r.size(0))
        aggr_out = torch.zeros(x_r.size(0), self.out_channels, device=x_r.device)

        weight = self.weight
        if self.num_bases is not None:
            weight = (self.comp @ weight.view(self.num_bases, -1)).view(
                self.num_relations, self.in_channels_l, self.out_channels)

        for i in range(self.num_relations):
            tmp = masked_edge_index(edge_index, edge_type == i)
            h = self.propagate(tmp, x=x_l, edge_type_ptr=None, size=size)
            aggr_out = aggr_out + (h @ weight[i])
        
        x_self = x_r
        combined = torch.cat([x_self, aggr_out], dim=-1)
        update_gate = torch.sigmoid(self.gate_nn(combined))
        update_candidate = torch.tanh(self.update_nn(combined))
        out = (1 - update_gate) * x_self + update_gate * update_candidate

        return out

### --------------------- DEMONSTRATION --------------------- ###

if __name__ == '__main__':
    # --- Shared Test Parameters ---
    channels = 32
    num_relations = 3
    num_nodes = 10
    num_edges = 50
    gate_init_val = -5.0
    relative_diff_threshold = 0.02

    # --- Test 1: Standard Normal Input ---
    print("--- Test 1: Standard Normal Input ---")
    x_input = torch.randn(num_nodes, channels)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
    edge_type = torch.randint(0, num_relations, (num_edges,), dtype=torch.long)
    
    layer = ResidualGatedRGCNConv(channels, num_relations, gate_bias_init=gate_init_val)
    layer.eval()

    with torch.no_grad():
        output = layer(x_input, edge_index, edge_type)

    diff_norm = torch.linalg.norm(x_input - output)
    input_norm = torch.linalg.norm(x_input)
    relative_diff = diff_norm / input_norm

    print(f"L2 Norm of input:          {input_norm.item():.4f}")
    print(f"L2 Norm of difference:     {diff_norm.item():.4f}")
    print(f"Relative difference:       {relative_diff.item():.4%}")
    print(f"Test threshold:            {relative_diff_threshold:.4%}")
    assert relative_diff < relative_diff_threshold, "Test failed for standard normal input."
    print("\nAssertion passed for standard normal input.")

    # --- Test 2: L2 Normalized Input ---
    print("\n" + "="*40)
    print("--- Test 2: L2 Normalized Input ---")
    
    # Create and L2 normalize the input tensor along the feature dimension
    x_unnormalized = torch.randn(num_nodes, channels)
    x_normalized = F.normalize(x_unnormalized, p=2, dim=1)
    
    # Verify that normalization worked
    norms = torch.linalg.norm(x_normalized, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms)), "L2 normalization failed."
    print("Input features successfully L2 normalized.")
    
    # Re-run the test with the normalized input
    with torch.no_grad():
        output_normalized = layer(x_normalized, edge_index, edge_type)

    diff_norm_norm = torch.linalg.norm(x_normalized - output_normalized)
    input_norm_norm = torch.linalg.norm(x_normalized)
    relative_diff_norm = diff_norm_norm / input_norm_norm

    print(f"L2 Norm of input:          {input_norm_norm.item():.4f}")
    print(f"L2 Norm of difference:     {diff_norm_norm.item():.4f}")
    print(f"Relative difference:       {relative_diff_norm.item():.4%}")
    print(f"Test threshold:            {relative_diff_threshold:.4%}")
    assert relative_diff_norm < relative_diff_threshold, "Test failed for L2 normalized input."
    print("\nAssertion passed for L2 normalized input as well.")