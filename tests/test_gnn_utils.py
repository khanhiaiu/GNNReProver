import pytest
import torch
from typing import List

# ==============================================================================
# IMPORT OFFICIAL PyG LAYERS AND FUNCTIONS TO BE TESTED
# ==============================================================================

from torch_geometric.nn import GCNConv, RGCNConv, GATConv, RGATConv

# Assuming your functions are in the specified location
from retrieval.gnn_utils import compute_ghost_node_embeddings, compute_ghost_node_embeddings_rgat, compute_ghost_node_embeddings_rgcn


# ==============================================================================
# PYTEST FIXTURES
# ==============================================================================

@pytest.fixture
def setup_gcn_data():
    """Provides a consistent set of data for GCN testing."""
    torch.manual_seed(42)
    in_channels, out_channels = 8, 16
    
    # Use the real GCNConv layer
    gcn_layer = GCNConv(in_channels, out_channels, normalize=True, add_self_loops=True)
    
    existing_node_embs = torch.randn(10, in_channels)
    original_edge_index = torch.tensor([[0, 1, 2, 3, 8], [1, 2, 3, 0, 9]], dtype=torch.long)
    ghost_node_initial_embs = [torch.randn(in_channels) for _ in range(2)]
    ghost_node_connections = [torch.tensor([0, 3]), torch.tensor([1, 2, 3, 9])]
    
    return {
        "gcn_layer": gcn_layer, "existing_node_embs": existing_node_embs,
        "original_edge_index": original_edge_index, "ghost_node_initial_embs": ghost_node_initial_embs,
        "ghost_node_connections": ghost_node_connections,
        "num_existing_nodes": 10, "num_ghost_nodes": 2
    }

@pytest.fixture
def setup_rgcn_data():
    """Provides a consistent set of data for RGCN testing."""
    torch.manual_seed(123)
    in_channels, out_channels, num_relations = 8, 16, 3
    
    existing_node_embs = torch.randn(10, in_channels)
    original_edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    original_edge_type = torch.tensor([0, 1, 2], dtype=torch.long)
    
    ghost_node_initial_embs = [torch.randn(in_channels) for _ in range(2)]
    ghost_node_connections = [
        (torch.tensor([0, 3, 5]), torch.tensor([0, 1, 0])),  # To ghost node 0
        (torch.tensor([1, 8]), torch.tensor([2, 1])),       # To ghost node 1
    ]
    
    return {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "num_relations": num_relations,
        "existing_node_embs": existing_node_embs,
        "original_edge_index": original_edge_index,
        "original_edge_type": original_edge_type,
        "ghost_node_initial_embs": ghost_node_initial_embs,
        "ghost_node_connections": ghost_node_connections,  # <<< THIS LINE IS NOW CORRECTLY INCLUDED
        "num_existing_nodes": 10,
        "num_ghost_nodes": 2
    }


# ==============================================================================
# GCNConv TESTS
# ==============================================================================

def test_gcn_computation_matches_full_pass(setup_gcn_data):
    """
    Tests if the efficient GCN function's output matches a full GCN pass.
    """
    data = setup_gcn_data
    efficient_output = compute_ghost_node_embeddings(
        gcn_layer=data["gcn_layer"],
        existing_node_embs=data["existing_node_embs"],
        original_edge_index=data["original_edge_index"],
        ghost_node_initial_embs=data["ghost_node_initial_embs"],
        ghost_node_connections=data["ghost_node_connections"],
    )
    
    # --- Ground truth calculation using a full pass ---
    num_existing = data["num_existing_nodes"]
    all_features = torch.cat([data["existing_node_embs"], torch.stack(data["ghost_node_initial_embs"])], dim=0)
    
    new_sources = torch.cat(data["ghost_node_connections"])
    new_dests = torch.cat([
        torch.full_like(conn, num_existing + i)
        for i, conn in enumerate(data["ghost_node_connections"])
    ])
    new_edges = torch.stack([new_sources, new_dests])
    
    aug_edge_index = torch.cat([data["original_edge_index"], new_edges], dim=1)
    
    # The layer's forward pass is the ground truth
    full_pass_all_nodes = data["gcn_layer"](all_features, aug_edge_index)
    ground_truth_output = full_pass_all_nodes[num_existing:]
    
    assert torch.allclose(efficient_output, ground_truth_output, atol=1e-6)


# ==============================================================================
# RGCNConv TESTS
# ==============================================================================

@pytest.mark.parametrize("aggr", ["mean", "add"])
@pytest.mark.parametrize("use_root", [True, False])
@pytest.mark.parametrize("num_bases", [None, 2])
def test_rgcn_computation_matches_full_pass(setup_rgcn_data, aggr, use_root, num_bases):
    """
    Tests if the efficient RGCN function's output matches a full RGCN pass.
    Covers aggregation, root weight, and basis regularization.
    """
    data = setup_rgcn_data
    rgcn_layer = RGCNConv(
        data["in_channels"], data["out_channels"], data["num_relations"],
        num_bases=num_bases, aggr=aggr, root_weight=use_root
    )
    
    # --- 1. Get result from the efficient function ---
    efficient_output = compute_ghost_node_embeddings_rgcn(
        rgcn_layer=rgcn_layer,
        existing_node_embs=data["existing_node_embs"],
        ghost_node_initial_embs=data["ghost_node_initial_embs"],
        ghost_node_connections=data["ghost_node_connections"],
    )

    # --- 2. Get ground truth result from a full pass ---
    num_existing = data["num_existing_nodes"]
    all_features = torch.cat([data["existing_node_embs"], torch.stack(data["ghost_node_initial_embs"])], dim=0)
    
    all_new_sources, all_new_dests, all_new_types = [], [], []
    for i, (sources, types) in enumerate(data["ghost_node_connections"]):
        all_new_sources.append(sources)
        all_new_dests.append(torch.full_like(sources, num_existing + i))
        all_new_types.append(types)
    
    new_edges = torch.stack([torch.cat(all_new_sources), torch.cat(all_new_dests)])
    new_types = torch.cat(all_new_types)

    aug_edge_index = torch.cat([data["original_edge_index"], new_edges], dim=1)
    aug_edge_type = torch.cat([data["original_edge_type"], new_types])

    # The layer's forward pass is the ground truth
    full_pass_all_nodes = rgcn_layer(all_features, aug_edge_index, aug_edge_type)
    ground_truth_output = full_pass_all_nodes[num_existing:]

    # --- 3. Assertions ---
    assert efficient_output.shape == ground_truth_output.shape
    assert efficient_output.shape[0] == data["num_ghost_nodes"]
    # Use a slightly higher tolerance for RGCN due to potential floating point differences in aggregation
    assert torch.allclose(efficient_output, ground_truth_output, atol=1e-5)

def test_rgcn_no_ghost_nodes(setup_rgcn_data):
    """Tests the edge case where no ghost nodes are provided."""
    data = setup_rgcn_data
    rgcn_layer = RGCNConv(data["in_channels"], data["out_channels"], data["num_relations"])
    
    output = compute_ghost_node_embeddings_rgcn(
        rgcn_layer=rgcn_layer,
        existing_node_embs=data["existing_node_embs"],
        ghost_node_initial_embs=[],
        ghost_node_connections=[],
    )
    
    assert isinstance(output, torch.Tensor)
    assert output.shape == (0, data["out_channels"])

def test_rgcn_ghost_node_with_no_connections(setup_rgcn_data):
    """
    Tests a ghost node with no incoming edges. Its embedding should be based
    only on the root transformation and bias.
    """
    data = setup_rgcn_data
    rgcn_layer = RGCNConv(
        data["in_channels"], data["out_channels"], data["num_relations"], root_weight=True
    )

    ghost_node_connections = [
        data["ghost_node_connections"][0],
        (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)),
    ]
    
    efficient_output = compute_ghost_node_embeddings_rgcn(
        rgcn_layer=rgcn_layer,
        existing_node_embs=data["existing_node_embs"],
        ghost_node_initial_embs=data["ghost_node_initial_embs"],
        ghost_node_connections=ghost_node_connections,
    )
    
    # Ground truth for the disconnected node is just root + bias
    disconnected_node_emb = data["ghost_node_initial_embs"][1]
    ground_truth_disconnected = disconnected_node_emb @ rgcn_layer.root + rgcn_layer.bias
    
    assert torch.allclose(efficient_output[1], ground_truth_disconnected, atol=1e-6)

import pytest
import torch
# ... (other imports) ...

# Import the official GATConv layer

# Import all three of your utility functions
from retrieval.gnn_utils import (
    compute_ghost_node_embeddings,
    compute_ghost_node_embeddings_rgcn,
    compute_ghost_node_embeddings_gat,
)

# ... (existing GCN and RGCN fixtures and tests) ...


# ==============================================================================
# GATConv FIXTURE AND TESTS
# ==============================================================================

@pytest.fixture
def setup_gat_data():
    """Provides a consistent set of data for GAT testing."""
    torch.manual_seed(456)
    in_channels, out_channels = 8, 16
    
    existing_node_embs = torch.randn(10, in_channels)
    original_edge_index = torch.tensor([[0, 1, 2, 4, 8], [1, 2, 0, 5, 9]], dtype=torch.long)
    
    ghost_node_initial_embs = [torch.randn(in_channels) for _ in range(2)]
    ghost_node_connections = [
        torch.tensor([0, 3, 5]),      # To ghost node 0
        torch.tensor([1, 2, 5, 9]),   # To ghost node 1
    ]
    
    return {
        "in_channels": in_channels, "out_channels": out_channels,
        "existing_node_embs": existing_node_embs,
        "original_edge_index": original_edge_index,
        "ghost_node_initial_embs": ghost_node_initial_embs,
        "ghost_node_connections": ghost_node_connections,
        "num_existing_nodes": 10, "num_ghost_nodes": 2
    }


@pytest.mark.parametrize("heads", [1, 4])
@pytest.mark.parametrize("concat", [True, False])
@pytest.mark.parametrize("add_self_loops", [True, False])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("residual", [True, False])
def test_gat_computation_matches_full_pass(setup_gat_data, heads, concat, add_self_loops, bias, residual):
    """
    Tests if the efficient GAT function's output matches a full GAT pass.
    Covers heads, concat, self-loops, bias, and residual connections.
    """
    data = setup_gat_data
    # Skip invalid combination: residual connection requires input and output channels to match
    if residual and (data["in_channels"] != data["out_channels"] * (heads if concat else 1)):
        pytest.skip("Residual connection requires matching input/output channels.")

    gat_layer = GATConv(
        data["in_channels"], data["out_channels"], heads=heads, concat=concat,
        add_self_loops=add_self_loops, bias=bias, residual=residual,
    )
    
    # --- 1. Get result from the efficient function ---
    efficient_output = compute_ghost_node_embeddings_gat(
        gat_layer=gat_layer,
        existing_node_embs=data["existing_node_embs"],
        ghost_node_initial_embs=data["ghost_node_initial_embs"],
        ghost_node_connections=data["ghost_node_connections"],
    )
    
    # --- 2. Get ground truth result from a full pass ---
    num_existing = data["num_existing_nodes"]
    all_features = torch.cat([data["existing_node_embs"], torch.stack(data["ghost_node_initial_embs"])], dim=0)
    
    new_sources = torch.cat(data["ghost_node_connections"])
    new_dests = torch.cat([
        torch.full_like(conn, num_existing + i)
        for i, conn in enumerate(data["ghost_node_connections"])
    ])
    new_edges = torch.stack([new_sources, new_dests])
    
    aug_edge_index = torch.cat([data["original_edge_index"], new_edges], dim=1)

    # The layer's forward pass is the ground truth
    full_pass_all_nodes = gat_layer(all_features, aug_edge_index)
    ground_truth_output = full_pass_all_nodes[num_existing:]
    
    # --- 3. Assertions ---
    assert efficient_output.shape == ground_truth_output.shape
    assert torch.allclose(efficient_output, ground_truth_output, atol=1e-5)

def test_gat_no_ghost_nodes(setup_gat_data):
    """Tests the edge case where no ghost nodes are provided."""
    data = setup_gat_data
    gat_layer = GATConv(data["in_channels"], data["out_channels"])
    
    output = compute_ghost_node_embeddings_gat(
        gat_layer=gat_layer,
        existing_node_embs=data["existing_node_embs"],
        ghost_node_initial_embs=[],
        ghost_node_connections=[],
    )
    
    assert isinstance(output, torch.Tensor)
    assert output.shape[0] == 0


@pytest.fixture
def setup_rgat_data():
    """Provides a consistent set of data for RGAT testing."""
    torch.manual_seed(789)
    in_channels, out_channels, num_relations = 8, 4, 3
    
    existing_node_embs = torch.randn(10, in_channels)
    original_edge_index = torch.tensor([[0, 1, 2, 4], [1, 2, 0, 5]], dtype=torch.long)
    original_edge_type = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    
    ghost_node_initial_embs = [torch.randn(in_channels) for _ in range(2)]
    ghost_node_connections = [
        (torch.tensor([0, 3, 5]), torch.tensor([0, 1, 2])),  # To ghost node 0
        (torch.tensor([1, 5, 9]), torch.tensor([2, 1, 0])),  # To ghost node 1
    ]
    
    return {
        "in_channels": in_channels, "out_channels": out_channels, "num_relations": num_relations,
        "existing_node_embs": existing_node_embs, "original_edge_index": original_edge_index,
        "original_edge_type": original_edge_type, "ghost_node_initial_embs": ghost_node_initial_embs,
        "ghost_node_connections": ghost_node_connections,
        "num_existing_nodes": 10, "num_ghost_nodes": 2
    }

@pytest.mark.parametrize("heads", [1, 2])
@pytest.mark.parametrize("concat", [True, False])
@pytest.mark.parametrize("attention_mechanism", ["within-relation", "across-relation"])
@pytest.mark.parametrize("attention_mode", ["additive-self-attention", "multiplicative-self-attention"])
@pytest.mark.parametrize("num_bases", [None, 4])
def test_rgat_computation_matches_full_pass(setup_rgat_data, heads, concat, attention_mechanism, attention_mode, num_bases):
    """
    Tests if the efficient RGAT function's output matches a full RGAT pass.
    """
    data = setup_rgat_data
    dim = 2 if attention_mode == "multiplicative-self-attention" else 1

    rgat_layer = RGATConv(
        data["in_channels"], data["out_channels"], data["num_relations"],
        heads=heads, concat=concat, dim=dim, attention_mechanism=attention_mechanism,
        attention_mode=attention_mode, num_bases=num_bases
    )

    # --- 1. Get result from the efficient function ---
    efficient_output = compute_ghost_node_embeddings_rgat(
        rgat_layer=rgat_layer,
        existing_node_embs=data["existing_node_embs"],
        ghost_node_initial_embs=data["ghost_node_initial_embs"],
        ghost_node_connections=data["ghost_node_connections"],
    )

    # --- 2. Get ground truth result from a full pass ---
    num_existing = data["num_existing_nodes"]
    all_features = torch.cat([data["existing_node_embs"], torch.stack(data["ghost_node_initial_embs"])], dim=0)
    
    all_new_sources, all_new_dests, all_new_types = [], [], []
    for i, (sources, types) in enumerate(data["ghost_node_connections"]):
        all_new_sources.append(sources)
        all_new_dests.append(torch.full_like(sources, num_existing + i))
        all_new_types.append(types)
    
    new_edges = torch.stack([torch.cat(all_new_sources), torch.cat(all_new_dests)])
    new_types = torch.cat(all_new_types)

    aug_edge_index = torch.cat([data["original_edge_index"], new_edges], dim=1)
    aug_edge_type = torch.cat([data["original_edge_type"], new_types])

    # The layer's forward pass is the ground truth
    full_pass_all_nodes = rgat_layer(all_features, aug_edge_index, aug_edge_type)
    ground_truth_output = full_pass_all_nodes[num_existing:]

    # --- 3. Assertions ---
    assert efficient_output.shape == ground_truth_output.shape
    assert torch.allclose(efficient_output, ground_truth_output, atol=1e-5)

def test_rgat_ghost_node_with_no_connections(setup_rgat_data):
    """
    Tests a ghost node with no incoming edges. The output should just be the bias.
    """
    data = setup_rgat_data
    rgat_layer = RGATConv(data["in_channels"], data["out_channels"], data["num_relations"])
    
    output = compute_ghost_node_embeddings_rgat(
        rgat_layer=rgat_layer,
        existing_node_embs=data["existing_node_embs"],
        ghost_node_initial_embs=data["ghost_node_initial_embs"],
        ghost_node_connections=[(torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))] * 2,
    )
    
    # Ground truth for disconnected nodes is just the bias repeated
    ground_truth = rgat_layer.bias.unsqueeze(0).repeat(data["num_ghost_nodes"], 1)
    
    assert torch.allclose(output, ground_truth, atol=1e-6)

import pytest
import torch
from torch import nn
# ... (other imports) ...

# Import the official GINConv and GINEConv layers
from torch_geometric.nn import GCNConv, RGCNConv, GATConv, RGATConv, GINConv, GINEConv

# Import all six of your utility functions
from retrieval.gnn_utils import (
    compute_ghost_node_embeddings,
    compute_ghost_node_embeddings_rgcn,
    compute_ghost_node_embeddings_gat,
    compute_ghost_node_embeddings_rgat,
    compute_ghost_node_embeddings_gin,
    compute_ghost_node_embeddings_gine,
)

# ... (existing fixtures and tests for GCN, RGCN, GAT, RGAT) ...


# ==============================================================================
# GINConv and GINEConv FIXTURES AND TESTS
# ==============================================================================

@pytest.fixture
def setup_gin_data():
    """Provides a consistent set of data for GIN and GINE testing."""
    torch.manual_seed(101)
    in_channels, out_channels, edge_dim = 16, 32, 8
    
    mlp = nn.Sequential(nn.Linear(in_channels, out_channels), nn.ReLU(), nn.Linear(out_channels, out_channels))
    
    existing_node_embs = torch.randn(10, in_channels)
    original_edge_index = torch.tensor([[0, 1, 2, 4], [1, 2, 0, 5]], dtype=torch.long)
    original_edge_attr = torch.randn(original_edge_index.size(1), edge_dim)
    
    ghost_node_initial_embs = [torch.randn(in_channels) for _ in range(2)]
    
    # Data for GINConv
    gin_connections = [torch.tensor([0, 3, 5]), torch.tensor([1, 5, 9])]
    
    # Data for GINEConv (includes edge attributes)
    gine_connections = [
        (gin_connections[0], torch.randn(len(gin_connections[0]), edge_dim)),
        (gin_connections[1], torch.randn(len(gin_connections[1]), edge_dim)),
    ]
    
    return {
        "mlp": mlp, "in_channels": in_channels, "out_channels": out_channels, "edge_dim": edge_dim,
        "existing_node_embs": existing_node_embs, "original_edge_index": original_edge_index,
        "original_edge_attr": original_edge_attr, "ghost_node_initial_embs": ghost_node_initial_embs,
        "gin_connections": gin_connections, "gine_connections": gine_connections,
        "num_existing_nodes": 10, "num_ghost_nodes": 2
    }

@pytest.mark.parametrize("train_eps", [True, False])
def test_gin_computation_matches_full_pass(setup_gin_data, train_eps):
    """Tests if the efficient GIN function's output matches a full GIN pass."""
    data = setup_gin_data
    gin_layer = GINConv(nn=data["mlp"], train_eps=train_eps)

    efficient_output = compute_ghost_node_embeddings_gin(
        gin_layer, data["existing_node_embs"], data["ghost_node_initial_embs"], data["gin_connections"]
    )

    # Ground truth
    num_existing = data["num_existing_nodes"]
    all_features = torch.cat([data["existing_node_embs"], torch.stack(data["ghost_node_initial_embs"])], dim=0)
    
    new_sources = torch.cat(data["gin_connections"])
    new_dests = torch.cat([torch.full_like(c, num_existing + i) for i, c in enumerate(data["gin_connections"])])
    new_edges = torch.stack([new_sources, new_dests])
    aug_edge_index = torch.cat([data["original_edge_index"], new_edges], dim=1)
    
    full_pass_all = gin_layer(all_features, aug_edge_index)
    ground_truth = full_pass_all[num_existing:]
    
    assert torch.allclose(efficient_output, ground_truth, atol=1e-6)

@pytest.mark.parametrize("train_eps", [True, False])
def test_gine_computation_matches_full_pass(setup_gin_data, train_eps):
    """Tests if the efficient GINE function's output matches a full GINE pass."""
    data = setup_gin_data
    # For GINE, edge_dim must be handled correctly
    gine_layer = GINEConv(nn=data["mlp"], train_eps=train_eps, edge_dim=data["edge_dim"])

    efficient_output = compute_ghost_node_embeddings_gine(
        gine_layer, data["existing_node_embs"], data["ghost_node_initial_embs"], data["gine_connections"]
    )

    # Ground truth
    num_existing = data["num_existing_nodes"]
    all_features = torch.cat([data["existing_node_embs"], torch.stack(data["ghost_node_initial_embs"])], dim=0)
    
    new_sources = torch.cat([c[0] for c in data["gine_connections"]])
    new_attrs = torch.cat([c[1] for c in data["gine_connections"]])
    new_dests = torch.cat([torch.full_like(c[0], num_existing + i) for i, c in enumerate(data["gine_connections"])])
    
    new_edges = torch.stack([new_sources, new_dests])
    aug_edge_index = torch.cat([data["original_edge_index"], new_edges], dim=1)
    aug_edge_attr = torch.cat([data["original_edge_attr"], new_attrs], dim=0)

    full_pass_all = gine_layer(all_features, aug_edge_index, aug_edge_attr)
    ground_truth = full_pass_all[num_existing:]
    
    assert torch.allclose(efficient_output, ground_truth, atol=1e-6)

def test_gin_ghost_node_with_no_connections(setup_gin_data):
    """Tests a GIN ghost node with no incoming edges."""
    data = setup_gin_data
    gin_layer = GINConv(nn=data["mlp"])
    
    # Ghost node 1 has no connections
    connections = [data["gin_connections"][0], torch.tensor([], dtype=torch.long)]
    
    output = compute_ghost_node_embeddings_gin(
        gin_layer, data["existing_node_embs"], data["ghost_node_initial_embs"], connections
    )
    
    # Ground truth for disconnected node is just nn((1+eps)*x_i)
    disconnected_emb = data["ghost_node_initial_embs"][1]
    ground_truth = gin_layer.nn((1 + gin_layer.eps) * disconnected_emb)
    
    assert torch.allclose(output[1], ground_truth, atol=1e-6)

@pytest.fixture
def base_data():
    """Provides a consistent set of raw data for all tests."""
    torch.manual_seed(101)
    in_channels, out_channels, edge_dim, num_relations = 16, 32, 8, 3
    num_existing_nodes, num_ghost_nodes = 10, 2
    
    existing_node_embs = torch.randn(num_existing_nodes, in_channels)
    original_edge_index = torch.tensor([[0, 1, 2, 4, 8], [1, 2, 0, 5, 9]], dtype=torch.long)
    original_edge_type = torch.randint(0, num_relations, (original_edge_index.size(1),), dtype=torch.long)
    original_edge_attr = torch.randn(original_edge_index.size(1), edge_dim)
    
    ghost_node_initial_embs = [torch.randn(in_channels) for _ in range(num_ghost_nodes)]
    
    # Define connections once to be reused
    base_connections = [torch.tensor([0, 3, 5]), torch.tensor([1, 5, 9])]
    
    # Connections for GIN/GAT (no edge types/attrs)
    simple_connections = base_connections
    
    # Connections for RGCN/RGAT (with edge types)
    relational_connections = [
        (base_connections[0], torch.randint(0, num_relations, (len(base_connections[0]),))),
        (base_connections[1], torch.randint(0, num_relations, (len(base_connections[1]),))),
    ]
    
    # Connections for GINE (with edge attributes)
    edge_attr_connections = [
        (base_connections[0], torch.randn(len(base_connections[0]), edge_dim)),
        (base_connections[1], torch.randn(len(base_connections[1]), edge_dim)),
    ]

    return {
        "in_channels": in_channels, "out_channels": out_channels, "edge_dim": edge_dim, "num_relations": num_relations,
        "existing_node_embs": existing_node_embs, "original_edge_index": original_edge_index,
        "original_edge_type": original_edge_type, "original_edge_attr": original_edge_attr,
        "ghost_node_initial_embs": ghost_node_initial_embs,
        "simple_connections": simple_connections, "relational_connections": relational_connections,
        "edge_attr_connections": edge_attr_connections,
        "num_existing_nodes": num_existing_nodes, "num_ghost_nodes": num_ghost_nodes
    }

# ==============================================================================
# GCNConv TESTS
# ==============================================================================

@pytest.mark.parametrize("improved", [True, False])
@pytest.mark.parametrize("add_self_loops", [True, False])
@pytest.mark.parametrize("bias", [True, False])
def test_gcn_computation_matches_full_pass(base_data, improved, add_self_loops, bias):
    data = base_data
    gcn_layer = GCNConv(
        data["in_channels"], data["out_channels"],
        improved=improved, add_self_loops=add_self_loops, bias=bias
    )
    
    efficient_output = compute_ghost_node_embeddings(
        gcn_layer,
        data["existing_node_embs"],
        data["original_edge_index"],  # This argument was missing
        data["ghost_node_initial_embs"],
        data["simple_connections"]
    )
    
    num_existing = data["num_existing_nodes"]
    all_features = torch.cat([data["existing_node_embs"], torch.stack(data["ghost_node_initial_embs"])], dim=0)
    new_sources = torch.cat(data["simple_connections"])
    new_dests = torch.cat([torch.full_like(c, num_existing + i) for i, c in enumerate(data["simple_connections"])])
    new_edges = torch.stack([new_sources, new_dests])
    aug_edge_index = torch.cat([data["original_edge_index"], new_edges], dim=1)
    
    ground_truth = gcn_layer(all_features, aug_edge_index)[num_existing:]
    assert torch.allclose(efficient_output, ground_truth, atol=1e-6)

# ==============================================================================
# RGCNConv TESTS
# ==============================================================================

@pytest.mark.parametrize("aggr", ["mean", "add"])
@pytest.mark.parametrize("num_bases", [None, 4])
@pytest.mark.parametrize("root_weight", [True, False])
@pytest.mark.parametrize("bias", [True, False])
def test_rgcn_computation_matches_full_pass(base_data, aggr, num_bases, root_weight, bias):
    data = base_data
    rgcn_layer = RGCNConv(
        data["in_channels"], data["out_channels"], data["num_relations"],
        num_bases=num_bases, aggr=aggr, root_weight=root_weight, bias=bias
    )

    efficient_output = compute_ghost_node_embeddings_rgcn(
        rgcn_layer, data["existing_node_embs"], data["ghost_node_initial_embs"], data["relational_connections"]
    )

    num_existing = data["num_existing_nodes"]
    all_features = torch.cat([data["existing_node_embs"], torch.stack(data["ghost_node_initial_embs"])], dim=0)
    new_sources = torch.cat([c[0] for c in data["relational_connections"]])
    new_types = torch.cat([c[1] for c in data["relational_connections"]])
    new_dests = torch.cat([torch.full_like(c[0], num_existing + i) for i, c in enumerate(data["relational_connections"])])
    new_edges = torch.stack([new_sources, new_dests])
    aug_edge_index = torch.cat([data["original_edge_index"], new_edges], dim=1)
    aug_edge_type = torch.cat([data["original_edge_type"], new_types])
    
    ground_truth = rgcn_layer(all_features, aug_edge_index, aug_edge_type)[num_existing:]
    assert torch.allclose(efficient_output, ground_truth, atol=1e-5)

# ==============================================================================
# GATConv TESTS
# ==============================================================================

@pytest.mark.parametrize("heads", [1, 4])
@pytest.mark.parametrize("concat", [True, False])
@pytest.mark.parametrize("add_self_loops", [True, False])
@pytest.mark.parametrize("bias", [True, False])
def test_gat_computation_matches_full_pass(base_data, heads, concat, add_self_loops, bias):
    data = base_data
    gat_layer = GATConv(
        data["in_channels"], data["out_channels"], heads=heads, concat=concat,
        add_self_loops=add_self_loops, bias=bias
    )

    efficient_output = compute_ghost_node_embeddings_gat(
        gat_layer, data["existing_node_embs"], data["ghost_node_initial_embs"], data["simple_connections"]
    )
    
    num_existing = data["num_existing_nodes"]
    all_features = torch.cat([data["existing_node_embs"], torch.stack(data["ghost_node_initial_embs"])], dim=0)
    new_sources = torch.cat(data["simple_connections"])
    new_dests = torch.cat([torch.full_like(c, num_existing + i) for i, c in enumerate(data["simple_connections"])])
    new_edges = torch.stack([new_sources, new_dests])
    aug_edge_index = torch.cat([data["original_edge_index"], new_edges], dim=1)
    
    ground_truth = gat_layer(all_features, aug_edge_index)[num_existing:]
    assert torch.allclose(efficient_output, ground_truth, atol=1e-5)

# ==============================================================================
# RGATConv TESTS
# ==============================================================================

@pytest.mark.parametrize("heads", [1, 2])
@pytest.mark.parametrize("concat", [True, False])
@pytest.mark.parametrize("attention_mechanism", ["within-relation", "across-relation"])
@pytest.mark.parametrize("attention_mode", ["additive-self-attention", "multiplicative-self-attention"])
@pytest.mark.parametrize("num_bases", [None, 4])
def test_rgat_computation_matches_full_pass(base_data, heads, concat, attention_mechanism, attention_mode, num_bases):
    data = base_data
    dim = 2 if attention_mode == "multiplicative-self-attention" else 1

    rgat_layer = RGATConv(
        data["in_channels"], data["out_channels"], data["num_relations"],
        heads=heads, concat=concat, dim=dim, attention_mechanism=attention_mechanism,
        attention_mode=attention_mode, num_bases=num_bases
    )

    efficient_output = compute_ghost_node_embeddings_rgat(
        rgat_layer, data["existing_node_embs"], data["ghost_node_initial_embs"], data["relational_connections"]
    )

    num_existing = data["num_existing_nodes"]
    all_features = torch.cat([data["existing_node_embs"], torch.stack(data["ghost_node_initial_embs"])], dim=0)
    new_sources = torch.cat([c[0] for c in data["relational_connections"]])
    new_types = torch.cat([c[1] for c in data["relational_connections"]])
    new_dests = torch.cat([torch.full_like(c[0], num_existing + i) for i, c in enumerate(data["relational_connections"])])
    new_edges = torch.stack([new_sources, new_dests])
    aug_edge_index = torch.cat([data["original_edge_index"], new_edges], dim=1)
    aug_edge_type = torch.cat([data["original_edge_type"], new_types])
    
    ground_truth = rgat_layer(all_features, aug_edge_index, aug_edge_type)[num_existing:]
    assert torch.allclose(efficient_output, ground_truth, atol=1e-5)

# ==============================================================================
# GINConv and GINEConv TESTS
# ==============================================================================

def create_mlp(in_channels, out_channels):
    return nn.Sequential(
        nn.Linear(in_channels, out_channels * 2), nn.ReLU(), nn.Linear(out_channels * 2, out_channels)
    )

@pytest.mark.parametrize("train_eps", [True, False])
def test_gin_computation_matches_full_pass(base_data, train_eps):
    data = base_data
    mlp = create_mlp(data["in_channels"], data["out_channels"])
    gin_layer = GINConv(nn=mlp, train_eps=train_eps)

    efficient_output = compute_ghost_node_embeddings_gin(
        gin_layer, data["existing_node_embs"], data["ghost_node_initial_embs"], data["simple_connections"]
    )

    num_existing = data["num_existing_nodes"]
    all_features = torch.cat([data["existing_node_embs"], torch.stack(data["ghost_node_initial_embs"])], dim=0)
    new_sources = torch.cat(data["simple_connections"])
    new_dests = torch.cat([torch.full_like(c, num_existing + i) for i, c in enumerate(data["simple_connections"])])
    new_edges = torch.stack([new_sources, new_dests])
    aug_edge_index = torch.cat([data["original_edge_index"], new_edges], dim=1)
    
    ground_truth = gin_layer(all_features, aug_edge_index)[num_existing:]
    assert torch.allclose(efficient_output, ground_truth, atol=1e-6)


@pytest.mark.parametrize("train_eps", [True, False])
def test_gine_computation_matches_full_pass(base_data, train_eps):
    data = base_data
    mlp = create_mlp(data["in_channels"], data["out_channels"])
    gine_layer = GINEConv(nn=mlp, train_eps=train_eps, edge_dim=data["edge_dim"])

    efficient_output = compute_ghost_node_embeddings_gine(
        gine_layer, data["existing_node_embs"], data["ghost_node_initial_embs"], data["edge_attr_connections"]
    )

    num_existing = data["num_existing_nodes"]
    all_features = torch.cat([data["existing_node_embs"], torch.stack(data["ghost_node_initial_embs"])], dim=0)
    new_sources = torch.cat([c[0] for c in data["edge_attr_connections"]])
    new_attrs = torch.cat([c[1] for c in data["edge_attr_connections"]])
    new_dests = torch.cat([torch.full_like(c[0], num_existing + i) for i, c in enumerate(data["edge_attr_connections"])])
    new_edges = torch.stack([new_sources, new_dests])
    aug_edge_index = torch.cat([data["original_edge_index"], new_edges], dim=1)
    aug_edge_attr = torch.cat([data["original_edge_attr"], new_attrs], dim=0)

    ground_truth = gine_layer(all_features, aug_edge_index, aug_edge_attr)[num_existing:]
    assert torch.allclose(efficient_output, ground_truth, atol=1e-6)