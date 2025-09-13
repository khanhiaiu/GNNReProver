from typing import List, Optional

import torch
from torch import Tensor
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import add_remaining_self_loops, scatter

from retrieval.gnn_modules.GatedRGCNConv import ResidualGatedRGCNConv as GatedRGCNConv

def compute_ghost_node_embeddings_gcn(
    gcn_layer: GCNConv,
    existing_node_embs: Tensor,
    original_edge_index: Tensor,
    ghost_node_initial_embs: List[Tensor],
    ghost_node_connections: List[Tensor],
    original_edge_weight: Optional[Tensor] = None,
) -> Tensor:
    """
    Efficiently computes the output embeddings for new "ghost" nodes after one GCN layer pass.

    This function avoids a full GNN pass over the combined graph. It works by:
    1.  Creating a temporary, augmented graph structure containing the original nodes
        and the new ghost nodes.
    2.  Applying the GCN layer's linear transformation to all node embeddings.
    3.  Calculating the GCN normalization coefficients for the augmented graph.
    4.  Performing message passing only for the new ghost nodes, aggregating features
        from their neighbors and themselves (self-loops).
    5.  Returning the final embeddings for only the ghost nodes.

    Args:
        gcn_layer (GCNConv): The pretrained GCNConv layer to use for computation.
        existing_node_embs (Tensor): A tensor of shape (num_existing_nodes, in_channels)
            containing the input features for the nodes already in the graph.
        original_edge_index (Tensor): The edge_index of the original graph, of shape
            (2, num_original_edges).
        ghost_node_initial_embs (List[Tensor]): A list of initial embedding vectors for
            the new ghost nodes. Each tensor should have shape (in_channels,).
        ghost_node_connections (List[Tensor]): A list where each element is a 1D tensor
            of indices. `ghost_node_connections[i]` contains the indices of the
            existing nodes that have an edge pointing to the i-th ghost node.
        original_edge_weight (Optional[Tensor]): Edge weights for the original graph.
            Defaults to None.

    Returns:
        Tensor: A tensor of shape (num_ghost_nodes, out_channels) containing the
                computed output embeddings for the ghost nodes.
    """
    # --- 1. Input Validation and Setup ---
    if not gcn_layer.normalize:
        raise NotImplementedError(
            "This efficient implementation currently requires 'normalize=True' "
            "on the GCNConv layer, as this is the standard and most complex case."
        )
    if not ghost_node_initial_embs:
        return torch.empty(0, gcn_layer.out_channels, device=existing_node_embs.device)

    device = existing_node_embs.device
    num_existing_nodes = existing_node_embs.shape[0]
    num_ghost_nodes = len(ghost_node_initial_embs)
    num_total_nodes = num_existing_nodes + num_ghost_nodes

    # Stack ghost node embeddings into a single tensor
    ghost_embs_tensor = torch.stack(ghost_node_initial_embs).to(device)
    
    # Combine all node features for the linear transformation step
    all_node_embs = torch.cat([existing_node_embs, ghost_embs_tensor], dim=0)

    # --- 2. Apply Linear Transformation ---
    # This is a single efficient matrix multiplication for all nodes involved.
    h = gcn_layer.lin(all_node_embs)

    # --- 3. Construct Augmented Graph and Normalize ---
    # Create the new edges pointing from existing nodes to ghost nodes
    new_edge_sources = torch.cat(ghost_node_connections).to(device)
    new_edge_dests_list = [
        torch.full_like(conn, fill_value=i + num_existing_nodes)
        for i, conn in enumerate(ghost_node_connections)
    ]
    new_edge_dests = torch.cat(new_edge_dests_list).to(device)
    new_edges = torch.stack([new_edge_sources, new_edge_dests], dim=0)

    # Combine with original edges to form the full edge index for degree calculation
    augmented_edge_index = torch.cat([original_edge_index.to(device), new_edges], dim=1)

    # Replicate the gcn_norm logic for the augmented graph
    fill_value = 2.0 if gcn_layer.improved else 1.0
    
    # Add self-loops to the entire augmented graph structure
    if gcn_layer.add_self_loops:
        edge_index_with_loops, _ = add_remaining_self_loops(
            augmented_edge_index, None, fill_value, num_total_nodes
        )
    else:
        edge_index_with_loops = augmented_edge_index

    # Create edge weights of 1 for all edges for degree calculation
    edge_weight_for_deg = torch.ones(
        (edge_index_with_loops.size(1),), device=device
    )

    # Calculate degrees for all nodes in the augmented graph
    row, col = edge_index_with_loops[0], edge_index_with_loops[1]
    deg = scatter(edge_weight_for_deg, col, dim=0, dim_size=num_total_nodes, reduce='sum')
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    
    # Normalization coefficients for all edges in the augmented graph
    norm_coeffs = deg_inv_sqrt[row] * edge_weight_for_deg * deg_inv_sqrt[col]

    # --- 4. Perform Message Passing and Aggregation (Vectorized) ---
    # Propagate messages along all edges (original, new, and self-loops)
    # The source of the message is the transformed feature vector `h[row]`
    messages = h[row] * norm_coeffs.view(-1, 1)

    # Aggregate messages at the destination nodes `col`
    # This computes the output for ALL nodes in one go, which is very efficient.
    out = scatter(messages, col, dim=0, dim_size=num_total_nodes, reduce='sum')

    # --- 5. Finalize and Return Ghost Node Embeddings ---
    if gcn_layer.bias is not None:
        out = out + gcn_layer.bias
        
    # Extract only the embeddings for the ghost nodes
    ghost_node_outputs = out[num_existing_nodes:]

    return ghost_node_outputs

def compute_ghost_node_embeddings_graphsage(
    sage_layer: SAGEConv,
    existing_node_embs: Tensor,
    ghost_node_initial_embs: List[Tensor],
    ghost_node_connections: List[Tensor],
) -> Tensor:
    """
    Efficiently computes ghost node embeddings for a SAGEConv layer.
    """
    if not ghost_node_initial_embs:
        return torch.empty(0, sage_layer.out_channels, device=existing_node_embs.device)

    device = existing_node_embs.device
    num_ghost_nodes = len(ghost_node_initial_embs)
    ghost_embs_tensor = torch.stack(ghost_node_initial_embs).to(device)

    # --- 1. Aggregate messages (neighbor features) ---
    # `aggr` defaults to 'mean' in SAGEConv, which is what we replicate here.
    source_indices_list = [conn.to(device) for conn in ghost_node_connections if conn.numel() > 0]
    
    if source_indices_list:
        source_indices = torch.cat(source_indices_list)
        dest_indices = torch.cat([
            torch.full_like(conn, i) for i, conn in enumerate(ghost_node_connections) if conn.numel() > 0
        ]).to(device)
        
        source_features = existing_node_embs[source_indices]
        aggregated_messages = scatter(source_features, dest_indices, dim=0, dim_size=num_ghost_nodes, reduce=sage_layer.aggr)
    else:
        # If no neighbors, the aggregated message is a zero vector
        aggregated_messages = torch.zeros(num_ghost_nodes, existing_node_embs.size(1), device=device)

    # --- 2. Apply GraphSAGE update rule ---
    # out = W_l * aggr_neighbors + W_r * self_features
    out = sage_layer.lin_l(aggregated_messages) + sage_layer.lin_r(ghost_embs_tensor)

    if sage_layer.normalize:
        out = torch.nn.functional.normalize(out, p=2, dim=-1)
    
    if sage_layer.bias is not None:
        out = out + sage_layer.bias
        
    return out

from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.nn import RGCNConv
from torch_geometric.utils import scatter

def compute_ghost_node_embeddings_rgcn(
    rgcn_layer: RGCNConv,
    existing_node_embs: Tensor,
    ghost_node_initial_embs: List[Tensor],
    ghost_node_connections: List[Tuple[Tensor, Tensor]],
) -> Tensor:
    """
    Efficiently computes the output embeddings for new "ghost" nodes after one RGCNConv layer pass.

    This function avoids a full GNN pass over the combined graph. It works by:
    1.  Calculating the effective weight matrices for all relations, applying basis
        decomposition if used.
    2.  Gathering the source node features ('existing_node_embs') and the appropriate
        relation weight matrix for each new incoming edge to a ghost node.
    3.  Computing all message vectors in a single batched matrix multiplication.
    4.  If 'mean' aggregation is used, calculating the correct normalization factor
        (1 / |N_r(i)|) for each message.
    5.  Aggregating the messages for each ghost node.
    6.  Applying the final root transformation and bias.

    Args:
        rgcn_layer (RGCNConv): The pretrained RGCNConv layer.
        existing_node_embs (Tensor): A tensor of shape (num_existing_nodes, in_channels)
            containing the input features for the nodes already in the graph.
        ghost_node_initial_embs (List[Tensor]): A list of initial embedding vectors for
            the new ghost nodes. Each tensor should have shape (in_channels,).
        ghost_node_connections (List[Tuple[Tensor, Tensor]]): A list where each element
            corresponds to a ghost node. The element is a tuple of
            (source_indices, edge_types), where `source_indices` are the existing
            nodes pointing to this ghost node, and `edge_types` are the corresponding
            relation types for those edges.

    Returns:
        Tensor: A tensor of shape (num_ghost_nodes, out_channels) containing the
                computed output embeddings for the ghost nodes.
    """
    # --- 1. Input Validation and Setup ---
    if rgcn_layer.num_blocks is not None:
        raise NotImplementedError(
            "Efficient ghost node computation for RGCNConv with 'num_blocks' "
            "regularization is not supported in this implementation."
        )
    if not ghost_node_initial_embs:
        return torch.empty(0, rgcn_layer.out_channels, device=existing_node_embs.device)

    device = existing_node_embs.device
    num_existing_nodes = existing_node_embs.shape[0]
    num_ghost_nodes = len(ghost_node_initial_embs)

    # --- 2. Prepare Augmented Graph Edges ---
    # Unpack connections into three flat tensors: sources, destinations, and types
    new_edge_sources_list = []
    new_edge_dests_list = []
    new_edge_types_list = []

    for i, (sources, types) in enumerate(ghost_node_connections):
        if sources.numel() > 0:
            ghost_node_idx = num_existing_nodes + i
            new_edge_sources_list.append(sources)
            new_edge_dests_list.append(
                torch.full_like(sources, fill_value=ghost_node_idx)
            )
            new_edge_types_list.append(types)

    # If there are no new edges, only the root/bias transformation applies
    if not new_edge_sources_list:
        aggregated_messages = torch.zeros(
            num_ghost_nodes, rgcn_layer.out_channels, device=device
        )
    else:
        new_edge_sources = torch.cat(new_edge_sources_list).to(device)
        new_edge_dests = torch.cat(new_edge_dests_list).to(device)
        new_edge_types = torch.cat(new_edge_types_list).to(device)

        # --- 3. Calculate Effective Weight Matrices ---
        # Handle basis-decomposition regularization if used
        weight = rgcn_layer.weight
        if rgcn_layer.num_bases is not None:
            weight = (rgcn_layer.comp @ weight.view(rgcn_layer.num_bases, -1)).view(
                rgcn_layer.num_relations, rgcn_layer.in_channels_l, rgcn_layer.out_channels
            )

        # --- 4. Compute Messages in a Batched Operation ---
        # Get the source node features for each new edge
        source_node_features = existing_node_embs[new_edge_sources]  # (num_new_edges, in_channels)

        # Get the corresponding weight matrix for each new edge
        edge_weights = weight[new_edge_types]  # (num_new_edges, in_channels, out_channels)

        # Compute messages: message_k = x_j @ W_r
        # Unsqueeze adds a dimension for bmm: (N, C_in) -> (N, 1, C_in)
        # Squeeze removes it: (N, 1, C_out) -> (N, C_out)
        messages = torch.bmm(source_node_features.unsqueeze(1), edge_weights).squeeze(1)

        # --- 5. Aggregate Messages ---
        # Apply normalization for 'mean' aggregation
        if rgcn_layer.aggr == 'mean':
            # This logic replicates the normalization used in PyG's FastRGCNConv
            # 1. Create a one-hot encoding for each edge's relation type
            one_hot_types = torch.nn.functional.one_hot(
                new_edge_types, num_classes=rgcn_layer.num_relations
            ).to(messages.dtype)

            # 2. Sum the one-hot vectors per destination node to get in-degrees for each relation
            # The size needs to cover all nodes in the augmented graph concept
            num_total_nodes = num_existing_nodes + num_ghost_nodes
            degree_matrix = scatter(
                one_hot_types, new_edge_dests, dim=0, dim_size=num_total_nodes, reduce='sum'
            ) # Shape: (num_total_nodes, num_relations)

            # 3. For each edge, look up the degree of its destination node and relation type
            # `degree_matrix[new_edge_dests]` gets the degree vector for each edge's destination
            # `gather` then picks the specific degree for that edge's relation type
            degrees_for_edges = degree_matrix[new_edge_dests]
            degree_of_specific_relation = torch.gather(
                degrees_for_edges, 1, new_edge_types.view(-1, 1)
            )

            # 4. Compute normalization factor and apply it to the messages
            norm = 1.0 / degree_of_specific_relation.clamp_(1.)
            messages = messages * norm

        # Sum the (normalized) messages for each ghost node
        # We subtract num_existing_nodes to map destination indices to [0, num_ghost_nodes-1]
        aggregated_messages = scatter(
            messages,
            new_edge_dests - num_existing_nodes,
            dim=0,
            dim_size=num_ghost_nodes,
            reduce='sum' # aggr is sum because mean normalization was already applied
        )

    # --- 6. Apply Root Transformation and Bias ---
    out = aggregated_messages

    if rgcn_layer.root is not None:
        ghost_embs_tensor = torch.stack(ghost_node_initial_embs).to(device)
        root_update = ghost_embs_tensor @ rgcn_layer.root
        out = out + root_update

    if rgcn_layer.bias is not None:
        out = out + rgcn_layer.bias

    return out

from typing import List, Tuple
import torch
from torch import Tensor
from torch_geometric.nn import GCNConv, RGCNConv, GATConv
from torch_geometric.utils import add_self_loops, softmax, scatter
# ... (your other two functions remain here) ...

import torch.nn.functional as F

def compute_ghost_node_embeddings_gat(
    gat_layer: GATConv,
    existing_node_embs: Tensor,
    ghost_node_initial_embs: List[Tensor],
    ghost_node_connections: List[Tensor],
) -> Tensor:
    """
    Efficiently computes the output embeddings for new "ghost" nodes after one GATConv layer pass.

    This function avoids a full GNN pass over the combined graph. It works by:
    1.  Applying the GAT layer's linear transformation to all involved nodes (existing
        neighbors and the new ghost nodes).
    2.  Constructing a local edge_index containing only the edges pointing to the
        ghost nodes, including self-loops for them.
    3.  Calculating the attention scores (alpha) for these local edges.
    4.  Performing softmax normalization over these scores, grouped by each ghost node.
    5.  Propagating messages along the local edges and aggregating them.
    6.  Applying final head concatenation/averaging, bias, and residual connections.

    Args:
        gat_layer (GATConv): The pretrained GATConv layer.
        existing_node_embs (Tensor): Tensor of shape (num_existing_nodes, in_channels).
        ghost_node_initial_embs (List[Tensor]): List of initial embedding vectors for
            the new ghost nodes, each of shape (in_channels,).
        ghost_node_connections (List[Tensor]): List where each element is a 1D tensor
            of indices of existing nodes pointing to the i-th ghost node.

    Returns:
        Tensor: A tensor of shape (num_ghost_nodes, out_channels * heads) containing
                the computed output embeddings for the ghost nodes.
    """
    # --- 1. Input Validation and Setup ---
    if gat_layer.edge_dim is not None:
        raise NotImplementedError("GATConv with edge_dim is not supported by this function.")
    if isinstance(gat_layer.in_channels, tuple):
        raise NotImplementedError("Bipartite GATConv is not supported by this function.")
    if not ghost_node_initial_embs:
        return torch.empty(0, gat_layer.out_channels, device=existing_node_embs.device)

    device = existing_node_embs.device
    H, C = gat_layer.heads, gat_layer.out_channels
    num_existing_nodes = existing_node_embs.shape[0]
    num_ghost_nodes = len(ghost_node_initial_embs)
    num_total_nodes = num_existing_nodes + num_ghost_nodes

    ghost_embs_tensor = torch.stack(ghost_node_initial_embs).to(device)
    all_node_embs = torch.cat([existing_node_embs, ghost_embs_tensor], dim=0)

    # --- 2. Linear Projection (Step 1 of GAT) ---
    # Apply linear layer to all nodes involved.
    x_proj = gat_layer.lin(all_node_embs).view(-1, H, C)

    # --- 3. Construct Local Edges for Attention ---
    # We only need the edges that point TO the ghost nodes.
    local_edge_sources, local_edge_dests = [], []

    # Add incoming edges from existing nodes
    for i, conn in enumerate(ghost_node_connections):
        if conn.numel() > 0:
            local_edge_sources.append(conn)
            local_edge_dests.append(torch.full_like(conn, num_existing_nodes + i))

    # Add self-loops for ghost nodes if the layer requires it
    if gat_layer.add_self_loops:
        ghost_indices = torch.arange(num_existing_nodes, num_total_nodes, device=device)
        local_edge_sources.append(ghost_indices)
        local_edge_dests.append(ghost_indices)

    if not local_edge_sources: # Only happens if no connections and no self-loops
        out = torch.zeros(num_ghost_nodes, H, C, device=device)
    else:
        local_edge_sources = torch.cat(local_edge_sources)
        local_edge_dests = torch.cat(local_edge_dests)
        local_edge_index = torch.stack([local_edge_sources, local_edge_dests], dim=0)

        # --- 4. Calculate and Normalize Attention Scores (Steps 2 & 3 of GAT) ---
        alpha_src = (x_proj * gat_layer.att_src).sum(dim=-1)
        alpha_dst = (x_proj * gat_layer.att_dst).sum(dim=-1)

        alpha_src_local = alpha_src[local_edge_index[0]]
        alpha_dst_local = alpha_dst[local_edge_index[1]]
        alpha = alpha_src_local + alpha_dst_local

        alpha = F.leaky_relu(alpha, gat_layer.negative_slope)
        # Softmax is computed per destination node (the ghost nodes)
        alpha = softmax(alpha, local_edge_index[1], num_nodes=num_total_nodes)
        alpha = F.dropout(alpha, p=gat_layer.dropout, training=gat_layer.training)

        # --- 5. Message Passing and Aggregation (Steps 4 & 5 of GAT) ---
        # Propagate messages from source nodes (existing + ghost self-loops)
        x_j = x_proj[local_edge_index[0]]
        messages = alpha.unsqueeze(-1) * x_j

        # Aggregate messages at the destination ghost nodes
        # The scatter size must cover all nodes conceptually, but we only care about the latter part
        aggregated = scatter(messages, local_edge_index[1], dim=0, dim_size=num_total_nodes, reduce='add')
        out = aggregated[num_existing_nodes:] # Slice to get only ghost node results

    # --- 6. Final Processing (Step 6 of GAT) ---
    if gat_layer.concat:
        out = out.view(-1, H * C)
    else:
        out = out.mean(dim=1)

    # Add residual connection if applicable
    if gat_layer.residual:
        res = gat_layer.res(ghost_embs_tensor)
        out = out + res

    if gat_layer.bias is not None:
        out = out + gat_layer.bias

    return out

from typing import List, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv, RGCNConv, GATConv, RGATConv # Add RGATConv
from torch_geometric.utils import softmax, scatter
# ... (your other three functions remain here) ...

def compute_ghost_node_embeddings_rgat(
    rgat_layer: RGATConv,
    existing_node_embs: Tensor,
    ghost_node_initial_embs: List[Tensor],
    ghost_node_connections: List[Tuple[Tensor, Tensor]],
) -> Tensor:
    """
    Efficiently computes the output embeddings for new "ghost" nodes after one RGATConv layer pass.

    This function supports the core features of RGATConv, including different attention
    mechanisms and modes, heads, and basis regularization.
    """
    # --- 1. Input Validation and Setup ---
    # (No changes here)
    if rgat_layer.num_blocks is not None or rgat_layer.mod is not None or rgat_layer.edge_dim is not None:
        raise NotImplementedError(
            "RGATConv with 'num_blocks', 'mod', or 'edge_dim' is not supported by this function."
        )
    if not ghost_node_initial_embs:
        return torch.empty(0, rgat_layer.out_channels, device=existing_node_embs.device)

    device = existing_node_embs.device
    num_existing_nodes = existing_node_embs.shape[0]
    num_ghost_nodes = len(ghost_node_initial_embs)
    num_total_nodes = num_existing_nodes + num_ghost_nodes

    ghost_embs_tensor = torch.stack(ghost_node_initial_embs).to(device)
    all_node_embs = torch.cat([existing_node_embs, ghost_embs_tensor], dim=0)

    # --- 2. Construct Local Edges for Attention ---
    # (No changes here)
    local_sources, local_dests, local_types = [], [], []
    for i, (sources, types) in enumerate(ghost_node_connections):
        if sources.numel() > 0:
            local_sources.append(sources)
            local_dests.append(torch.full_like(sources, num_existing_nodes + i))
            local_types.append(types)

    if not local_sources:
        out_shape = rgat_layer.heads * rgat_layer.dim * rgat_layer.out_channels if rgat_layer.concat else rgat_layer.dim * rgat_layer.out_channels
        out = torch.zeros(num_ghost_nodes, out_shape, device=device)
        if rgat_layer.bias is not None:
            out = out + rgat_layer.bias
        return out

    local_sources = torch.cat(local_sources)
    local_dests = torch.cat(local_dests)
    local_types = torch.cat(local_types)

    # --- 3. Replicate the `message` method logic on the local subgraph ---
    x_i = all_node_embs[local_dests]
    x_j = all_node_embs[local_sources]

    # <<< START OF CORRECTION >>>
    # Calculate effective relation weights (handles basis-decomposition)
    if rgat_layer.num_bases is not None:
        # If using bases, construct the effective weight matrix from `att` and `basis`
        w = (rgat_layer.att @ rgat_layer.basis.view(rgat_layer.num_bases, -1)).view(
            rgat_layer.num_relations, rgat_layer.in_channels, rgat_layer.heads * rgat_layer.out_channels
        )
    else:
        # Otherwise, the weight matrix is stored directly in `weight`
        w = rgat_layer.weight
    # <<< END OF CORRECTION >>>

    w = w[local_types]

    # Project source and destination features
    out_i = torch.bmm(x_i.unsqueeze(1), w).squeeze(1)
    out_j = torch.bmm(x_j.unsqueeze(1), w).squeeze(1)

    # (The rest of the function remains the same)
    q_i = torch.matmul(out_i, rgat_layer.q)
    k_j = torch.matmul(out_j, rgat_layer.k)
    
    if rgat_layer.attention_mode == "additive-self-attention":
        alpha = F.leaky_relu(q_i + k_j, rgat_layer.negative_slope)
    else:
        alpha = q_i * k_j

    if rgat_layer.attention_mechanism == "across-relation":
        alpha = softmax(alpha, local_dests, num_nodes=num_total_nodes)
    else:
        alpha_out = torch.zeros_like(alpha)
        for r in range(rgat_layer.num_relations):
            mask = local_types == r
            if mask.any():
                alpha_out[mask] = softmax(alpha[mask], local_dests[mask], num_nodes=num_total_nodes)
        alpha = alpha_out

    alpha = F.dropout(alpha, p=rgat_layer.dropout, training=rgat_layer.training)

    if rgat_layer.attention_mode == "additive-self-attention":
        messages = alpha.view(-1, rgat_layer.heads, 1) * out_j.view(-1, rgat_layer.heads, rgat_layer.out_channels)
    else:
        messages = alpha.view(-1, rgat_layer.heads, rgat_layer.dim, 1) * out_j.view(-1, rgat_layer.heads, 1, rgat_layer.out_channels)

    local_dest_indices = local_dests - num_existing_nodes
    aggr_out = scatter(messages, local_dest_indices, dim=0, dim_size=num_ghost_nodes, reduce='add')

    if rgat_layer.attention_mode == "additive-self-attention":
        if rgat_layer.concat:
            aggr_out = aggr_out.view(-1, rgat_layer.heads * rgat_layer.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)
    else:
        if rgat_layer.concat:
            aggr_out = aggr_out.view(-1, rgat_layer.heads * rgat_layer.dim * rgat_layer.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1).view(-1, rgat_layer.dim * rgat_layer.out_channels)

    if rgat_layer.bias is not None:
        aggr_out = aggr_out + rgat_layer.bias

    return aggr_out

from typing import List, Tuple, Callable
import torch
import torch.nn.functional as F
from torch import Tensor
# Add GINConv and GINEConv to the import list
from torch_geometric.nn import GCNConv, RGCNConv, GATConv, RGATConv, GINConv, GINEConv
from torch_geometric.utils import softmax, scatter
# ... (your other four functions remain here) ...

def compute_ghost_node_embeddings_gin(
    gin_layer: GINConv,
    existing_node_embs: Tensor,
    ghost_node_initial_embs: List[Tensor],
    ghost_node_connections: List[Tensor],
) -> Tensor:
    """
    Efficiently computes the output embeddings for new "ghost" nodes after one GINConv layer pass.

    This function replicates the GIN logic by first aggregating neighbor features, adding
    the scaled central node feature, and finally passing the result through the layer's MLP.

    Args:
        gin_layer (GINConv): The pretrained GINConv layer.
        existing_node_embs (Tensor): Tensor of shape (num_existing_nodes, in_channels).
        ghost_node_initial_embs (List[Tensor]): List of initial embeddings for the new
            ghost nodes, each of shape (in_channels,).
        ghost_node_connections (List[Tensor]): List where each element is a 1D tensor
            of indices of existing nodes pointing to the i-th ghost node.

    Returns:
        Tensor: A tensor of shape (num_ghost_nodes, out_channels) containing the
                computed output embeddings for the ghost nodes.
    """
    if not ghost_node_initial_embs:
        # Get out_channels from the nn module
        if hasattr(gin_layer.nn, 'out_features'):
            out_channels = gin_layer.nn.out_features
        elif hasattr(gin_layer.nn, 'out_channels'):
            out_channels = gin_layer.nn.out_channels
        else: # Fallback for Sequential
            last_layer = list(gin_layer.nn.modules())[-1]
            out_channels = last_layer.out_features
        return torch.empty(0, out_channels, device=existing_node_embs.device)

    device = existing_node_embs.device
    num_ghost_nodes = len(ghost_node_initial_embs)
    ghost_embs_tensor = torch.stack(ghost_node_initial_embs).to(device)

    # --- 1. Aggregate messages (neighbor features) ---
    source_indices = torch.cat(ghost_node_connections).to(device)
    dest_indices = torch.cat([
        torch.full_like(conn, i) for i, conn in enumerate(ghost_node_connections)
    ]).to(device)

    # If there are no incoming edges, aggregation result is zero
    if source_indices.numel() > 0:
        source_features = existing_node_embs[source_indices]
        aggregated_messages = scatter(source_features, dest_indices, dim=0, dim_size=num_ghost_nodes, reduce='add')
    else:
        aggregated_messages = torch.zeros_like(ghost_embs_tensor)

    # --- 2. Add scaled central node features ---
    # This is the step: out = aggregated + (1 + eps) * x_i
    pre_mlp_embs = aggregated_messages + (1 + gin_layer.eps) * ghost_embs_tensor

    # --- 3. Apply the final MLP (`nn`) ---
    return gin_layer.nn(pre_mlp_embs)


def compute_ghost_node_embeddings_gine(
    gine_layer: GINEConv,
    existing_node_embs: Tensor,
    ghost_node_initial_embs: List[Tensor],
    ghost_node_connections: List[Tuple[Tensor, Tensor]],
) -> Tensor:
    """
    Efficiently computes ghost node embeddings for a GINEConv layer, which includes edge attributes.

    Args:
        gine_layer (GINEConv): The pretrained GINEConv layer.
        existing_node_embs (Tensor): Tensor of shape (num_existing_nodes, in_channels).
        ghost_node_initial_embs (List[Tensor]): List of initial embeddings for the new
            ghost nodes, each of shape (in_channels,).
        ghost_node_connections (List[Tuple[Tensor, Tensor]]): List where each element is
            a tuple of (source_indices, edge_attrs) for a ghost node.

    Returns:
        Tensor: A tensor of shape (num_ghost_nodes, out_channels).
    """
    if not ghost_node_initial_embs:
        if hasattr(gine_layer.nn, 'out_features'):
            out_channels = gine_layer.nn.out_features
        else: # Fallback for Sequential
            last_layer = list(gine_layer.nn.modules())[-1]
            out_channels = last_layer.out_features
        return torch.empty(0, out_channels, device=existing_node_embs.device)

    device = existing_node_embs.device
    num_ghost_nodes = len(ghost_node_initial_embs)
    ghost_embs_tensor = torch.stack(ghost_node_initial_embs).to(device)

    # --- 1. Compute and aggregate messages (neighbor features + edge attributes) ---
    all_sources, all_attrs, all_dests = [], [], []
    for i, (sources, attrs) in enumerate(ghost_node_connections):
        if sources.numel() > 0:
            all_sources.append(sources)
            all_attrs.append(attrs)
            all_dests.append(torch.full_like(sources, i))
    
    # If there are no incoming edges, aggregation result is zero
    if all_sources:
        source_indices = torch.cat(all_sources).to(device)
        edge_attrs = torch.cat(all_attrs).to(device)
        dest_indices = torch.cat(all_dests).to(device)
        
        source_features = existing_node_embs[source_indices]

        # Replicate the GINE message function: message = (x_j + edge_attr).relu()
        if gine_layer.lin is not None:
            edge_attrs = gine_layer.lin(edge_attrs)
        
        messages = (source_features + edge_attrs).relu()
        aggregated_messages = scatter(messages, dest_indices, dim=0, dim_size=num_ghost_nodes, reduce='add')
    else:
        aggregated_messages = torch.zeros_like(ghost_embs_tensor)

    # --- 2. Add scaled central node features ---
    pre_mlp_embs = aggregated_messages + (1 + gine_layer.eps) * ghost_embs_tensor

    # --- 3. Apply the final MLP (`nn`) ---
    return gine_layer.nn(pre_mlp_embs)

def compute_ghost_node_embeddings_gated_rgcn(
    gated_rgcn_layer: GatedRGCNConv,
    existing_node_embs: Tensor,
    ghost_node_initial_embs: List[Tensor],
    ghost_node_connections: List[Tuple[Tensor, Tensor]],
) -> Tensor:
    """
    Efficiently computes ghost node embeddings for a ResidualGatedRGCNConv layer.
    """
    # GatedRGCNConv disables root weight and bias, so the rgcn util effectively
    # computes only the aggregated messages from neighbors.
    aggregated_messages = compute_ghost_node_embeddings_rgcn(
        gated_rgcn_layer,
        existing_node_embs,
        # Pass dummy embeddings as root connection is not used.
        [torch.zeros_like(e) for e in ghost_node_initial_embs],
        ghost_node_connections,
    )

    ghost_embs_tensor = torch.stack(ghost_node_initial_embs).to(
        existing_node_embs.device
    )

    # Apply the GRU-like gating mechanism from the layer
    combined = torch.cat([ghost_embs_tensor, aggregated_messages], dim=-1)
    update_gate = torch.sigmoid(gated_rgcn_layer.gate_nn(combined))
    update_candidate = torch.tanh(gated_rgcn_layer.update_nn(combined))
    out = (1 - update_gate) * ghost_embs_tensor + update_gate * update_candidate

    return out