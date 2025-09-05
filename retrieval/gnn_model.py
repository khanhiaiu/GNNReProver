import torch

# Monkeypatch torch.load before Lightning/DeepSpeed calls it
_orig_load = torch.load
def patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)

torch.load = patched_load


import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, List, Tuple, Optional
from torch_geometric.nn import RGCNConv, GCNConv, GATConv, RGATConv, GINConv

from common import _get_edge_type_name_from_tag, get_optimizers, load_checkpoint

class GNNRetriever(pl.LightningModule):
    def __init__(
        self,
        feature_size: int,
        num_layers: int,
        graph_dependencies: Dict[str, Any],         # <-- ADD THIS
        context_neighbor_verbosity: str, 
        edge_type_to_id: Dict[str, int],
        lr : float,
        warmup_steps : int,
        gnn_layer_type: str = "rgcn",  # Options: "gcn", "rgcn", "gat", "rgat", "gin"
        hidden_size: Optional[int] = None,  # Hidden layer size (default: same as feature_size)
        num_relations: int = 3,  # Number of edge types (signature_lctx, signature_goal, proof)
        use_edge_attr: bool = True,  # Whether to use edge attributes
        gat_heads: int = 8,  # Number of attention heads for GAT/RGAT
        use_residual: bool = True,  # Whether to use residual/skip connections
        dropout_p: float = 0.5,  # Dropout probability
        norm_type: str = "layer",  # Normalization type: "none", "batch", "layer", "instance", "group"
        use_initial_projection: bool = True,  # Whether to apply initial projection to embeddings
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        assert num_layers >= 1, "Number of GNN layers must be at least 1."
        
        self.feature_size = feature_size
        self.hidden_size = hidden_size if hidden_size is not None else feature_size
        self.num_layers = num_layers
        self.gnn_layer_type = gnn_layer_type.lower()
        self.use_edge_attr = use_edge_attr
        self.graph_dependencies_config = graph_dependencies
        self.context_neighbor_verbosity = context_neighbor_verbosity
        self.num_relations = num_relations
        self.gat_heads = gat_heads
        self.use_residual = use_residual
        self.dropout_p = dropout_p
        self.norm_type = norm_type.lower() if norm_type else "none"
        self.use_initial_projection = use_initial_projection

        self.edge_type_to_id = edge_type_to_id

        # Initial projection layer (optional)
        if self.use_initial_projection:
            self.initial_projection = nn.Linear(feature_size, self.hidden_size)
        else:
            self.initial_projection = None

        # Residual projection layer (if hidden_size != feature_size)
        if self.use_residual and self.hidden_size != feature_size:
            self.residual_projection = nn.Linear(feature_size, self.hidden_size)
        else:
            self.residual_projection = None

        # Final projection layer (to get back to feature_size if needed)
        if self.hidden_size != feature_size:
            self.final_projection = nn.Linear(self.hidden_size, feature_size)
        else:
            self.final_projection = None

        # Helper method to create normalization layers
        def create_norm_layer(norm_type: str, num_features: int) -> Optional[nn.Module]:
            """Create normalization layer based on type."""
            if norm_type == "batch":
                return nn.BatchNorm1d(num_features)
            elif norm_type == "layer":
                return nn.LayerNorm(num_features)
            elif norm_type == "instance":
                return nn.InstanceNorm1d(num_features)
            elif norm_type == "group":
                # Use 8 groups by default, ensure it divides num_features
                num_groups = min(8, num_features)
                while num_features % num_groups != 0 and num_groups > 1:
                    num_groups -= 1
                return nn.GroupNorm(num_groups, num_features)
            else:  # "none" or invalid
                return None

        # Build GNN layers based on the specified type
        self.layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList() if self.norm_type != "none" else None
        
        # Determine input/output sizes for layers
        input_size = self.hidden_size if use_initial_projection else feature_size
        
        if self.gnn_layer_type == "gcn":
            for i in range(num_layers):
                self.layers.append(GCNConv(input_size, self.hidden_size))
                if self.norm_type != "none":
                    norm_layer = create_norm_layer(self.norm_type, self.hidden_size)
                    self.norm_layers.append(norm_layer)
                input_size = self.hidden_size
        elif self.gnn_layer_type == "rgcn":
            for i in range(num_layers):
                self.layers.append(RGCNConv(input_size, self.hidden_size, num_relations=num_relations))
                if self.norm_type != "none":
                    norm_layer = create_norm_layer(self.norm_type, self.hidden_size)
                    self.norm_layers.append(norm_layer)
                input_size = self.hidden_size
        elif self.gnn_layer_type == "gat":
            for i in range(num_layers):
                # GAT: When concat=False, outputs are averaged across heads
                # So we need each head to output hidden_size dimensions to get hidden_size output
                self.layers.append(GATConv(input_size, self.hidden_size, heads=gat_heads, concat=False))
                if self.norm_type != "none":
                    norm_layer = create_norm_layer(self.norm_type, self.hidden_size)
                    self.norm_layers.append(norm_layer)
                input_size = self.hidden_size
        elif self.gnn_layer_type == "rgat":
            for i in range(num_layers):
                # RGAT: Same logic as GAT - each head outputs hidden_size dimensions
                self.layers.append(RGATConv(input_size, self.hidden_size, heads=gat_heads, num_relations=num_relations, concat=False))
                if self.norm_type != "none":
                    norm_layer = create_norm_layer(self.norm_type, self.hidden_size)
                    self.norm_layers.append(norm_layer)
                input_size = self.hidden_size
        elif self.gnn_layer_type == "gin":
            for i in range(num_layers):
                # GIN uses an MLP for the neural network component
                mlp = nn.Sequential(
                    nn.Linear(input_size, self.hidden_size * 2),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size * 2, self.hidden_size)
                )
                self.layers.append(GINConv(mlp))
                if self.norm_type != "none":
                    norm_layer = create_norm_layer(self.norm_type, self.hidden_size)
                    self.norm_layers.append(norm_layer)
                input_size = self.hidden_size
        else:
            raise ValueError(f"Unsupported GNN layer type: {gnn_layer_type}. Supported types: gcn, rgcn, gat, rgat, gin")

    def _gnn_forward_pass(self, x: torch.FloatTensor, edge_index: torch.LongTensor, edge_attr: Optional[torch.LongTensor] = None) -> torch.FloatTensor:
        """
        Apply GNN layers with optional initial projection, residual connections, 
        batch normalization, ReLU activation and dropout between layers.
        
        Args:
            x: Input node features
            edge_index: Edge connectivity information  
            edge_attr: Edge attributes (edge types)
            
        Returns:
            Output node features after passing through all GNN layers
        """
        # Store original input for residual connection
        x_orig = x
        
        # Apply initial projection if enabled
        if self.initial_projection is not None:
            x = self.initial_projection(x)
        
        for i, layer in enumerate(self.layers):
            # Store input for potential residual connection
            x_residual = x
            
            # Apply GNN layer based on type
            if self.gnn_layer_type == "gcn":
                x = layer(x, edge_index)
            elif self.gnn_layer_type == "rgcn":
                if self.use_edge_attr and edge_attr is not None:
                    x = layer(x, edge_index, edge_attr)
                else:
                    # If no edge attributes, use edge type 0 for all edges
                    default_edge_attr = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)
                    x = layer(x, edge_index, default_edge_attr)
            elif self.gnn_layer_type == "gat":
                x = layer(x, edge_index)
            elif self.gnn_layer_type == "rgat":
                if self.use_edge_attr and edge_attr is not None:
                    x = layer(x, edge_index, edge_attr)
                else:
                    # If no edge attributes, use edge type 0 for all edges
                    default_edge_attr = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)
                    x = layer(x, edge_index, default_edge_attr)
            elif self.gnn_layer_type == "gin":
                x = layer(x, edge_index)
            
            # Apply normalization if enabled
            if self.norm_layers is not None and i < len(self.norm_layers):
                x = self.norm_layers[i](x)
            
            # Apply residual connection if enabled and shapes match
            if self.use_residual and x.shape == x_residual.shape:
                x = x + x_residual
            elif self.use_residual and i == 0 and self.residual_projection is not None:
                # For the first layer, project original input if shapes don't match
                x = x + self.residual_projection(x_orig)
                    
            # Apply activation and dropout (except for the last layer)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        # Apply final projection if needed to get back to feature_size
        if self.final_projection is not None:
            x = self.final_projection(x)
        
        return x

    def _get_target_device_and_dtype(self) -> Tuple[torch.device, torch.dtype]:
        """Get target device and dtype from the first layer."""
        first_layer = self.layers[0]
        if hasattr(first_layer, 'weight'):
            return first_layer.weight.device, first_layer.weight.dtype
        elif hasattr(first_layer, 'lin_l') and hasattr(first_layer.lin_l, 'weight'):
            return first_layer.lin_l.weight.device, first_layer.lin_l.weight.dtype
        else:
            # Fallback to module parameters
            param = next(first_layer.parameters())
            return param.device, param.dtype

    def _create_augmented_graph(
        self, 
        node_features: torch.FloatTensor, 
        context_features: torch.FloatTensor, 
        edge_index: torch.LongTensor,
        edge_attr: Optional[torch.LongTensor],
        lctx_neighbor_indices: List[torch.LongTensor],
        goal_neighbor_indices: List[torch.LongTensor],
    ) -> Tuple[torch.FloatTensor, torch.LongTensor, Optional[torch.LongTensor]]:
        """
        Create augmented graph by adding ghost nodes and their connections.
        """
        target_device, target_dtype = self._get_target_device_and_dtype()
        num_premises = node_features.shape[0]
        
        # --- 1. Augment Node Features (no change) ---
        augmented_features = torch.cat(
            [node_features.to(target_device, target_dtype), context_features.to(target_device, target_dtype)], 
            dim=0
        )

        # --- 2. Augment Edges using the refactored logic ---
        new_edges_src = []
        new_edges_dst = []
        new_edge_attrs = []
        
        # Dynamically create tags based on the configured verbosity
        lctx_tag = f"signature_{self.context_neighbor_verbosity}_lctx"
        goal_tag = f"signature_{self.context_neighbor_verbosity}_goal"

        # Determine edge types using the common helper function
        lctx_edge_name = _get_edge_type_name_from_tag(lctx_tag, self.graph_dependencies_config)
        goal_edge_name = _get_edge_type_name_from_tag(goal_tag, self.graph_dependencies_config)

        if lctx_edge_name and lctx_edge_name in self.edge_type_to_id:
            lctx_edge_type_id = self.edge_type_to_id[lctx_edge_name]
            for i, indices in enumerate(lctx_neighbor_indices):
                ghost_node_idx = num_premises + i
                for neighbor_idx in indices:
                    new_edges_src.append(neighbor_idx.item())
                    new_edges_dst.append(ghost_node_idx)
                    new_edge_attrs.append(lctx_edge_type_id)

        if goal_edge_name and goal_edge_name in self.edge_type_to_id:
            goal_edge_type_id = self.edge_type_to_id[goal_edge_name]
            for i, indices in enumerate(goal_neighbor_indices):
                ghost_node_idx = num_premises + i
                for neighbor_idx in indices:
                    new_edges_src.append(neighbor_idx.item())
                    new_edges_dst.append(ghost_node_idx)
                    new_edge_attrs.append(goal_edge_type_id)

        # The rest of the function remains the same...
        if new_edges_src:
            new_edges = torch.tensor([new_edges_src, new_edges_dst], dtype=torch.long, device=target_device)
            augmented_edge_index = torch.cat([edge_index.to(target_device), new_edges], dim=1)
            
            if self.use_edge_attr and edge_attr is not None:
                new_edge_attrs_tensor = torch.tensor(new_edge_attrs, dtype=torch.long, device=target_device)
                augmented_edge_attr = torch.cat([edge_attr.to(target_device), new_edge_attrs_tensor])
            else:
                augmented_edge_attr = None
        else:
            augmented_edge_index = edge_index.to(target_device)
            augmented_edge_attr = edge_attr.to(target_device) if edge_attr is not None else None
            
        return augmented_features, augmented_edge_index, augmented_edge_attr

    def _extract_and_normalize_embeddings(
        self, 
        x: torch.FloatTensor, 
        num_premises: int
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Extract premise and context embeddings from combined tensor and normalize them.
        
        Args:
            x: Combined tensor containing both premise and context embeddings
            num_premises: Number of premise nodes (used to split the tensor)
            
        Returns:
            Tuple of (normalized_premise_embeddings, normalized_context_embeddings)
        """
        final_premise_embs = F.normalize(x[:num_premises], p=2, dim=1)
        final_context_embs = F.normalize(x[num_premises:], p=2, dim=1)
        return final_premise_embs, final_context_embs

    @torch.no_grad()
    def forward_embeddings(self, x: torch.FloatTensor, edge_index: torch.LongTensor, edge_attr: Optional[torch.LongTensor] = None) -> torch.FloatTensor:
        """
        Public method to apply GNN layers to embeddings. Used for external inference.
        This method disables training mode for consistent inference behavior.
        
        Args:
            x: Input node features
            edge_index: Edge connectivity information
            edge_attr: Edge attributes (edge types)
            
        Returns:
            Output node features after passing through all GNN layers
        """
        was_training = self.training
        self.eval()  # Ensure we're in eval mode
        try:
            return self._gnn_forward_pass(x, edge_index, edge_attr)
        finally:
            self.train(was_training)  # Restore original training mode

    @classmethod
    def load(cls, ckpt_path: str, device, freeze: bool) -> "GNNRetriever":
        return load_checkpoint(cls, ckpt_path, device, freeze)

    def forward(
        self,
        node_features: torch.FloatTensor,
        edge_index: torch.LongTensor,
        context_features: torch.FloatTensor,
        lctx_neighbor_indices: List[torch.LongTensor],
        goal_neighbor_indices: List[torch.LongTensor],
        pos_premise_indices: torch.LongTensor,
        neg_premises_indices: List[torch.LongTensor],
        label: torch.FloatTensor,
        edge_attr: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        num_premises = node_features.shape[0]

        # --- 1. Create Augmented Graph ---
        augmented_features, augmented_edge_index, augmented_edge_attr = self._create_augmented_graph(
            node_features, context_features, edge_index, edge_attr, lctx_neighbor_indices, goal_neighbor_indices
        )
            
        # --- 2. Single GNN Pass on Augmented Graph ---
        x = self._gnn_forward_pass(augmented_features, augmented_edge_index, augmented_edge_attr)
        
        # --- 3. Extract and Normalize Final Embeddings ---
        final_premise_embs, final_context_embs = self._extract_and_normalize_embeddings(x, num_premises)

        # --- 4. Contrastive Loss Calculation ---
        pos_premise_emb = final_premise_embs[pos_premise_indices]
        neg_premise_embs_flat = [final_premise_embs[neg_idxs] for neg_idxs in neg_premises_indices]
        
        all_premise_embs = torch.cat([pos_premise_emb] + neg_premise_embs_flat, dim=0)

        similarity = torch.mm(final_context_embs, all_premise_embs.t())
        assert -1.001 <= similarity.min() <= similarity.max() <= 1.001, f"Got {similarity.min()} and {similarity.max()}"
        loss = F.mse_loss(similarity, label)
        return loss

    def training_step(self, batch: Dict[str, Any], _) -> torch.Tensor:
        loss = self(
            batch["node_features"],
            batch["edge_index"],
            batch["context_features"],
            batch["lctx_neighbor_indices"],
            batch["goal_neighbor_indices"],
            batch["pos_premise_indices"],
            batch["neg_premises_indices"],
            batch["label"],
            batch.get("edge_attr", None),  # Handle optional edge attributes
        )
        self.log(
            "loss_train",
            loss,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch["context_features"]),
        )
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        return get_optimizers(
            self.parameters(), self.trainer, self.hparams.lr, self.hparams.warmup_steps
        )
    
    @torch.no_grad()
    def get_dynamic_context_embedding(
        self,
        initial_context_embs: torch.FloatTensor,
        batch_lctx_neighbor_indices: List[torch.LongTensor],
        batch_goal_neighbor_indices: List[torch.LongTensor],
        initial_node_features: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_attr: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Computes final context and premise embeddings by adding ghost nodes to the graph.
        This method is intended for inference.
        """
        self.eval()
        num_premises = initial_node_features.shape[0]

        # Create augmented graph with ghost nodes
        augmented_features, augmented_edge_index, augmented_edge_attr = self._create_augmented_graph(
            initial_node_features, initial_context_embs, edge_index, edge_attr, batch_lctx_neighbor_indices, batch_goal_neighbor_indices
        )

        # Single GNN pass
        x = self._gnn_forward_pass(augmented_features, augmented_edge_index, augmented_edge_attr)

        # Extract and normalize final embeddings
        final_premise_embs, final_context_embs = self._extract_and_normalize_embeddings(x, num_premises)
        
        return final_context_embs, final_premise_embs