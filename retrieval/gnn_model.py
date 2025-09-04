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
from torch_geometric.nn import RGCNConv, GCNConv, GATConv, RGATConv

from common import get_optimizers, load_checkpoint

class GNNRetriever(pl.LightningModule):
    def __init__(
        self,
        feature_size: int,
        num_layers: int,
        gnn_layer_type: str = "rgcn",  # Options: "gcn", "rgcn", "gat", "rgat"
        num_relations: int = 3,  # Number of edge types (signature_lctx, signature_goal, proof)
        use_edge_attr: bool = True,  # Whether to use edge attributes
        gat_heads: int = 8,  # Number of attention heads for GAT/RGAT
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        assert num_layers >= 1, "Number of GNN layers must be at least 1."
        
        self.gnn_layer_type = gnn_layer_type.lower()
        self.use_edge_attr = use_edge_attr
        self.num_relations = num_relations
        self.gat_heads = gat_heads

        # Build GNN layers based on the specified type
        self.layers = nn.ModuleList()
        
        if self.gnn_layer_type == "gcn":
            for _ in range(num_layers):
                self.layers.append(GCNConv(feature_size, feature_size))
        elif self.gnn_layer_type == "rgcn":
            for _ in range(num_layers):
                self.layers.append(RGCNConv(feature_size, feature_size, num_relations=num_relations))
        elif self.gnn_layer_type == "gat":
            for _ in range(num_layers):
                # GAT typically concatenates or averages multi-head outputs
                self.layers.append(GATConv(feature_size, feature_size // gat_heads, heads=gat_heads, concat=False))
        elif self.gnn_layer_type == "rgat":
            for _ in range(num_layers):
                self.layers.append(RGATConv(feature_size, feature_size // gat_heads, heads=gat_heads, num_relations=num_relations, concat=False))
        else:
            raise ValueError(f"Unsupported GNN layer type: {gnn_layer_type}. Supported types: gcn, rgcn, gat, rgat")

        self.dropout_p = 0.5

    def _gnn_forward_pass(self, x: torch.FloatTensor, edge_index: torch.LongTensor, edge_attr: Optional[torch.LongTensor] = None) -> torch.FloatTensor:
        """
        Apply GNN layers with ReLU activation and dropout between layers.
        
        Args:
            x: Input node features
            edge_index: Edge connectivity information  
            edge_attr: Edge attributes (edge types)
            
        Returns:
            Output node features after passing through all GNN layers
        """
        for i, layer in enumerate(self.layers):
            # Apply layer based on type
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
                    
            # Apply activation and dropout (except for the last layer)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_p, training=self.training)
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
        neighbor_indices: List[torch.LongTensor]
    ) -> Tuple[torch.FloatTensor, torch.LongTensor, Optional[torch.LongTensor]]:
        """
        Create augmented graph by adding ghost nodes and their connections.
        
        Args:
            node_features: Original node features
            context_features: Context features (will become ghost nodes)
            edge_index: Original edge connectivity
            edge_attr: Original edge attributes
            neighbor_indices: Lists of neighbor indices for each context
            
        Returns:
            Tuple of (augmented_features, augmented_edge_index, augmented_edge_attr)
        """
        target_device, target_dtype = self._get_target_device_and_dtype()
        num_premises = node_features.shape[0]
        
        # --- 1. Augment Node Features ---
        augmented_features = torch.cat(
            [node_features.to(target_device, target_dtype), context_features.to(target_device, target_dtype)], 
            dim=0
        )

        # --- 2. Augment Edges ---
        new_edges_src = []
        new_edges_dst = []
        new_edge_attrs = []
        
        for i, indices in enumerate(neighbor_indices):
            ghost_node_idx = num_premises + i
            # Create DIRECTED edges from neighbors to their corresponding ghost nodes.
            for neighbor_idx in indices:
                new_edges_src.append(neighbor_idx.item())
                new_edges_dst.append(ghost_node_idx)
                # For new edges to ghost nodes, we can use a special edge type (e.g., type 0)
                # or we could add a new edge type specifically for context connections
                new_edge_attrs.append(TODO we have to pass stuff for this to this function)
        
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
        neighbor_indices: List[torch.LongTensor],
        pos_premise_indices: torch.LongTensor,
        neg_premises_indices: List[torch.LongTensor],
        label: torch.FloatTensor,
        edge_attr: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        num_premises = node_features.shape[0]

        # --- 1. Create Augmented Graph ---
        augmented_features, augmented_edge_index, augmented_edge_attr = self._create_augmented_graph(
            node_features, context_features, edge_index, edge_attr, neighbor_indices
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
            batch["neighbor_indices"],
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
        batch_neighbor_indices: List[torch.LongTensor],
        # We need the initial premise embeddings and edge index for the combined pass
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
            initial_node_features, initial_context_embs, edge_index, edge_attr, batch_neighbor_indices
        )

        # Single GNN pass
        x = self._gnn_forward_pass(augmented_features, augmented_edge_index, augmented_edge_attr)

        # Extract and normalize final embeddings
        final_premise_embs, final_context_embs = self._extract_and_normalize_embeddings(x, num_premises)
        
        return final_context_embs, final_premise_embs