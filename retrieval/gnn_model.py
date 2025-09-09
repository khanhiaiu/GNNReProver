import torch

# Monkeypatch torch.load before Lightning/DeepSpeed calls it
_orig_load = torch.load
def patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)

torch.load = patched_load

torch.set_float32_matmul_precision("medium")


import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, List, Tuple, Optional
from torch_geometric.nn import RGCNConv, GCNConv, GATConv, RGATConv, GINConv, SAGEConv

from common import _get_edge_type_name_from_tag, get_optimizers, load_checkpoint

from loguru import logger

from retrieval.gnn_utils import (
        compute_ghost_node_embeddings_gcn,
        compute_ghost_node_embeddings_rgcn,
        compute_ghost_node_embeddings_gat,
        compute_ghost_node_embeddings_rgat,
        compute_ghost_node_embeddings_gin,
        compute_ghost_node_embeddings_graphsage,
    )


class GNNRetriever(pl.LightningModule):
    def __init__(
        self,
        feature_size: int,
        num_layers: int,
        graph_dependencies_config: Dict[str, Any], 
        edge_type_to_id: Dict[str, int],
        lr : float,
        warmup_steps : int,
        gnn_layer_type: str = "rgcn",
        hidden_size: Optional[int] = None,
        num_relations: int = 3,
        use_edge_attr: bool = True,
        gat_heads: int = 8,
        use_residual: bool = True,
        dropout_p: float = 0.5,
        edge_dropout_p: float = 0.1,
        lm_mask_p: float = 0.5,
        norm_type: str = "layer",
        use_initial_projection: bool = True,
        postprocess_gnn_embeddings: str = "gnn",
        num_logs_per_epoch: int = 5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        assert num_layers >= 1, "Number of GNN layers must be at least 1."
        
        self.feature_size = feature_size
        self.hidden_size = hidden_size if hidden_size is not None else feature_size
        self.num_layers = num_layers
        self.gnn_layer_type = gnn_layer_type.lower()
        self.use_edge_attr = use_edge_attr
        self.graph_dependencies_config = graph_dependencies_config
        self.num_relations = num_relations
        self.gat_heads = gat_heads
        self.use_residual = use_residual
        self.dropout_p = dropout_p
        self.edge_dropout_p = edge_dropout_p
        self.lm_mask_p = lm_mask_p
        self.norm_type = norm_type.lower() if norm_type else "none"
        self.use_initial_projection = use_initial_projection
        self.edge_type_to_id = edge_type_to_id

        if self.hparams.postprocess_gnn_embeddings == "gating":
            # Gating layer input is the concatenation of GNN and original embeddings
            self.gating_layer = nn.Linear(self.feature_size * 2, self.feature_size)

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
                num_groups = min(8, num_features)
                while num_features % num_groups != 0 and num_groups > 1:
                    num_groups -= 1
                return nn.GroupNorm(num_groups, num_features)
            else:
                return None

        # --- OPTIMIZATION 2: Refactor GNN Layer Initialization ---
        # Define GNN layer configurations in a dictionary for cleaner code
        layer_configs = {
            "gcn": (GCNConv, {"out_channels": self.hidden_size}),
            "rgcn": (RGCNConv, {"out_channels": self.hidden_size, "num_relations": num_relations}),
            "gat": (GATConv, {"out_channels": self.hidden_size, "heads": gat_heads, "concat": False}),
            "rgat": (RGATConv, {"out_channels": self.hidden_size, "heads": gat_heads, "num_relations": num_relations, "concat": False}),
            "gin": (GINConv, {}),  # GIN requires a special case for its MLP
            "graphsage": (SAGEConv, {"out_channels": self.hidden_size}),
        }

        if self.gnn_layer_type not in layer_configs:
            raise ValueError(f"Unsupported GNN layer type: {self.gnn_layer_type}. Supported types: {list(layer_configs.keys())}")

        self.layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList() if self.norm_type != "none" else None
        
        input_size = self.hidden_size if use_initial_projection else feature_size
        
        for i in range(num_layers):
            if self.gnn_layer_type == "gin":
                mlp = nn.Sequential(
                    nn.Linear(input_size, self.hidden_size * 2),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size * 2, self.hidden_size)
                )
                layer = GINConv(mlp)
            else:
                layer_class, kwargs = layer_configs[self.gnn_layer_type]
                # Dynamically set the input channels for the current layer
                current_kwargs = kwargs.copy()
                # PyG layers use different names for input size, handle common ones
                if "in_channels" in GCNConv.__init__.__code__.co_varnames:
                    current_kwargs["in_channels"] = input_size
                else: # Fallback for older/different APIs
                    current_kwargs["in_feats"] = input_size

                layer = layer_class(**current_kwargs)
            
            self.layers.append(layer)

            if self.norm_type != "none":
                self.norm_layers.append(create_norm_layer(self.norm_type, self.hidden_size))
                
            input_size = self.hidden_size

    def _apply_edge_dropout(self, edge_index: torch.LongTensor, edge_attr: Optional[torch.LongTensor] = None) -> Tuple[torch.LongTensor, Optional[torch.LongTensor]]:
        if not self.training or self.edge_dropout_p == 0.0:
            return edge_index, edge_attr
        
        num_edges = edge_index.size(1)
        keep_prob = 1.0 - self.edge_dropout_p
        edge_mask = torch.rand(num_edges, device=edge_index.device) < keep_prob
        
        dropped_edge_index = edge_index[:, edge_mask]
        dropped_edge_attr = edge_attr[edge_mask] if edge_attr is not None else None
        return dropped_edge_index, dropped_edge_attr

    # --- OPTIMIZATION 3: Refactor GNN Layer Application ---
    def _apply_gnn_layer(self, layer: nn.Module, x: torch.FloatTensor, edge_index: torch.LongTensor, edge_attr: Optional[torch.LongTensor] = None) -> torch.FloatTensor:
        """Applies a single GNN layer based on its type, centralizing logic."""
        if self.gnn_layer_type in ["gcn", "gat", "gin", "graphsage"]:
            return layer(x, edge_index)
        elif self.gnn_layer_type in ["rgcn", "rgat"]:
            if self.use_edge_attr and edge_attr is not None:
                return layer(x, edge_index, edge_attr)
            else:
                default_edge_attr = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)
                return layer(x, edge_index, default_edge_attr)
        else:
            # This path should not be reached due to the check in __init__
            raise ValueError(f"Unsupported GNN layer type: {self.gnn_layer_type}")

    def _gnn_forward_pass(self, x: torch.FloatTensor, edge_index: torch.LongTensor, edge_attr: Optional[torch.LongTensor] = None) -> torch.FloatTensor:
        """
        Apply GNN layers with optional initial projection, residual connections, 
        normalization, activation and dropout.
        """
        x_orig = x
        
        if self.initial_projection is not None:
            x = self.initial_projection(x)
        
        # Apply edge dropout once for all layers
        dropped_edge_index, dropped_edge_attr = self._apply_edge_dropout(edge_index, edge_attr)
        
        for i, layer in enumerate(self.layers):
            x_residual = x
            
            # Use the refactored helper method with dropped edges
            x = self._apply_gnn_layer(layer, x, dropped_edge_index, dropped_edge_attr)
            
            if self.norm_layers is not None and i < len(self.norm_layers):
                x = self.norm_layers[i](x)
            
            if self.use_residual and x.shape == x_residual.shape:
                x = x + x_residual
            elif self.use_residual and i == 0 and self.residual_projection is not None:
                x = x + self.residual_projection(x_orig)
                    
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_p, training=self.training)
        
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
            param = next(self.parameters())
            return param.device, param.dtype

    def _create_augmented_graph_full(
        self, 
        node_features: torch.FloatTensor, 
        context_features: torch.FloatTensor, 
        edge_index: torch.LongTensor,
        edge_attr: Optional[torch.LongTensor],
        lctx_neighbor_indices: List[torch.LongTensor],
        goal_neighbor_indices: List[torch.LongTensor],
    ) -> Tuple[torch.FloatTensor, torch.LongTensor, Optional[torch.LongTensor]]:
        """
        Create augmented graph using vectorized operations to avoid CPU-bound loops.
        """
        target_device, target_dtype = self._get_target_device_and_dtype()
        num_premises = node_features.shape[0]
        num_contexts = context_features.shape[0]
        
        augmented_features = torch.cat(
            [node_features.to(target_device, target_dtype), context_features.to(target_device, target_dtype)], 
            dim=0
        )

        all_new_edges_src = []
        all_new_edges_dst = []
        all_new_edge_attrs = []
        
        def process_neighbors(neighbor_indices_list, edge_name_key):
            edge_name = _get_edge_type_name_from_tag(edge_name_key, self.graph_dependencies_config)
            if not edge_name or edge_name not in self.edge_type_to_id:
                return

            edge_type_id = self.edge_type_to_id[edge_name]
            
            valid_indices = [t for t in neighbor_indices_list if t.numel() > 0]
            if not valid_indices: return
                
            src_nodes = torch.cat(valid_indices).to(target_device)
            counts = torch.tensor([t.numel() for t in neighbor_indices_list], device=target_device)
            context_ids = torch.arange(num_contexts, device=target_device)
            dst_nodes = num_premises + context_ids.repeat_interleave(counts)
            
            all_new_edges_src.append(src_nodes)
            all_new_edges_dst.append(dst_nodes)

            if self.use_edge_attr:
                all_new_edge_attrs.append(torch.full_like(src_nodes, fill_value=edge_type_id))

        sig_cfg = self.graph_dependencies_config.get('signature_and_state', {})
        verbosity = sig_cfg.get('verbosity', 'verbose')
        process_neighbors(lctx_neighbor_indices, f"signature_{verbosity}_lctx")
        process_neighbors(goal_neighbor_indices, f"signature_{verbosity}_goal")

        if all_new_edges_src:
            new_edges = torch.stack([torch.cat(all_new_edges_src), torch.cat(all_new_edges_dst)])
            augmented_edge_index = torch.cat([edge_index.to(target_device), new_edges], dim=1)
            
            if self.use_edge_attr and edge_attr is not None and all_new_edge_attrs:
                augmented_edge_attr = torch.cat([edge_attr.to(target_device), torch.cat(all_new_edge_attrs)])
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
        """
        final_premise_embs = F.normalize(x[:num_premises], p=2, dim=1)
        final_context_embs = F.normalize(x[num_premises:], p=2, dim=1)
        return final_premise_embs, final_context_embs

    @torch.no_grad()
    def forward_embeddings(self, x: torch.FloatTensor, edge_index: torch.LongTensor, edge_attr: Optional[torch.LongTensor] = None) -> torch.FloatTensor:
        """
        Public method to apply GNN layers to embeddings. Used for external inference.
        """
        was_training = self.training
        self.eval()
        try:
            return self._gnn_forward_pass(x, edge_index, edge_attr)
        finally:
            self.train(was_training)

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

        # This now calls the new, vectorized implementation
        augmented_features, augmented_edge_index, augmented_edge_attr = self._create_augmented_graph_full(
            node_features, context_features, edge_index, edge_attr, lctx_neighbor_indices, goal_neighbor_indices
        )
            
        x = self._gnn_forward_pass(augmented_features, augmented_edge_index, augmented_edge_attr)
        gnn_premise_embs, gnn_context_embs = self._extract_and_normalize_embeddings(x, num_premises)

        if self.hparams.postprocess_gnn_embeddings == "gnn":
            final_premise_embs = gnn_premise_embs
            final_context_embs = gnn_context_embs
        else:
            # Both 'concat' and 'gating' need the original embeddings
            original_premise_embs = F.normalize(node_features.to(gnn_premise_embs.device), p=2, dim=1)
            original_context_embs = F.normalize(context_features.to(gnn_context_embs.device), p=2, dim=1)

            # Stochastic LM masking during training
            if self.training and (self.lm_mask_p is not None) and (self.lm_mask_p > 0.0):
                if torch.rand(1, device=gnn_premise_embs.device).item() < float(self.lm_mask_p):
                    original_premise_embs = torch.zeros_like(original_premise_embs)
                    original_context_embs = torch.zeros_like(original_context_embs)
            
            if self.hparams.postprocess_gnn_embeddings == "concat":
                final_premise_embs = F.normalize(torch.cat([gnn_premise_embs, original_premise_embs], dim=1), p=2, dim=1)
                final_context_embs = F.normalize(torch.cat([gnn_context_embs, original_context_embs], dim=1), p=2, dim=1)
            elif self.hparams.postprocess_gnn_embeddings == "gating":
                # Fuse premise embeddings
                gate_premise = torch.sigmoid(self.gating_layer(torch.cat([gnn_premise_embs, original_premise_embs], dim=1)))
                fused_premise_embs = gnn_premise_embs + gate_premise * original_premise_embs
                final_premise_embs = F.normalize(fused_premise_embs, p=2, dim=1)

                # Fuse context embeddings
                gate_context = torch.sigmoid(self.gating_layer(torch.cat([gnn_context_embs, original_context_embs], dim=1)))
                fused_context_embs = gnn_context_embs + gate_context * original_context_embs
                final_context_embs = F.normalize(fused_context_embs, p=2, dim=1)
            else:
                 raise ValueError(f"Unknown postprocessing option: {self.hparams.postprocess_gnn_embeddings}")

        pos_premise_emb = final_premise_embs[pos_premise_indices]
        neg_premise_embs_flat = [final_premise_embs[neg_idxs] for neg_idxs in neg_premises_indices]
        all_premise_embs = torch.cat([pos_premise_emb] + neg_premise_embs_flat, dim=0)

        similarity = torch.mm(final_context_embs, all_premise_embs.t())
        assert -1.001 <= similarity.min() <= similarity.max() <= 1.001, f"Got {similarity.min()} and {similarity.max()}"
        loss = F.mse_loss(similarity, label)
        return loss

    def on_train_epoch_start(self) -> None:
        """Calculate and store the batch indices for logging in this epoch."""
        if not self.trainer or not hasattr(self.trainer, 'train_dataloader'):
            self.logging_steps = []
            return

        total_batches = len(self.trainer.train_dataloader)
        if total_batches == 0:
            self.logging_steps = []
            return
            
        num_logs = self.hparams.num_logs_per_epoch
        if num_logs <= 0:
            self.logging_steps = []
            return

        if num_logs > total_batches:
            num_logs = total_batches

        interval = total_batches // num_logs if num_logs > 0 else total_batches
        self.logging_steps = [(i * interval) for i in range(num_logs)]
        # Log on the very last batch if it's not already included
        if total_batches - 1 not in self.logging_steps:
             self.logging_steps.append(total_batches - 1)
        self.logging_steps = sorted(list(set(self.logging_steps)))

        logger.info(f"Epoch {self.current_epoch}: Will log training samples at batch indices: {self.logging_steps}")

    @torch.no_grad()
    def _log_training_sample(self, batch: Dict[str, Any], batch_idx: int):
        """Logs a single sample from the batch for inspection during training."""
        self.eval()
        context_text = batch["context"][0].serialize()
        lctx_indices = batch["lctx_neighbor_indices"][0]
        goal_indices = batch["goal_neighbor_indices"][0]
        corpus = self.trainer.datamodule.corpus

        logger.info(f"\n--- Training Sample Log (Epoch: {self.current_epoch}, Batch: {batch_idx}, Global Step: {self.global_step}) ---")
        logger.info(f"State/Goal Text: \n{context_text}")
        logger.info("  Ingoing Edges to Ghost Node:")
        for idx in lctx_indices:
            logger.info(f"    <- [lctx] <- {corpus.all_premises[idx.item()].full_name}")
        for idx in goal_indices:
            logger.info(f"    <- [goal] <- {corpus.all_premises[idx.item()].full_name}")

        initial_context_emb = batch["context_features"][0]
        logger.info(f"  Initial Context Embedding (first 8 values): {initial_context_emb[:8].cpu().numpy()}")

        num_premises = batch["node_features"].shape[0]
        aug_features, aug_edge_index, aug_edge_attr = self._create_augmented_graph_full(
            batch["node_features"], batch["context_features"][[0]], batch["edge_index"],
            batch.get("edge_attr"), [batch["lctx_neighbor_indices"][0]], [batch["goal_neighbor_indices"][0]],
        )
        x = self._gnn_forward_pass(aug_features, aug_edge_index, aug_edge_attr)
        _, gnn_context_embs = self._extract_and_normalize_embeddings(x, num_premises)

        postprocess_option = self.hparams.postprocess_gnn_embeddings
        if postprocess_option == "gnn":
            final_context_emb = gnn_context_embs
        else:
            # Both 'concat' and 'gating' need the original embeddings
            original_context_emb = F.normalize(initial_context_emb.unsqueeze(0).to(gnn_context_embs.device), p=2, dim=1)
            
            if postprocess_option == "concat":
                concatenated_emb = torch.cat([gnn_context_embs, original_context_emb], dim=1)
                final_context_emb = F.normalize(concatenated_emb, p=2, dim=1)
            elif postprocess_option == "gating":
                gate = torch.sigmoid(self.gating_layer(torch.cat([gnn_context_embs, original_context_emb], dim=1)))
                fused_emb = gnn_context_embs + gate * original_context_emb
                final_context_emb = F.normalize(fused_emb, p=2, dim=1)
            else:
                # Fallback to just GNN embeddings if option is unknown
                final_context_emb = gnn_context_embs

        final_context_emb_vec = final_context_emb.squeeze(0)

        logger.info(f"  Final GNN Context Embedding (first 8 values):   {final_context_emb_vec[:8].cpu().numpy()}")
        logger.info("--- End of Training Sample Log ---")
        self.train()

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        if hasattr(self, "logging_steps") and batch_idx in self.logging_steps:
            if hasattr(self, "trainer") and hasattr(self.trainer, "datamodule"):
                self._log_training_sample(batch, batch_idx)

        loss = self(
            batch["node_features"], batch["edge_index"], batch["context_features"],
            batch["lctx_neighbor_indices"], batch["goal_neighbor_indices"],
            batch["pos_premise_indices"], batch["neg_premises_indices"],
            batch["label"], batch.get("edge_attr", None),
        )
        self.log("loss_train", loss, on_epoch=True, sync_dist=True, batch_size=len(batch["context_features"]))
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        return get_optimizers(self.parameters(), self.trainer, self.hparams.lr, self.hparams.warmup_steps)
    
    @torch.no_grad()
    def compute_premise_layer_embeddings(
        self,
        initial_embeddings: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_attr: Optional[torch.LongTensor],
    ) -> List[torch.FloatTensor]:
        """
        Performs a GNN forward pass on the premise graph and caches the embeddings at each layer.
        """
        self.eval()
        layer_embeddings = []
        x = initial_embeddings
        x_orig = x
        
        if self.initial_projection is not None:
            x = self.initial_projection(x)
        layer_embeddings.append(x.clone())
        
        for i, layer in enumerate(self.layers):
            x_residual = x
            
            # Use the refactored helper method
            x = self._apply_gnn_layer(layer, x, edge_index, edge_attr)

            if self.norm_layers is not None and i < len(self.norm_layers):
                x = self.norm_layers[i](x)

            if self.use_residual and x.shape == x_residual.shape:
                x = x + x_residual
            elif self.use_residual and i == 0 and self.residual_projection is not None:
                x = x + self.residual_projection(x_orig)

            if i < len(self.layers) - 1:
                x = F.relu(x)

            layer_embeddings.append(x.clone())

        if self.final_projection is not None:
            final_embs = self.final_projection(layer_embeddings[-1])
            layer_embeddings.append(final_embs.clone())

        return layer_embeddings

    # The rest of the file remains unchanged as it was not part of the requested optimizations
    @torch.no_grad()
    def get_dynamic_context_embedding(
        self,
        initial_context_embs: torch.FloatTensor,
        batch_lctx_neighbor_indices: List[torch.LongTensor],
        batch_goal_neighbor_indices: List[torch.LongTensor],
        premise_layer_embeddings: List[torch.FloatTensor],
    ) -> torch.FloatTensor:
        """
        Efficiently computes final GNN-refined embeddings for a batch of contexts (ghost nodes).
        """
        self.eval()
        context_embs = initial_context_embs

        if self.initial_projection is not None:
            context_embs = self.initial_projection(context_embs)

        sig_cfg = self.graph_dependencies_config.get('signature_and_state', {})
        verbosity = sig_cfg.get('verbosity', 'verbose')
        lctx_tag = f"signature_{verbosity}_lctx"
        goal_tag = f"signature_{verbosity}_goal"
        lctx_edge_name = _get_edge_type_name_from_tag(lctx_tag, self.graph_dependencies_config)
        goal_edge_name = _get_edge_type_name_from_tag(goal_tag, self.graph_dependencies_config)
        lctx_edge_type_id = self.edge_type_to_id.get(lctx_edge_name)
        goal_edge_type_id = self.edge_type_to_id.get(goal_edge_name)

        batch_connections = []
        for i in range(len(initial_context_embs)):
            conn_indices, conn_types = [], []
            if lctx_edge_type_id is not None and len(batch_lctx_neighbor_indices[i]) > 0:
                conn_indices.append(batch_lctx_neighbor_indices[i])
                conn_types.append(torch.full_like(batch_lctx_neighbor_indices[i], lctx_edge_type_id))
            if goal_edge_type_id is not None and len(batch_goal_neighbor_indices[i]) > 0:
                conn_indices.append(batch_goal_neighbor_indices[i])
                conn_types.append(torch.full_like(batch_goal_neighbor_indices[i], goal_edge_type_id))
            
            batch_connections.append(
                (torch.cat(conn_indices), torch.cat(conn_types)) if conn_indices else
                (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))
            )

        for i, layer in enumerate(self.layers):
            context_residual = context_embs
            current_premise_embs = premise_layer_embeddings[i]

            if self.gnn_layer_type == "gcn":
                simple_conns = [con[0] for con in batch_connections]
                context_embs = compute_ghost_node_embeddings_gcn(layer, current_premise_embs, None, context_embs.unbind(0), simple_conns)
            elif self.gnn_layer_type == "rgcn":
                context_embs = compute_ghost_node_embeddings_rgcn(layer, current_premise_embs, context_embs.unbind(0), batch_connections)
            elif self.gnn_layer_type == "graphsage":
                simple_conns = [con[0] for con in batch_connections]
                context_embs = compute_ghost_node_embeddings_graphsage(layer, current_premise_embs, context_embs.unbind(0), simple_conns)
            elif self.gnn_layer_type == "gat":
                simple_conns = [con[0] for con in batch_connections]
                context_embs = compute_ghost_node_embeddings_gat(layer, current_premise_embs, context_embs.unbind(0), simple_conns)
            elif self.gnn_layer_type == "rgat":
                context_embs = compute_ghost_node_embeddings_rgat(layer, current_premise_embs, context_embs.unbind(0), batch_connections)
            elif self.gnn_layer_type == "gin":
                simple_conns = [con[0] for con in batch_connections]
                context_embs = compute_ghost_node_embeddings_gin(layer, current_premise_embs, context_embs.unbind(0), simple_conns)
            
            if self.norm_layers is not None and i < len(self.norm_layers):
                context_embs = self.norm_layers[i](context_embs)
            if self.use_residual and context_embs.shape == context_residual.shape:
                context_embs = context_embs + context_residual
            elif self.use_residual and i == 0 and self.residual_projection is not None:
                context_embs = context_embs + self.residual_projection(initial_context_embs)
            if i < len(self.layers) - 1:
                context_embs = F.relu(context_embs)

        if self.final_projection is not None:
            context_embs = self.final_projection(context_embs)

        if self.hparams.postprocess_gnn_embeddings == "gnn":
            final_embs = F.normalize(context_embs, p=2, dim=1)
        else:
            # Both 'concat' and 'gating' need the original embeddings
            orig_embs_norm = F.normalize(initial_context_embs, p=2, dim=1)
            gnn_embs_norm = F.normalize(context_embs, p=2, dim=1)

            if self.hparams.postprocess_gnn_embeddings == "concat":
                final_embs = F.normalize(torch.cat([gnn_embs_norm, orig_embs_norm], dim=1), p=2, dim=1)
            elif self.hparams.postprocess_gnn_embeddings == "gating":
                gate = torch.sigmoid(self.gating_layer(torch.cat([gnn_embs_norm, orig_embs_norm], dim=1)))
                fused_embs = gnn_embs_norm + gate * orig_embs_norm
                final_embs = F.normalize(fused_embs, p=2, dim=1)
            else:
                raise ValueError(f"Unknown postprocessing option: {self.hparams.postprocess_gnn_embeddings}")
            
        return final_embs