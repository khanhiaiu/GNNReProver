# retrieval/gnn_model.py

import torch
from torch_geometric.data import Data


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
        loss_function: str = "mse",
        l1_lambda: float = 0.0,
        weight_decay: float = 0.0,
        gnn_layer_type: str = "rgcn",
        hidden_size: Optional[int] = None,
        num_relations: int = 3,
        use_edge_attr: bool = True,
        gat_heads: int = 8,
        use_residual: bool = True,
        dropout_p: float = 0.5,
        edge_dropout_p: float = 0.1,
        mask_regime_probs: Optional[Dict[str, float]] = None,
        full_drop_prob: float = 0.5,
        norm_type: str = "layer",
        use_initial_projection: bool = True,
        postprocess_gnn_embeddings: str = "gnn",
        negative_mining: Optional[Dict[str, Any]] = None,
        num_logs_per_epoch: int = 5,
        neighbor_sampling_sizes: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        # Avoid mutable default argument
        if mask_regime_probs is None:
            mask_regime_probs = {"no_drop": 0.1, "drop_concat": 0.2, "full_drop": 0.7}

        if negative_mining is None:
            negative_mining = {
                "strategy": "random",
                "num_negatives": 3,
                "num_in_file_negatives": 1
            }

        self.save_hyperparameters()
        assert self.hparams.loss_function in ["mse", "cross_entropy"], \
           f"loss_function must be 'mse' or 'cross_entropy', but got {self.hparams.loss_function}"
        assert num_layers >= 1, "Number of GNN layers must be at least 1."

        # Validate and normalize mask_regime_probs
        expected_keys = {"no_drop", "drop_concat", "full_drop"}
        assert expected_keys == set(mask_regime_probs.keys()), f"mask_regime_probs must contain keys: {expected_keys}"
        assert all(p >= 0 for p in mask_regime_probs.values()), "Probabilities in mask_regime_probs must be non-negative."
        prob_sum = sum(mask_regime_probs.values())
        assert prob_sum > 1e-6, "Sum of probabilities in mask_regime_probs must be positive."
        
        self.regimes = list(mask_regime_probs.keys())
        self.regime_probs = [p / prob_sum for p in mask_regime_probs.values()]
        
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
        self.norm_type = norm_type.lower() if norm_type else "none"
        self.use_initial_projection = use_initial_projection
        self.edge_type_to_id = edge_type_to_id

        if self.hparams.postprocess_gnn_embeddings == "gating":
            self.gating_layer = nn.Linear(self.feature_size * 2, self.feature_size)

        if self.use_initial_projection:
            self.initial_projection = nn.Linear(feature_size, self.hidden_size)
        else:
            self.initial_projection = None

        if self.use_residual and self.hidden_size != feature_size:
            self.residual_projection = nn.Linear(feature_size, self.hidden_size)
        else:
            self.residual_projection = None

        if self.hidden_size != feature_size:
            self.final_projection = nn.Linear(self.hidden_size, feature_size)
        else:
            self.final_projection = None

        def create_norm_layer(norm_type: str, num_features: int) -> Optional[nn.Module]:
            if norm_type == "batch": return nn.BatchNorm1d(num_features)
            elif norm_type == "layer": return nn.LayerNorm(num_features)
            elif norm_type == "instance": return nn.InstanceNorm1d(num_features)
            elif norm_type == "group":
                num_groups = min(8, num_features)
                while num_features % num_groups != 0 and num_groups > 1:
                    num_groups -= 1
                return nn.GroupNorm(num_groups, num_features)
            else: return None

        layer_configs = {
            "gcn": (GCNConv, {"out_channels": self.hidden_size}),
            "rgcn": (RGCNConv, {"out_channels": self.hidden_size, "num_relations": num_relations}),
            "gat": (GATConv, {"out_channels": self.hidden_size, "heads": gat_heads, "concat": False}),
            "rgat": (RGATConv, {"out_channels": self.hidden_size, "heads": gat_heads, "num_relations": num_relations, "concat": False}),
            "gin": (GINConv, {}),
            "graphsage": (SAGEConv, {"out_channels": self.hidden_size}),
        }

        if self.gnn_layer_type not in layer_configs:
            raise ValueError(f"Unsupported GNN layer type: {self.gnn_layer_type}. Supported types: {list(layer_configs.keys())}")

        self.layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList() if self.norm_type != "none" else None
        
        input_size = self.hidden_size if use_initial_projection else feature_size
        
        for i in range(num_layers):
            if self.gnn_layer_type == "gin":
                mlp = nn.Sequential(nn.Linear(input_size, self.hidden_size * 2), nn.ReLU(), nn.Linear(self.hidden_size * 2, self.hidden_size))
                layer = GINConv(mlp)
            else:
                layer_class, kwargs = layer_configs[self.gnn_layer_type]
                current_kwargs = kwargs.copy()
                if "in_channels" in GCNConv.__init__.__code__.co_varnames:
                    current_kwargs["in_channels"] = input_size
                else:
                    current_kwargs["in_feats"] = input_size
                layer = layer_class(**current_kwargs)
            
            self.layers.append(layer)
            if self.norm_type != "none":
                self.norm_layers.append(create_norm_layer(self.norm_type, self.hidden_size))
            input_size = self.hidden_size

    def _sample_neighbors(self, edge_index: torch.LongTensor, edge_attr: Optional[torch.LongTensor], k: int) -> Tuple[torch.LongTensor, Optional[torch.LongTensor]]:
        if k < 0 or not self.training: return edge_index, edge_attr
        num_edges, dest_nodes = edge_index.size(1), edge_index[1]
        sort_key = dest_nodes * self.hparams.num_relations + edge_attr if edge_attr is not None else dest_nodes
        perm = torch.randperm(num_edges, device=edge_index.device)
        sorted_perm_by_key = torch.argsort(sort_key[perm], stable=True)
        perm_sorted = perm[sorted_perm_by_key]
        edge_index_perm, edge_attr_perm, sort_key_perm = edge_index[:, perm_sorted], (edge_attr[perm_sorted] if edge_attr is not None else None), sort_key[perm_sorted]
        _, group_counts = torch.unique_consecutive(sort_key_perm, return_counts=True)
        group_starts = torch.cat([torch.tensor([0], device=edge_index.device), torch.cumsum(group_counts, 0)[:-1]])
        intra_group_idx = torch.arange(num_edges, device=edge_index.device) - group_starts.repeat_interleave(group_counts)
        mask = intra_group_idx < k
        return edge_index_perm[:, mask], (edge_attr_perm[mask] if edge_attr is not None else None)

    def _apply_edge_dropout(self, edge_index: torch.LongTensor, edge_attr: Optional[torch.LongTensor] = None) -> Tuple[torch.LongTensor, Optional[torch.LongTensor]]:
        if not self.training or self.edge_dropout_p == 0.0: return edge_index, edge_attr
        keep_prob = 1.0 - self.edge_dropout_p
        edge_mask = torch.rand(edge_index.size(1), device=edge_index.device) < keep_prob
        return edge_index[:, edge_mask], (edge_attr[edge_mask] if edge_attr is not None else None)

    def _apply_gnn_layer(self, layer: nn.Module, x: torch.FloatTensor, edge_index: torch.LongTensor, edge_attr: Optional[torch.LongTensor] = None) -> torch.FloatTensor:
        if self.gnn_layer_type in ["gcn", "gat", "gin", "graphsage"]: return layer(x, edge_index)
        elif self.gnn_layer_type in ["rgcn", "rgat"]:
            if self.use_edge_attr and edge_attr is not None: return layer(x, edge_index, edge_attr)
            else: return layer(x, edge_index, torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device))
        else: raise ValueError(f"Unsupported GNN layer type: {self.gnn_layer_type}")

    def _gnn_forward_pass(self, x: torch.FloatTensor, edge_index: torch.LongTensor, edge_attr: Optional[torch.LongTensor] = None) -> torch.FloatTensor:
        x, x_orig = x.to(next(self.parameters()).dtype), x
        if self.initial_projection is not None: x = self.initial_projection(x)
        for i, layer in enumerate(self.layers):
            use_neighbor_sampling = self.hparams.neighbor_sampling_sizes is not None and self.training and i < len(self.hparams.neighbor_sampling_sizes) and self.hparams.neighbor_sampling_sizes[i] >= 0
            if use_neighbor_sampling:
                k = self.hparams.neighbor_sampling_sizes[i]
                layer_edge_index, layer_edge_attr = self._sample_neighbors(edge_index, edge_attr, k)
            else:
                layer_edge_index, layer_edge_attr = self._apply_edge_dropout(edge_index, edge_attr)
            x_residual = x
            x = self._apply_gnn_layer(layer, x, layer_edge_index, layer_edge_attr)
            if self.norm_layers is not None and i < len(self.norm_layers): x = self.norm_layers[i](x)
            if self.use_residual and x.shape == x_residual.shape: x = x + x_residual
            elif self.use_residual and i == 0 and self.residual_projection is not None: x = x + self.residual_projection(x_orig)
            if i < len(self.layers) - 1: x = F.relu(F.dropout(x, p=self.dropout_p, training=self.training))
        if self.final_projection is not None: x = self.final_projection(x)
        return x

    def _get_target_device_and_dtype(self) -> Tuple[torch.device, torch.dtype]:
        first_layer = self.layers[0]
        if hasattr(first_layer, 'weight'): return first_layer.weight.device, first_layer.weight.dtype
        elif hasattr(first_layer, 'lin_l') and hasattr(first_layer.lin_l, 'weight'): return first_layer.lin_l.weight.device, first_layer.lin_l.weight.dtype
        else: param = next(self.parameters()); return param.device, param.dtype

    def _create_augmented_graph_full(self, node_features: torch.FloatTensor, context_features: torch.FloatTensor, edge_index: torch.LongTensor, edge_attr: Optional[torch.LongTensor], lctx_neighbor_indices: List[torch.LongTensor], goal_neighbor_indices: List[torch.LongTensor]) -> Tuple[torch.FloatTensor, torch.LongTensor, Optional[torch.LongTensor]]:
        target_device, target_dtype = self._get_target_device_and_dtype()
        num_premises, num_contexts = node_features.shape[0], context_features.shape[0]
        augmented_features = torch.cat([node_features.to(target_device, target_dtype), context_features.to(target_device, target_dtype)], dim=0)
        all_new_edges_src, all_new_edges_dst, all_new_edge_attrs = [], [], []
        def process_neighbors(neighbor_indices_list, edge_name_key):
            edge_name = _get_edge_type_name_from_tag(edge_name_key, self.graph_dependencies_config)
            if not edge_name or edge_name not in self.edge_type_to_id: return
            edge_type_id = self.edge_type_to_id[edge_name]
            valid_indices = [t for t in neighbor_indices_list if t.numel() > 0]
            if not valid_indices: return
            src_nodes = torch.cat(valid_indices).to(target_device)
            counts = torch.tensor([t.numel() for t in neighbor_indices_list], device=target_device)
            dst_nodes = num_premises + torch.arange(num_contexts, device=target_device).repeat_interleave(counts)
            all_new_edges_src.append(src_nodes); all_new_edges_dst.append(dst_nodes)
            if self.use_edge_attr: all_new_edge_attrs.append(torch.full_like(src_nodes, fill_value=edge_type_id))
        sig_cfg = self.graph_dependencies_config.get('signature_and_state', {})
        verbosity = sig_cfg.get('verbosity', 'verbose')
        process_neighbors(lctx_neighbor_indices, f"signature_{verbosity}_lctx")
        process_neighbors(goal_neighbor_indices, f"signature_{verbosity}_goal")
        if all_new_edges_src:
            new_edges = torch.stack([torch.cat(all_new_edges_src), torch.cat(all_new_edges_dst)])
            augmented_edge_index = torch.cat([edge_index.to(target_device), new_edges], dim=1)
            augmented_edge_attr = torch.cat([edge_attr.to(target_device), torch.cat(all_new_edge_attrs)]) if self.use_edge_attr and edge_attr is not None and all_new_edge_attrs else None
        else:
            augmented_edge_index, augmented_edge_attr = edge_index.to(target_device), (edge_attr.to(target_device) if edge_attr is not None else None)
        return augmented_features, augmented_edge_index, augmented_edge_attr

    def _get_final_embeddings(self, x: torch.FloatTensor, node_features: torch.FloatTensor, context_features: torch.FloatTensor, normalize: bool, regimes: List[str]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        num_premises = node_features.shape[0]
        gnn_premise_embs, gnn_context_embs = x[:num_premises], x[num_premises:]
        if self.hparams.postprocess_gnn_embeddings == "gnn":
            final_premise_embs, final_context_embs = gnn_premise_embs, gnn_context_embs
        else:
            original_premise_embs, original_context_embs = node_features.to(gnn_premise_embs.device, gnn_premise_embs.dtype), context_features.to(gnn_context_embs.device, gnn_context_embs.dtype)
            if normalize:
                gnn_premise_embs, gnn_context_embs = F.normalize(gnn_premise_embs, p=2, dim=1), F.normalize(gnn_context_embs, p=2, dim=1)
                original_premise_embs, original_context_embs = F.normalize(original_premise_embs, p=2, dim=1), F.normalize(original_context_embs, p=2, dim=1)
            if self.hparams.postprocess_gnn_embeddings == "concat": final_premise_embs = torch.cat([gnn_premise_embs, original_premise_embs], dim=1)
            elif self.hparams.postprocess_gnn_embeddings == "gating": final_premise_embs = gnn_premise_embs + torch.sigmoid(self.gating_layer(torch.cat([gnn_premise_embs, original_premise_embs], dim=1))) * original_premise_embs
            else: raise ValueError(f"Unknown postprocessing option: {self.hparams.postprocess_gnn_embeddings}")
            original_part = torch.zeros_like(original_context_embs)
            if self.training:
                no_drop_mask = torch.tensor([r == 'no_drop' for r in regimes], device=self.device)
                if no_drop_mask.any(): original_part[no_drop_mask] = original_context_embs[no_drop_mask]
            else: original_part = original_context_embs
            if self.hparams.postprocess_gnn_embeddings == "concat": final_context_embs = torch.cat([gnn_context_embs, original_part], dim=1)
            elif self.hparams.postprocess_gnn_embeddings == "gating": final_context_embs = gnn_context_embs + torch.sigmoid(self.gating_layer(torch.cat([gnn_context_embs, original_part], dim=1))) * original_part
        if normalize:
            final_premise_embs, final_context_embs = F.normalize(final_premise_embs, p=2, dim=1), F.normalize(final_context_embs, p=2, dim=1)
        return final_premise_embs, final_context_embs

    @classmethod
    def load(cls, ckpt_path: str, device, freeze: bool) -> "GNNRetriever":
        return load_checkpoint(cls, ckpt_path, device, freeze)

    def on_train_epoch_start(self) -> None:
        if not self.trainer or not hasattr(self.trainer, 'train_dataloader'): self.logging_steps = []; return
        total_batches = len(self.trainer.train_dataloader)
        if total_batches == 0: self.logging_steps = []; return
        num_logs = self.hparams.num_logs_per_epoch
        if num_logs <= 0: self.logging_steps = []; return
        if num_logs > total_batches: num_logs = total_batches
        interval = total_batches // num_logs if num_logs > 0 else total_batches
        self.logging_steps = [(i * interval) for i in range(num_logs)]
        if total_batches - 1 not in self.logging_steps: self.logging_steps.append(total_batches - 1)
        self.logging_steps = sorted(list(set(self.logging_steps)))
        logger.info(f"Epoch {self.current_epoch}: Will log training samples at batch indices: {self.logging_steps}")

    @torch.no_grad()
    def _log_training_sample(self, batch: Dict[str, Any], batch_idx: int):
        self.eval()
        context_text, lctx_indices, goal_indices, corpus = batch["context"][0].serialize(), batch["lctx_neighbor_indices"][0], batch["goal_neighbor_indices"][0], self.trainer.datamodule.corpus
        logger.info(f"\n--- Training Sample Log (Epoch: {self.current_epoch}, Batch: {batch_idx}, Global Step: {self.global_step}) ---")
        logger.info(f"State/Goal Text: \n{context_text}"); logger.info("  Ingoing Edges to Ghost Node:")
        for idx in lctx_indices: logger.info(f"    <- [lctx] <- {corpus.all_premises[idx.item()].full_name}")
        for idx in goal_indices: logger.info(f"    <- [goal] <- {corpus.all_premises[idx.item()].full_name}")
        initial_context_emb = batch["context_features"][0]
        logger.info(f"  Initial Context Embedding (first 8 values): {initial_context_emb[:8].cpu().numpy()}")
        aug_features, aug_edge_index, aug_edge_attr = self._create_augmented_graph_full(batch["node_features"], batch["context_features"][[0]], batch["edge_index"], batch.get("edge_attr"), [batch["lctx_neighbor_indices"][0]], [batch["goal_neighbor_indices"][0]])
        x = self._gnn_forward_pass(aug_features, aug_edge_index, aug_edge_attr)
        normalize = self.hparams.loss_function == "mse"
        _, final_context_emb = self._get_final_embeddings(x, batch["node_features"], batch["context_features"][[0]], normalize=normalize, regimes=["no_drop"])
        final_context_emb_vec = final_context_emb.squeeze(0)
        logger.info(f"  Final ({'Normalized' if normalize else 'Unnormalized'}) Context Embedding (first 8 values):   {final_context_emb_vec[:8].cpu().numpy()}"); logger.info("--- End of Training Sample Log ---")
        self.train()

    @torch.no_grad()
    def forward_embeddings(self, x: torch.FloatTensor, edge_index: torch.LongTensor, edge_attr: Optional[torch.LongTensor] = None) -> torch.FloatTensor:
        was_training = self.training; self.eval()
        try: return self._gnn_forward_pass(x, edge_index, edge_attr)
        finally: self.train(was_training)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        if self.hparams.negative_mining['strategy'] == 'hard':
            augmented_features, aug_edge_index, aug_edge_attr = self._create_augmented_graph_full(batch["node_features"], batch["context_features"], batch["edge_index"], batch.get("edge_attr"), batch["lctx_neighbor_indices"], batch["goal_neighbor_indices"])
            x = self._gnn_forward_pass(augmented_features, aug_edge_index, aug_edge_attr)
            normalize = self.hparams.loss_function == "mse"
            regimes = ["no_drop"] * batch["context_features"].shape[0]
            final_premise_embs, final_context_embs = self._get_final_embeddings(x, batch["node_features"], batch["context_features"], normalize=normalize, regimes=regimes)
            all_scores = torch.mm(final_context_embs, final_premise_embs.t())
            batch_size, num_premises = all_scores.shape
            
            # --- FIX: Initialize pos_mask BEFORE it is used ---
            pos_mask = torch.zeros_like(all_scores, dtype=torch.bool)
            
            all_pos_indices_flat = [p for p in batch['all_pos_premise_indices'] if p.numel() > 0]
            if all_pos_indices_flat:
                 all_pos_indices = torch.cat(all_pos_indices_flat)
                 batch_indices = torch.arange(batch_size, device=all_scores.device).repeat_interleave(
                     torch.tensor([p.numel() for p in batch['all_pos_premise_indices']], device=all_scores.device)
                 )
                 pos_mask[batch_indices, all_pos_indices] = True
            
            all_scores.masked_fill_(pos_mask, -torch.inf)
            num_neg, num_in_file_neg = self.hparams.negative_mining['num_negatives'], self.hparams.negative_mining['num_in_file_negatives']
            hard_neg_indices_parts = []
            
            if num_in_file_neg > 0:
                in_file_mask = torch.zeros_like(all_scores, dtype=torch.bool)
                all_in_file_indices_flat = [p for p in batch['accessible_in_file_indices'] if p.numel() > 0]
                if all_in_file_indices_flat:
                    all_in_file_indices = torch.cat(all_in_file_indices_flat)
                    in_file_batch_indices = torch.arange(batch_size, device=all_scores.device).repeat_interleave(
                        torch.tensor([p.numel() for p in batch['accessible_in_file_indices']], device=all_scores.device)
                    )
                    in_file_mask[in_file_batch_indices, all_in_file_indices] = True
                
                in_file_scores = all_scores.clone(); in_file_scores.masked_fill_(~in_file_mask, -torch.inf)
                _, in_file_negs = torch.topk(in_file_scores, k=num_in_file_neg, dim=1)
                hard_neg_indices_parts.append(in_file_negs); all_scores.masked_fill_(in_file_mask, -torch.inf)
            
            num_global_neg = num_neg - num_in_file_neg
            if num_global_neg > 0:
                _, global_negs = torch.topk(all_scores, k=num_global_neg, dim=1)
                hard_neg_indices_parts.append(global_negs)
            
            hard_neg_indices = torch.cat(hard_neg_indices_parts, dim=1)
            pos_indices = batch['pos_premise_indices'].unsqueeze(1)
            all_indices = torch.cat([pos_indices, hard_neg_indices], dim=1)
            batch_embs = final_premise_embs[all_indices]
            scores = torch.bmm(final_context_embs.unsqueeze(1), batch_embs.transpose(1, 2)).squeeze(1)
            new_label = torch.zeros_like(scores); new_label[:, 0] = 1.0
            is_actually_positive = torch.gather(pos_mask, 1, hard_neg_indices)
            new_label[:, 1:] = is_actually_positive.float()
            
            loss = F.mse_loss(scores, new_label) if self.hparams.loss_function == "mse" else F.binary_cross_entropy_with_logits(scores, new_label)
        else:
            node_features, edge_index, context_features = batch["node_features"], batch["edge_index"], batch["context_features"]
            lctx_neighbor_indices, goal_neighbor_indices = batch["lctx_neighbor_indices"], batch["goal_neighbor_indices"]
            edge_attr, pos_premise_indices, neg_premises_indices, label = batch.get("edge_attr", None), batch["pos_premise_indices"], batch["neg_premises_indices"], batch["label"]
            augmented_features, augmented_edge_index, augmented_edge_attr = self._create_augmented_graph_full(node_features, context_features, edge_index, edge_attr, lctx_neighbor_indices, goal_neighbor_indices)
            num_contexts = context_features.shape[0]
            if self.training:
                regime_indices = torch.multinomial(torch.tensor(self.regime_probs, device=self.device), num_samples=num_contexts, replacement=True)
                per_context_regimes = [self.regimes[i] for i in regime_indices]
                full_drop_mask = (regime_indices == self.regimes.index('full_drop'))
                if full_drop_mask.any():
                    context_part_features = augmented_features[node_features.shape[0]:]; context_mask = torch.ones_like(context_part_features)
                    random_mask = (torch.rand(full_drop_mask.sum(), 1, device=self.device) > self.hparams.full_drop_prob).to(augmented_features.dtype)
                    context_mask[full_drop_mask] = random_mask; augmented_features[node_features.shape[0]:] = context_part_features * context_mask
            else: per_context_regimes = ["no_drop"] * num_contexts
            x = self._gnn_forward_pass(augmented_features, augmented_edge_index, augmented_edge_attr)
            final_premise_embs, final_context_embs = self._get_final_embeddings(x, node_features, context_features, normalize=self.hparams.loss_function == "mse", regimes=per_context_regimes)
            pos_premise_emb = final_premise_embs[pos_premise_indices]
            neg_premise_embs_flat = [final_premise_embs[neg_idxs] for neg_idxs in neg_premises_indices]
            all_premise_embs = torch.cat([pos_premise_emb] + neg_premise_embs_flat, dim=0)
            scores = torch.mm(final_context_embs, all_premise_embs.t())
            if self.hparams.loss_function == "mse":
                assert -1.001 <= scores.min() <= scores.max() <= 1.001, f"Got {scores.min()} and {scores.max()}"
                loss = F.mse_loss(scores, label)
            else: loss = F.binary_cross_entropy_with_logits(scores, label)
        
        if hasattr(self, "logging_steps") and batch_idx in self.logging_steps:
            if hasattr(self, "trainer") and hasattr(self.trainer, "datamodule"): self._log_training_sample(batch, batch_idx)
        self.log("loss_train_unregularized", loss, on_epoch=True, sync_dist=True, batch_size=len(batch["context_features"]))
        total_loss = loss
        if self.hparams.l1_lambda > 0:
            l1_penalty = 0.0; modules_to_regularize = list(self.layers)
            if self.initial_projection is not None: modules_to_regularize.append(self.initial_projection)
            if self.residual_projection is not None: modules_to_regularize.append(self.residual_projection)
            if self.final_projection is not None: modules_to_regularize.append(self.final_projection)
            if hasattr(self, 'gating_layer'): modules_to_regularize.append(self.gating_layer)
            for module in modules_to_regularize:
                for name, param in module.named_parameters():
                    if 'weight' in name: l1_penalty += torch.norm(param, 1)
            l1_loss = self.hparams.l1_lambda * l1_penalty
            self.log("l1_loss_train", l1_loss, on_step=True, on_epoch=True, sync_dist=True)
            total_loss = loss + l1_loss
        self.log("loss_train", total_loss, on_epoch=True, sync_dist=True, batch_size=len(batch["context_features"]))
        return total_loss

    def configure_optimizers(self) -> Dict[str, Any]:
        return get_optimizers(self.parameters(), self.trainer, self.hparams.lr, self.hparams.warmup_steps, weight_decay=self.hparams.weight_decay)
    
    @torch.no_grad()
    def compute_premise_layer_embeddings(self, initial_embeddings: torch.FloatTensor, edge_index: torch.LongTensor, edge_attr: Optional[torch.LongTensor]) -> List[torch.FloatTensor]:
        self.eval(); layer_embeddings = []
        x, x_orig = initial_embeddings, initial_embeddings
        if self.initial_projection is not None: x = self.initial_projection(x)
        layer_embeddings.append(x.clone())
        for i, layer in enumerate(self.layers):
            x_residual = x
            x = self._apply_gnn_layer(layer, x, edge_index, edge_attr)
            if self.norm_layers is not None and i < len(self.norm_layers): x = self.norm_layers[i](x)
            if self.use_residual and x.shape == x_residual.shape: x = x + x_residual
            elif self.use_residual and i == 0 and self.residual_projection is not None: x = x + self.residual_projection(x_orig)
            if i < len(self.layers) - 1: x = F.relu(x)
            layer_embeddings.append(x.clone())
        if self.final_projection is not None: layer_embeddings.append(self.final_projection(layer_embeddings[-1]).clone())
        return layer_embeddings

    @torch.no_grad()
    def get_dynamic_context_embedding(self, initial_context_embs: torch.FloatTensor, batch_lctx_neighbor_indices: List[torch.LongTensor], batch_goal_neighbor_indices: List[torch.LongTensor], premise_layer_embeddings: List[torch.FloatTensor]) -> torch.FloatTensor:
        self.eval(); context_embs = initial_context_embs
        if self.initial_projection is not None: context_embs = self.initial_projection(context_embs)
        sig_cfg = self.graph_dependencies_config.get('signature_and_state', {})
        verbosity = sig_cfg.get('verbosity', 'verbose')
        lctx_tag, goal_tag = f"signature_{verbosity}_lctx", f"signature_{verbosity}_goal"
        lctx_edge_name, goal_edge_name = _get_edge_type_name_from_tag(lctx_tag, self.graph_dependencies_config), _get_edge_type_name_from_tag(goal_tag, self.graph_dependencies_config)
        lctx_edge_type_id, goal_edge_type_id = self.edge_type_to_id.get(lctx_edge_name), self.edge_type_to_id.get(goal_edge_name)
        batch_connections = []
        for i in range(len(initial_context_embs)):
            conn_indices, conn_types = [], []
            if lctx_edge_type_id is not None and len(batch_lctx_neighbor_indices[i]) > 0:
                conn_indices.append(batch_lctx_neighbor_indices[i]); conn_types.append(torch.full_like(batch_lctx_neighbor_indices[i], lctx_edge_type_id))
            if goal_edge_type_id is not None and len(batch_goal_neighbor_indices[i]) > 0:
                conn_indices.append(batch_goal_neighbor_indices[i]); conn_types.append(torch.full_like(batch_goal_neighbor_indices[i], goal_edge_type_id))
            batch_connections.append((torch.cat(conn_indices), torch.cat(conn_types)) if conn_indices else (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)))
        for i, layer in enumerate(self.layers):
            context_residual, current_premise_embs = context_embs, premise_layer_embeddings[i]
            if self.gnn_layer_type == "gcn": context_embs = compute_ghost_node_embeddings_gcn(layer, current_premise_embs, None, context_embs.unbind(0), [con[0] for con in batch_connections])
            elif self.gnn_layer_type == "rgcn": context_embs = compute_ghost_node_embeddings_rgcn(layer, current_premise_embs, context_embs.unbind(0), batch_connections)
            elif self.gnn_layer_type == "graphsage": context_embs = compute_ghost_node_embeddings_graphsage(layer, current_premise_embs, context_embs.unbind(0), [con[0] for con in batch_connections])
            elif self.gnn_layer_type == "gat": context_embs = compute_ghost_node_embeddings_gat(layer, current_premise_embs, context_embs.unbind(0), [con[0] for con in batch_connections])
            elif self.gnn_layer_type == "rgat": context_embs = compute_ghost_node_embeddings_rgat(layer, current_premise_embs, context_embs.unbind(0), batch_connections)
            elif self.gnn_layer_type == "gin": context_embs = compute_ghost_node_embeddings_gin(layer, current_premise_embs, context_embs.unbind(0), [con[0] for con in batch_connections])
            if self.norm_layers is not None and i < len(self.norm_layers): context_embs = self.norm_layers[i](context_embs)
            if self.use_residual and context_embs.shape == context_residual.shape: context_embs = context_embs + context_residual
            elif self.use_residual and i == 0 and self.residual_projection is not None: context_embs = context_embs + self.residual_projection(initial_context_embs)
            if i < len(self.layers) - 1: context_embs = F.relu(context_embs)
        if self.final_projection is not None: context_embs = self.final_projection(context_embs)
        if self.hparams.postprocess_gnn_embeddings == "gnn": final_embs = F.normalize(context_embs, p=2, dim=1)
        else:
            orig_embs_norm, gnn_embs_norm = F.normalize(initial_context_embs, p=2, dim=1), F.normalize(context_embs, p=2, dim=1)
            if self.hparams.postprocess_gnn_embeddings == "concat": final_embs = F.normalize(torch.cat([gnn_embs_norm, orig_embs_norm], dim=1), p=2, dim=1)
            elif self.hparams.postprocess_gnn_embeddings == "gating":
                fused_embs = gnn_embs_norm + torch.sigmoid(self.gating_layer(torch.cat([gnn_embs_norm, orig_embs_norm], dim=1))) * orig_embs_norm
                final_embs = F.normalize(fused_embs, p=2, dim=1)
            else: raise ValueError(f"Unknown postprocessing option: {self.hparams.postprocess_gnn_embeddings}")
        return final_embs