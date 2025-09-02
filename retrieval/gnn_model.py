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
from typing import Dict, Any, List
from torch_geometric.nn import GCNConv

from common import get_optimizers, load_checkpoint

# TODO: test different layers

class GNNRetriever(pl.LightningModule):
    def __init__(
        self,
        feature_size: int,
        gnn_hidden_size: int,
        num_layers: int,
        lr: float,
        warmup_steps: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        assert num_layers >= 1, "Number of GNN layers must be at least 1."

        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(GCNConv(feature_size, feature_size))
        else:
            # Input layer
            self.layers.append(GCNConv(feature_size, gnn_hidden_size))
            # Hidden layers
            for _ in range(num_layers - 2):
                self.layers.append(GCNConv(gnn_hidden_size, gnn_hidden_size))
            # Output layer
            self.layers.append(GCNConv(gnn_hidden_size, feature_size))

        # For aggregating ghost node neighbors and combining with its own embedding
        self.aggregation_layer = nn.Linear(feature_size * 2, feature_size)

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
    ) -> torch.FloatTensor:
        # GNN propagation on the whole graph
        x = node_features
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            # Apply activation and dropout to all but the last layer
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        gnn_node_features = F.normalize(x, p=2, dim=1)

        # Ghost node aggregation
        final_context_embs = []
        for i, indices in enumerate(neighbor_indices):
            if len(indices) == 0:
                aggregated_neighbors_emb = torch.zeros_like(context_features[i])
            else:
                neighbor_embs = gnn_node_features[indices]
                aggregated_neighbors_emb = torch.mean(neighbor_embs, dim=0)

            initial_context_emb = context_features[i]
            combined_emb = torch.cat([initial_context_emb, aggregated_neighbors_emb])
            final_context_emb = self.aggregation_layer(combined_emb)
            final_context_embs.append(final_context_emb)

        context_emb = torch.stack(final_context_embs)
        context_emb = F.normalize(context_emb, dim=1)

        # In-batch contrastive loss using GNN-enhanced premise embeddings
        batch_size = context_emb.shape[0]
        pos_premise_emb = gnn_node_features[pos_premise_indices]
        neg_premise_embs_flat = [gnn_node_features[neg_idxs] for neg_idxs in neg_premises_indices]
        
        all_premise_embs = torch.cat([pos_premise_emb] + neg_premise_embs_flat, dim=0)

        similarity = torch.mm(context_emb, all_premise_embs.t())
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
        gnn_node_features: torch.FloatTensor,
        batch_neighbor_indices: List[torch.LongTensor],
    ) -> torch.FloatTensor:
        """
        Computes the final context embedding for a batch by combining initial text-based
        embeddings with aggregated neighbor embeddings (from before_premises).
        This method is intended for inference.
        """
        self.eval()  # Ensure the model is in evaluation mode

        final_context_embs = []
        # Loop through each item in the batch
        for i, indices in enumerate(batch_neighbor_indices):
            if len(indices) == 0:
                # If there are no neighbors, the aggregated embedding is a zero vector.
                aggregated_neighbors_emb = torch.zeros_like(initial_context_embs[i])
            else:
                # Ensure indices are on the correct device
                indices = indices.to(gnn_node_features.device)
                neighbor_embs = gnn_node_features[indices]
                aggregated_neighbors_emb = torch.mean(neighbor_embs, dim=0)

        
            current_initial_emb = initial_context_embs[i]
            combined_emb = torch.cat([current_initial_emb, aggregated_neighbors_emb])
            final_context_emb = self.aggregation_layer(combined_emb)
            final_context_embs.append(final_context_emb)

        context_emb = torch.stack(final_context_embs)
        return F.normalize(context_emb, dim=1)