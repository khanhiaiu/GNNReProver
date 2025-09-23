import numpy as np
import torch
from torch import Tensor
from torch_geometric.nn import RGCNConv, RGATConv, GATConv, GCNConv # type: ignore
from typing import Callable, Dict, Any, List, Literal, Tuple, Optional
import json
import copy

from tqdm import tqdm

from lightweight_graph.dataset import LightweightGraphDataset

# New imports for running Optuna
import optuna
import os

import multiprocessing as mp
# New import for command-line arguments
import argparse

import matplotlib.pyplot as plt
import seaborn as sns


def plot_score_distributions(
    scores: Tensor,
    targets: Tensor,
    file_path: Optional[str] = "score_distribution.png",
    show_plot: bool = False
) -> None:
    # Ensure tensors are on CPU and detached from the computation graph
    scores_np = scores.detach().cpu().flatten().numpy()
    targets_np = targets.detach().cpu().flatten().numpy().astype(bool)

    positive_scores = scores_np[targets_np]
    negative_scores = scores_np[~targets_np]

    if len(positive_scores) == 0:
        print(f"Warning: No positive samples found. Skipping plot generation.")
        return
    if len(negative_scores) == 0:
        print(f"Warning: No negative samples found. Skipping plot generation.")
        return

    # Sample an equal number of scores for a balanced plot
    n_pos = len(positive_scores)
    n_neg = len(negative_scores)
    sample_size = min(n_pos, n_neg, 50000) # Cap sample size for performance

    # Use replace=False to ensure we sample unique elements
    if n_pos > sample_size:
        sampled_positives = np.random.choice(positive_scores, size=sample_size, replace=False)
    else:
        sampled_positives = positive_scores

    if n_neg > sample_size:
        sampled_negatives = np.random.choice(negative_scores, size=sample_size, replace=False)
    else:
        sampled_negatives = negative_scores

    # Create the plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(
        sampled_negatives,
        bins=50,
        ax=ax,
        color='coral',
        label=f'Negative (Sampled N={len(sampled_negatives)})',
        kde=True,
        stat='density'
    )
    sns.histplot(
        sampled_positives,
        bins=50,
        ax=ax,
        color='steelblue',
        label=f'Positive (Sampled N={len(sampled_positives)})',
        kde=True,
        stat='density'
    )

    ax.set_title("Score Distributions for Positive vs. Negative Pairs (Balanced Samples)")
    ax.set_xlabel("Predicted Score (Logit)")
    ax.set_ylabel("Density")
    ax.legend()
    
    plt.tight_layout()
    
    if file_path:
        plt.savefig(file_path)
        print(f"Score distribution plot saved to {file_path}")
    
    if show_plot:
        plt.show()

    plt.close(fig)


class L2Norm(torch.nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__() # type: ignore

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.normalize(x, p=2, dim=-1)


class GNN(torch.nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(GNN, self).__init__() # type: ignore
        self.config = config
        
        input_size = config['input_size']
        hidden_size = config['hidden_size']

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        layer_type = config['layer_type']
        norm_type = config.get('normalization', 'none')

        for i in range(config['n_layers']):
            in_channels = input_size if i == 0 else hidden_size
            
            if layer_type == 'RGCN':
                conv = RGCNConv(in_channels, hidden_size, num_relations=config['n_relations'])
            elif layer_type == 'RGAT':
                assert hidden_size % config['heads'] == 0
                conv = RGATConv(in_channels, hidden_size // config['heads'], num_relations=config['n_relations'], heads=config['heads'])
            elif layer_type == 'GAT':
                assert hidden_size % config['heads'] == 0
                conv = GATConv(in_channels, hidden_size // config['heads'], heads=config['heads'])
            elif layer_type == 'GCN':
                conv = GCNConv(in_channels, hidden_size)
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
            self.convs.append(conv)

            if norm_type in ['batchnorm', 'batch']:
                norm = torch.nn.BatchNorm1d(hidden_size)
            elif norm_type in ['layernorm', 'layer']:
                norm = torch.nn.LayerNorm(hidden_size)
            elif norm_type in ['l2']:
                norm = L2Norm()
            else:
                norm = torch.nn.Identity()
            self.norms.append(norm)

        # Add a projection for the residual connection on the first layer if dimensions mismatch
        if self.config['residual'] and input_size != hidden_size:
            self.residual_projection = torch.nn.Linear(input_size, hidden_size)
        else:
            self.residual_projection = torch.nn.Identity()

    def forward(self, x: Tensor, edge_index: Tensor, edge_type: Tensor) -> Tensor:
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_res = x
            if isinstance(conv, (RGCNConv, RGATConv)):
                x = conv(x, edge_index, edge_type)
            else:
                x = conv(x, edge_index)

            if self.config['activation'] == 'relu':
                x = torch.relu(x)
            elif self.config['activation'] == 'gelu':
                x = torch.nn.functional.gelu(x)

            if self.config['dropout'] > 0.0:
                x = torch.nn.functional.dropout(x, p=self.config['dropout'], training=self.training)
            
            if self.config['residual']:
                if i == 0:
                    x_res = self.residual_projection(x_res)
                x = x + x_res

            x = norm(x)
                
        return x
    
class Scorer(torch.nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(Scorer, self).__init__() # type: ignore
        self.config = config
        self.heads: int = config.get('heads', 1)

        preprocess_config = config['preprocess']
        if preprocess_config["type"] == 'linear':
            self.preprocess = torch.nn.Linear(config['embedding_dim'], config['embedding_dim'])
        elif preprocess_config["type"] == 'nn':
            depth = preprocess_config['depth']
            layers: list[torch.nn.Module] = []
            hidden_size = config['embedding_dim']
            for _ in range(depth):
                layers.append(torch.nn.Linear(hidden_size, hidden_size))
                layers.append(torch.nn.ReLU())
                hidden_size = config['embedding_dim']
            self.preprocess = torch.nn.Sequential(*layers)
        elif preprocess_config["type"] == 'cosine':
            self.preprocess: Callable[[Tensor], Tensor] = lambda x: torch.nn.functional.normalize(x, p=2, dim=-1)
        elif preprocess_config["type"] == 'none':
            self.preprocess = lambda x: x
        else:
            raise ValueError(f"Unknown preprocess type: {preprocess_config['type']}")
        
        aggregation_type = config['aggregation']
        if aggregation_type == 'logsumexp':
            self.aggregate : Callable[[Tensor], Tensor] = lambda x: torch.logsumexp(x, dim=1)
        elif aggregation_type == 'mean':
            self.aggregate : Callable[[Tensor], Tensor] = lambda x: torch.mean(x, dim=1)
        elif aggregation_type == 'max':
            self.aggregate : Callable[[Tensor], Tensor] = lambda x: torch.max(x, dim=1).values
        elif aggregation_type == 'gated':
            self.gate = torch.nn.Linear(config['embedding_dim'], 1)
            self.aggregate : Callable[[Tensor], Tensor] = lambda x: torch.sum(torch.sigmoid(self.gate(x)) * x, dim=1)
        else:
            # Fallback for old configs that may not have this parameter
            print("Warning: 'aggregation' not specified in scorer config. Defaulting to 'mean'.")
            self.aggregate : Callable[[Tensor], Tensor] = lambda x: torch.mean(x, dim=1)


    def forward(self, context_embeddings: Tensor, premise_embeddings: Tensor) -> Tensor:
        b, e = context_embeddings.shape
        p, e_prime = premise_embeddings.shape
        assert e == e_prime, f"Embedding dimensions must match: {e} != {e_prime}"

        context_embeddings = self.preprocess(context_embeddings) # (batch_size, embedding_dim)
        premise_embeddings = self.preprocess(premise_embeddings) # (n_premises, embedding_dim)

        if self.heads > 1:
            context_embeddings = context_embeddings.view(b, self.heads, e // self.heads) # (batch_size, heads, head_dim)
            premise_embeddings = premise_embeddings.view(p, self.heads, e // self.heads) # (n_premises, heads, head_dim)
            scores = torch.einsum('bhd, phd -> bhp', context_embeddings, premise_embeddings) # (batch_size, heads, n_premises)
            scores = self.aggregate(scores).squeeze(1) # (batch_size, n_premises)
        else:
            scores = torch.matmul(context_embeddings, premise_embeddings.t()) # (batch_size, n_premises)

        assert scores.shape == (b, p), f"Scores shape must be (batch_size, n_premises): {scores.shape}"
        return scores


class LossFunction:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.weigh_by_class: bool = config["weigh_by_class"]
        self.loss_function: Literal["bce", "mse", "info_nce"] = config["loss_function"]

        if self.loss_function == "info_nce":
            assert self.weigh_by_class == False, "Class weighting not supported for InfoNCE loss"
            self.temperature = config.get("temperature", 1.0)

    def compute(self, logits: Tensor, targets: Tensor) -> Tensor:
        weights = torch.ones_like(logits, dtype=torch.float)
        if self.weigh_by_class:
            n_positives = targets.sum().item()
            n_negatives = targets.numel() - n_positives
            total = targets.numel()
            
            if n_positives > 0:
                weights[targets] = total / (2.0 * n_positives)
            if n_negatives > 0:
                weights[~targets] = total / (2.0 * n_negatives)

        # calculate unweighted, unreduced loss
        targets_float = targets.float()
        if self.loss_function == "bce":
            unweighted_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets_float, reduction='none')
        elif self.loss_function == "mse":
            unweighted_loss = torch.nn.functional.mse_loss(torch.sigmoid(logits), targets_float, reduction='none')
        elif self.loss_function == "info_nce":
            logits = logits / self.temperature
            log_probs = torch.nn.functional.log_softmax(logits, dim=1)
            positive_log_probs = log_probs[targets]
            loss = -positive_log_probs
            return loss.mean()
        else:
            raise ValueError(f"Unknown loss function: {self.loss_function}")
        
        weighted_loss = (unweighted_loss * weights).mean()
        return weighted_loss

class NegativeSampler:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sampler: Literal["random", "hard_top_k", "hard_sampling", "all"] = config["sampler"]
        self.scorer = Scorer(config["scorer"])

    def forward(self, context_embeddings: Tensor, premise_embeddings: Tensor, retrieved_labels : Tensor) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        n_contexts = context_embeddings.shape[0]
        n_premises = premise_embeddings.shape[0]

        label_premise_indices = retrieved_labels[1]
        if self.sampler == "random":
            n_negatives = self.config["n_negatives_per_context"] * n_contexts
            all_indices = torch.arange(n_premises, device=premise_embeddings.device)
            negative_premise_indices = all_indices[torch.randint(0, n_premises, (n_negatives,), device=premise_embeddings.device)]
            premise_indices = torch.cat([label_premise_indices, negative_premise_indices], dim=0)
        elif self.sampler == "all":
            premise_indices = torch.arange(n_premises, device=premise_embeddings.device)
        elif self.sampler == "hard_top_k" or self.sampler == "hard_sampling":
            with torch.no_grad():
                all_scores = self.scorer(context_embeddings, premise_embeddings) 
            k = self.config["n_negatives_per_context"]
            if self.sampler == "hard_top_k":
                _, hard_negative_indices = torch.topk(all_scores, k=k, dim=1)
            elif self.sampler == "hard_sampling":
                probs = torch.softmax(all_scores, dim=1) 
                hard_negative_indices = torch.multinomial(probs, num_samples=k, replacement=False)
            else:
                raise RuntimeError("Unreachable")
            hard_negative_indices = hard_negative_indices.flatten()
            premise_indices = torch.cat([label_premise_indices, hard_negative_indices], dim=0)
        else:
            raise ValueError(f"Unknown sampler: {self.sampler}")
        
        premise_indices = torch.unique(premise_indices) 
        selected_premise_embeddings = premise_embeddings[premise_indices]
        
        scores = self.scorer(context_embeddings, selected_premise_embeddings)

        metrics = {}
        if self.sampler in ["all", "hard_sampling", "hard_top_k"]:
            with torch.no_grad():
                if "all_scores" not in locals():
                    all_scores = self.scorer(context_embeddings, premise_embeddings)
                metrics = calculate_metrics_batch(all_scores, retrieved_labels)

        targets = torch.zeros_like(scores, dtype=torch.bool)
        context_indices = retrieved_labels[0]
        premise_indices_correct = retrieved_labels[1]
        
        assert torch.all(torch.isin(premise_indices_correct, premise_indices)), "Some correct premise indices are not in the selected premise indices" # type: ignore
        premise_positions = torch.searchsorted(premise_indices, premise_indices_correct) # type: ignore

        targets[context_indices, premise_positions] = True

        return scores, targets, metrics


def calculate_metrics_batch(all_premise_scores_batch: Tensor, batch_labels : Tensor) -> Dict[str, Tensor]:
    targets = torch.zeros_like(all_premise_scores_batch, dtype=torch.bool)
    context_indices = batch_labels[0]
    premise_indices_correct = batch_labels[1]
    targets[context_indices, premise_indices_correct] = True

    context_premise_count = targets.sum(dim=1) 

    valid_contexts_mask = context_premise_count > 0
    if not valid_contexts_mask.any():
        # Handle cases where a batch might not have any positive labels (e.g., during overfitting test)
        return {"R@1": torch.tensor(0.0), "R@10": torch.tensor(0.0), "MRR": torch.tensor(0.0)}

    metrics: Dict[str, Any] = {}
    for k in [1, 10]:
        valid_scores = all_premise_scores_batch[valid_contexts_mask]
        valid_targets = targets[valid_contexts_mask]
        valid_context_premise_count = context_premise_count[valid_contexts_mask]

        topk_indices = torch.topk(valid_scores, k=k, dim=1).indices 
        
        hits_at_k = valid_targets.gather(1, topk_indices).sum(dim=-1) 
        single_R_at_k = hits_at_k / valid_context_premise_count 
        metrics[f"R@{k}"] = single_R_at_k
    
    ranks = torch.argsort(torch.argsort(all_premise_scores_batch, dim=1, descending=True), dim=1) + 1 
    
    ranks_of_correct_only = torch.where(targets, ranks.float(), torch.inf)
    
    min_ranks, _ = torch.min(ranks_of_correct_only, dim=1) 
    
    reciprocal_ranks = 1.0 / min_ranks
    
    single_MRR = reciprocal_ranks[valid_contexts_mask]
    metrics["MRR"] = single_MRR

    return metrics


def edge_dropout(
    edge_index: Tensor, edge_type: Tensor, p: float, training: bool = True
) -> Tuple[Tensor, Tensor]:
    """Randomly drops edges from the edge_index and edge_type."""
    if p == 0.0 or not training:
        return edge_index, edge_type
    
    keep_prob = 1 - p
    mask = torch.rand(edge_index.size(1), device=edge_index.device) < keep_prob
    
    return edge_index[:, mask], edge_type[mask]


def batchify_dataset(dataset: LightweightGraphDataset, split_indices: Tensor, batch_size: int):
    n_contexts = dataset.context_embeddings.shape[0]

    premise_to_context_edge_mask = torch.isin(dataset.context_edge_index[1], split_indices)
    premise_to_context_edge_index = dataset.context_edge_index[:, premise_to_context_edge_mask]
    premise_to_context_edge_type = dataset.context_edge_attr[premise_to_context_edge_mask]

    for start in range(0, len(split_indices), batch_size):
        end = min(start + batch_size, len(split_indices))
        batch_global_indices = split_indices[start:end]

        batch_global_to_local_map = torch.full((n_contexts,), -1, dtype=torch.long, device=split_indices.device)
        batch_global_to_local_map[batch_global_indices] = torch.arange(len(batch_global_indices), device=split_indices.device)
        batch_context_embeddings = dataset.context_embeddings[batch_global_indices]
        
        batch_context_file_indices = dataset.context_to_file_idx_map[batch_global_indices]
        batch_context_theorem_pos = dataset.context_theorem_pos[batch_global_indices]
        
        batch_edge_mask = torch.isin(premise_to_context_edge_index[1], batch_global_indices)
        batch_premise_to_context_edge_index_global = premise_to_context_edge_index[:, batch_edge_mask].clone()
        batch_premise_to_context_edge_type = premise_to_context_edge_type[batch_edge_mask]
        batch_premise_to_context_edge_index_global[1] = batch_global_to_local_map[batch_premise_to_context_edge_index_global[1]]

        retrieved_labels_mask = torch.isin(dataset.context_premise_labels[0], batch_global_indices)
        retrieved_labels_global = dataset.context_premise_labels[:, retrieved_labels_mask]
        retrieved_labels = retrieved_labels_global.clone()
        retrieved_labels[0] = batch_global_to_local_map[retrieved_labels[0]]

        data : Dict[str, Any] = {
            "context_embeddings": batch_context_embeddings.float(),
            "context_to_file_idx_map": batch_context_file_indices,
            "context_theorem_pos": batch_context_theorem_pos,
            "premise_embeddings": dataset.premise_embeddings.float(),

            "premise_edge_index": dataset.premise_edge_index,
            "premise_edge_type": dataset.premise_edge_attr,

            "premise_to_context_edge_index": batch_premise_to_context_edge_index_global,
            "premise_to_context_edge_type": batch_premise_to_context_edge_type,

            "retrieved_labels" : retrieved_labels
        }
        yield data


class GNNRetrievalModel(torch.nn.Module):
    def __init__(self, config : Dict[str, Any], data_config: Dict[str, Any]):
        super(GNNRetrievalModel, self).__init__() # type: ignore
        self.config = config
        self.use_random_embeddings: bool = config["use_random_embeddings"]
        self.embedding_size: int = data_config["embedding_dim"]
        
        gnn_config = config["gnn"].copy()
        if self.use_random_embeddings:
            gnn_input_dim = gnn_config["hidden_size"]
        else:
            gnn_input_dim = self.embedding_size
        gnn_config['input_size'] = gnn_input_dim

        self.context_gnn = GNN(gnn_config)
        
        if self.config["separate_premise_GNN"]:
            self.premise_gnn = GNN(gnn_config)
            if self.use_random_embeddings:
                hidden_size = gnn_config["hidden_size"]
                self.random_initial_premise_embeddings = torch.nn.Parameter(torch.randn(data_config["n_premises"], hidden_size))
                self.random_initial_premise_embeddings_for_context = torch.nn.Parameter(torch.randn(data_config["n_premises"], hidden_size))
        else:
            self.premise_gnn = self.context_gnn
            if self.use_random_embeddings:
                hidden_size = gnn_config["hidden_size"]
                self.random_initial_premise_embeddings = torch.nn.Parameter(torch.randn(data_config["n_premises"], hidden_size))
        
        self.loss_function = LossFunction(config["loss"])
        self.negative_sampler = NegativeSampler(config["negative_sampler"])
        self.scorer = self.negative_sampler.scorer

        optimizer_config = config["optimizer"]
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=optimizer_config["lr"], weight_decay=optimizer_config.get("weight_decay", 0.0))

    def forward(self, batch_data: Dict[str, Any]) -> Tuple[Tensor, Tensor]:
        initial_premise_embeddings = batch_data["premise_embeddings"]
        initial_batch_context_embeddings = batch_data["context_embeddings"]

        n_premises = initial_premise_embeddings.shape[0]
        n_contexts = initial_batch_context_embeddings.shape[0]
        hidden_size = self.config["gnn"]["hidden_size"]

        premise_edge_index = batch_data["premise_edge_index"]
        premise_edge_type = batch_data["premise_edge_type"]

        premise_to_context_edge_index = batch_data["premise_to_context_edge_index"].clone()
        premise_to_context_edge_index[1] += n_premises 

        premise_to_context_edge_type = batch_data["premise_to_context_edge_type"]

        all_edge_index = torch.cat([premise_edge_index, premise_to_context_edge_index], dim=1)
        all_edge_type = torch.cat([premise_edge_type, premise_to_context_edge_type], dim=0)
        
        edge_dropout_p = self.config['gnn'].get('edge_dropout', 0.0)

        if self.config["separate_premise_GNN"]:
            premise_ei_train, premise_et_train = edge_dropout(
                premise_edge_index, premise_edge_type, p=edge_dropout_p, training=self.training
            )
            all_ei_train, all_et_train = edge_dropout(
                all_edge_index, all_edge_type, p=edge_dropout_p, training=self.training
            )

            if self.use_random_embeddings:
                premise_embeddings = self.premise_gnn(self.random_initial_premise_embeddings, premise_ei_train, premise_et_train)
                dummy_context_embeddings = torch.zeros(n_contexts, hidden_size, device=self.random_initial_premise_embeddings_for_context.device)
                all_random_embeddings_for_context = torch.cat([self.random_initial_premise_embeddings_for_context, dummy_context_embeddings], dim=0)
                all_embeddings_for_context = self.context_gnn(all_random_embeddings_for_context, all_ei_train, all_et_train)
                context_embeddings = all_embeddings_for_context[n_premises:]
            else:
                premise_embeddings = self.premise_gnn(initial_premise_embeddings, premise_ei_train, premise_et_train)
                all_initial_embeddings_for_context = torch.cat([initial_premise_embeddings, initial_batch_context_embeddings], dim=0)
                all_embeddings_for_context = self.context_gnn(all_initial_embeddings_for_context, all_ei_train, all_et_train)
                context_embeddings = all_embeddings_for_context[n_premises:]
        else: # not separate_premise_GNN
            all_ei_train, all_et_train = edge_dropout(
                all_edge_index, all_edge_type, p=edge_dropout_p, training=self.training
            )
            if self.use_random_embeddings:
                dummy_context_embeddings = torch.zeros(n_contexts, hidden_size, device=self.random_initial_premise_embeddings.device)
                all_random_embeddings = torch.cat([self.random_initial_premise_embeddings, dummy_context_embeddings], dim=0)
                all_embeddings = self.context_gnn(all_random_embeddings, all_ei_train, all_et_train)
                premise_embeddings = all_embeddings[:n_premises]
                context_embeddings = all_embeddings[n_premises:]
            else:
                all_initial_embeddings = torch.cat([initial_premise_embeddings, initial_batch_context_embeddings], dim=0)
                all_embeddings = self.context_gnn(all_initial_embeddings, all_ei_train, all_et_train)
                premise_embeddings = all_embeddings[:n_premises]
                context_embeddings = all_embeddings[n_premises:]

        assert context_embeddings.shape[0] == n_contexts
        assert premise_embeddings.shape[0] == n_premises

        return context_embeddings, premise_embeddings
    
    def predict_batch(self, batch_data: Dict[str, Any]):
        context_embeddings, premise_embeddings = self.forward(batch_data)
        return self.scorer.forward(context_embeddings, premise_embeddings)

    def compute_loss(self, batch_data: Dict[str, Any]):
        context_embeddings, premise_embeddings = self.forward(batch_data)
        retrieved_labels = batch_data["retrieved_labels"]
        logits, targets, metrics = self.negative_sampler.forward(context_embeddings, premise_embeddings, retrieved_labels)
        loss = self.loss_function.compute(logits, targets)

        if self.config.get("plot_score_distributions", False):
            with torch.no_grad():
                plot_score_distributions(
                    scores=logits,
                    targets=targets,
                    file_path="score_distribution.png",
                    show_plot=False
                )
        return loss, metrics

    def train_batch(self, batch_data: Dict[str, Any], should_updated: bool) -> Dict[str, Any]:
        self.train()
        
        loss, batch_metrics_tensors = self.compute_loss(batch_data)
        memory = torch.cuda.memory_allocated(device=loss.device) / (1024 ** 3)
        loss.backward() # type: ignore
        if should_updated:
            self.optimizer.step() # type: ignore
            self.optimizer.zero_grad()

        batch_metrics = {k: v.mean().item() for k, v in batch_metrics_tensors.items()}
        batch_metrics["loss"] = loss.item()
        batch_metrics["memory"] = memory
        return batch_metrics
    
    def train_epoch(self, dataset: LightweightGraphDataset, config: Dict[str, Any]) -> None:
        self.train()
        self.optimizer.zero_grad()
        gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
        
        train_split_indices = dataset.train_mask.nonzero(as_tuple=False).view(-1)
        train_generator = batchify_dataset(dataset, train_split_indices, config["batch_size"])
        
        num_batches = (len(train_split_indices) + config["batch_size"] - 1) // config["batch_size"]
        pbar = tqdm(enumerate(train_generator), total=num_batches, desc="Training")

        all_batch_metrics : Dict[str, List[Any]] = {}
        for i, batch_data in pbar:
            should_update = ((i + 1) % gradient_accumulation_steps == 0) or (i + 1 == num_batches)
            batch_metrics = self.train_batch(batch_data, should_updated=should_update)
            for key, value in batch_metrics.items():
                if key not in all_batch_metrics:
                    all_batch_metrics[key] = []
                all_batch_metrics[key].append(value)
            
            mean_metrics = {key: sum(values) / len(values) for key, values in all_batch_metrics.items()}
            pbar.set_postfix({key: f"{value:.4f}" for key, value in mean_metrics.items()})

    def evaluate_batch(self, batch_data : Dict[str, Any], dataset : LightweightGraphDataset) -> Dict[str, Tensor]:
        self.eval()
        with torch.no_grad():
            context_embeddings, premise_embeddings = self.forward(batch_data)
            scores = self.scorer(context_embeddings, premise_embeddings)
            device = scores.device

            batch_context_file_indices = batch_data["context_to_file_idx_map"]
            batch_context_theorem_pos = batch_data["context_theorem_pos"]
            premise_to_file_idx_map = dataset.premise_to_file_idx_map
            file_dependency_edge_index = dataset.file_dependency_edge_index
            premise_end_pos = dataset.premise_pos[:, 2:]

            batch_same_file_mask = batch_context_file_indices.unsqueeze(1) == premise_to_file_idx_map.unsqueeze(0)

            batch_before_pos_mask = (premise_end_pos.unsqueeze(0)[:, :, 0] < batch_context_theorem_pos.unsqueeze(1)[:, :, 0]) | \
                                    ((premise_end_pos.unsqueeze(0)[:, :, 0] == batch_context_theorem_pos.unsqueeze(1)[:, :, 0]) & \
                                     (premise_end_pos.unsqueeze(0)[:, :, 1] <= batch_context_theorem_pos.unsqueeze(1)[:, :, 1]))
            
            in_file_accessible_mask = batch_same_file_mask & batch_before_pos_mask

            n_files = len(dataset.file_idx_to_path_map)
            file_dependency_adj = torch.zeros((n_files, n_files), dtype=torch.bool, device=device)
            src, dst = file_dependency_edge_index
            file_dependency_adj[src, dst] = True
            
            imported_mask = file_dependency_adj[batch_context_file_indices][:, premise_to_file_idx_map]

            accessible_mask = in_file_accessible_mask | imported_mask
            scores.masked_fill_(~accessible_mask, -torch.inf)
            
            retrieved_labels = batch_data["retrieved_labels"]
            metrics = calculate_metrics_batch(scores, retrieved_labels)
            return metrics

    def evaluate(self, dataset: LightweightGraphDataset, split: str, batch_size: int) -> Dict[str, float]:
        self.eval()
        with torch.no_grad():
            mask = getattr(dataset, f"{split}_mask", None)
            if mask is None: raise ValueError(f"Invalid split: {split}")
            split_indices = mask.nonzero(as_tuple=False).view(-1)

            eval_generator = batchify_dataset(dataset, split_indices, batch_size)

            metrics_data : list[Dict[str, Tensor]] = []
            pbar = tqdm(eval_generator, desc=f"Evaluating on {split} split")
            for batch_data in pbar:
                metrics_data.append(self.evaluate_batch(batch_data, dataset))

        all_metrics_lists: Dict[str, list[Tensor]] = {}
        for m in metrics_data:
            for key, value in m.items():
                if key not in all_metrics_lists:
                    all_metrics_lists[key] = []
                all_metrics_lists[key].append(value)

        all_metrics_tensor = {key: torch.cat(value, dim=0) for key, value in all_metrics_lists.items()}
        
        metrics : Dict[str, float] = {}
        for key, value in all_metrics_tensor.items():
            metrics[key] = value.mean().item()

        return metrics


class EnsembleGNNRetrievalModel(torch.nn.Module):
    """A wrapper to handle evaluation for an ensemble of GNNRetrievalModels."""
    def __init__(self, config: Dict[str, Any], data_config: Dict[str, Any], checkpoint_paths: List[str], device: str):
        super().__init__()
        self.models = torch.nn.ModuleList()
        self.device = device
        
        print(f"Loading {len(checkpoint_paths)} models for the ensemble...")
        for path in checkpoint_paths:
            # Create a model instance with the same config
            model = GNNRetrievalModel(config, data_config)
            # Load the saved state dictionary
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device)
            model.eval() # Set to evaluation mode
            self.models.append(model)
        print("Ensemble models loaded successfully.")

    @torch.no_grad()
    def evaluate_batch(self, batch_data: Dict[str, Any], dataset: LightweightGraphDataset) -> Dict[str, Tensor]:
        all_scores = []
        # Get scores from each model in the ensemble
        for model in self.models:
            # We only need the final scores, so we can call a simplified forward pass
            context_embeddings, premise_embeddings = model.forward(batch_data)
            scores = model.scorer(context_embeddings, premise_embeddings)
            all_scores.append(scores)

        # Average the scores across the ensemble
        ensemble_scores = torch.stack(all_scores).mean(dim=0)

        # The rest of this function is copied directly from GNNRetrievalModel.evaluate_batch
        # to apply the accessibility mask to the final averaged scores.
        device = ensemble_scores.device

        batch_context_file_indices = batch_data["context_to_file_idx_map"]
        batch_context_theorem_pos = batch_data["context_theorem_pos"]
        premise_to_file_idx_map = dataset.premise_to_file_idx_map
        file_dependency_edge_index = dataset.file_dependency_edge_index
        premise_end_pos = dataset.premise_pos[:, 2:]

        batch_same_file_mask = batch_context_file_indices.unsqueeze(1) == premise_to_file_idx_map.unsqueeze(0)
        batch_before_pos_mask = (premise_end_pos.unsqueeze(0)[:, :, 0] < batch_context_theorem_pos.unsqueeze(1)[:, :, 0]) | \
                                ((premise_end_pos.unsqueeze(0)[:, :, 0] == batch_context_theorem_pos.unsqueeze(1)[:, :, 0]) & \
                                (premise_end_pos.unsqueeze(0)[:, :, 1] <= batch_context_theorem_pos.unsqueeze(1)[:, :, 1]))
        in_file_accessible_mask = batch_same_file_mask & batch_before_pos_mask

        n_files = len(dataset.file_idx_to_path_map)
        file_dependency_adj = torch.zeros((n_files, n_files), dtype=torch.bool, device=device)
        src, dst = file_dependency_edge_index
        file_dependency_adj[src, dst] = True
        imported_mask = file_dependency_adj[batch_context_file_indices][:, premise_to_file_idx_map]
        
        accessible_mask = in_file_accessible_mask | imported_mask
        ensemble_scores.masked_fill_(~accessible_mask, -torch.inf)
        
        retrieved_labels = batch_data["retrieved_labels"]
        metrics = calculate_metrics_batch(ensemble_scores, retrieved_labels)
        return metrics

    def evaluate(self, dataset: LightweightGraphDataset, split: str, batch_size: int) -> Dict[str, float]:
        # This function is identical to GNNRetrievalModel.evaluate
        self.eval()
        with torch.no_grad():
            mask = getattr(dataset, f"{split}_mask", None)
            if mask is None: raise ValueError(f"Invalid split: {split}")
            split_indices = mask.nonzero(as_tuple=False).view(-1)
            
            eval_generator = batchify_dataset(dataset, split_indices, batch_size)
            metrics_data: list[Dict[str, Tensor]] = []
            pbar = tqdm(eval_generator, desc=f"Evaluating Ensemble on {split} split")
            for batch_data in pbar:
                metrics_data.append(self.evaluate_batch(batch_data, dataset))
            
        all_metrics_lists: Dict[str, list[Tensor]] = {}
        for m in metrics_data:
            for key, value in m.items():
                if key not in all_metrics_lists: all_metrics_lists[key] = []
                all_metrics_lists[key].append(value)
        
        all_metrics_tensor = {key: torch.cat(value, dim=0) for key, value in all_metrics_lists.items()}
        metrics = {key: value.mean().item() for key, value in all_metrics_tensor.items()}
        return metrics


def get_data_configs(dataset: LightweightGraphDataset) -> Dict[str, Any]:
    data_config = {
        "n_premises": dataset.premise_embeddings.shape[0],
        "n_contexts": dataset.context_embeddings.shape[0],
        "embedding_dim": dataset.premise_embeddings.shape[1],
    }
    return data_config


def create_config_from_params(params: Dict[str, Any], n_relations: int, data_config: Dict[str, Any]) -> Dict[str, Any]:
    gnn_hidden_size = params['gnn_head_dim'] * params['gnn_heads']
    
    if params['loss_function'] == 'info_nce':
        weigh_by_class = False
    else:
        weigh_by_class = params.get('weigh_by_class', False)

    config = {
        'use_random_embeddings': params['use_random_embeddings'],
        'separate_premise_GNN': params['separate_premise_GNN'],
        'gnn': {
            'layer_type': params['layer_type'],
            'n_layers': params['n_layers'],
            'hidden_size': gnn_hidden_size,
            'n_relations': n_relations,
            'activation': params['activation'],
            'dropout': params['dropout'],
            'edge_dropout': params.get('edge_dropout', 0.0),
            'residual': params['residual'],
            'normalization': params['normalization'],
            'heads': params['gnn_heads'],
        },
        'negative_sampler': {
            'sampler': params['sampler'],
            'scorer': {
                'preprocess': {
                    'type': params['scorer_preprocess_type'],
                    'depth': params.get('scorer_nn_depth', 1),
                },
                'heads': params['scorer_heads'],
                'aggregation': params.get('scorer_aggregation', 'mean'), 
                'embedding_dim': gnn_hidden_size,
            }
        },
        'loss': {
            'loss_function': params['loss_function'],
            'weigh_by_class': weigh_by_class,
            'temperature': params.get('temperature', 1.0),
        },
        'optimizer': {
            'lr': params['lr'],
            'weight_decay': params['weight_decay'],
        },
        'training': {
            'batch_size': params['batch_size'],
            'gradient_accumulation_steps': 2
        },
        'evaluation': {
            'batch_size': 2048
        },
    }
    return config

# ===== OPTUNA SCRIPT FOR REGULARIZATION SEARCH ================================

FIXED_PARAMS = {
    "use_random_embeddings": False, "separate_premise_GNN": True, "layer_type": "RGCN",
    "n_layers": 2, "gnn_head_dim": 128, "gnn_heads": 8, "activation": "relu",
    "residual": True, "normalization": "l2", "sampler": "all",
    "scorer_preprocess_type": "cosine", "scorer_nn_depth": 1, "scorer_heads": 2,
    "scorer_aggregation": "mean", "loss_function": "info_nce",
    "temperature": 0.013777448539592157, "lr": 0.004997184962551209, "batch_size": 1024,
} # NOTE: The parameters were automatically tuned

SAVE_DIR = "lightweight_graph/data_random_updated"
N_EPOCHS_PER_TRIAL = 50 

# These will be initialized within each worker process
dataset: Optional[LightweightGraphDataset] = None
data_config: Optional[Dict[str, Any]] = None
N_RELATIONS: Optional[int] = None
DEVICE: Optional[str] = None

def objective(trial: optuna.Trial) -> float:
    global data_config, N_RELATIONS, DEVICE # Access globals set by the worker
    assert data_config is not None and N_RELATIONS is not None and DEVICE is not None

    trial_params = FIXED_PARAMS.copy()
    trial_params['dropout'] = trial.suggest_float("dropout", 1e-5, 0.5, log=True)
    trial_params['weight_decay'] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    trial_params['edge_dropout'] = trial.suggest_float("edge_dropout", 0.0, 0.3)
    
    config = create_config_from_params(trial_params, N_RELATIONS, data_config)
    
    pid = os.getpid()
    print(f"\n--- [PID {pid}] Starting Trial {trial.number} ---")
    print(f"  [PID {pid}] Dropout: {config['gnn']['dropout']:.6f}, Weight Decay: {config['optimizer']['weight_decay']:.6f}, Edge Dropout: {config['gnn']['edge_dropout']:.6f}")
    
    torch.manual_seed(42)
    model = GNNRetrievalModel(config=config, data_config=data_config).to(DEVICE)
    
    best_trial_score = -1.0
    
    for epoch in range(1, N_EPOCHS_PER_TRIAL + 1):
        model.train_epoch(dataset, config=config['training'])
        val_metrics = model.evaluate(dataset, 'val', batch_size=config['evaluation']['batch_size'])
        validation_score = val_metrics.get("R@10", 0.0)
        best_trial_score = max(best_trial_score, validation_score)
        
        trial.report(validation_score, epoch)
        
        print(f"  [PID {pid}] Epoch {epoch}/{N_EPOCHS_PER_TRIAL} - Val R@10: {validation_score:.4f}")

        if trial.should_prune():
            print(f"[PID {pid}] Trial {trial.number} pruned at epoch {epoch}.")
            raise optuna.exceptions.TrialPruned()

    print(f"--- [PID {pid}] Trial {trial.number} finished. Best Val R@10: {best_trial_score:.4f} ---")
    return best_trial_score


def run_worker(gpu_id: int, study_name: str, db_path: str, n_trials_per_worker: int):
    # Declare which global variables this worker will initialize
    global dataset, data_config, N_RELATIONS, DEVICE
    
    # Use torch.cuda.set_device() to select the GPU for this process.
    # This must be done at the beginning of the worker function.
    torch.cuda.set_device(gpu_id)
    DEVICE = "cuda" # This will now correctly refer to the selected GPU
    
    # 2. Load dataset and configs *within this new process*
    print(f"[Worker on GPU {gpu_id}, PID {os.getpid()}] Loading dataset...")
    dataset = LightweightGraphDataset.load_or_create(save_dir=SAVE_DIR)
    data_config = get_data_configs(dataset)
    
    max_premise_rel = dataset.premise_edge_attr.max().item() if dataset.premise_edge_attr.numel() > 0 else -1
    max_context_rel = dataset.context_edge_attr.max().item() if dataset.context_edge_attr.numel() > 0 else -1
    N_RELATIONS = max(max_premise_rel, max_context_rel) + 1
    print(f"[Worker on GPU {gpu_id}] Dataset loaded with {N_RELATIONS} relations. Moving to device '{DEVICE}'...")
    
    dataset.to(DEVICE)
    print(f"[Worker on GPU {gpu_id}] Dataset moved. Starting optimization.")

    # 3. Connect to the shared Optuna study
    storage = optuna.storages.RDBStorage(url=f"sqlite:///{db_path}")
    study = optuna.load_study(study_name=study_name, storage=storage)

    # 4. Run the optimization loop
    study.optimize(objective, n_trials=n_trials_per_worker)
    
    print(f"[Worker on GPU {gpu_id}] Finished its assigned trials.")


def train_ensemble_member_worker(
    gpu_id: int, 
    model_index: int, 
    config: Dict[str, Any], 
    save_path: str
):
    """Trains a single model for the ensemble on a specific GPU."""
    # This setup is similar to the Optuna worker
    torch.cuda.set_device(gpu_id)
    device = "cuda"
    
    # Use a different seed for each model to ensure they are unique
    torch.manual_seed(42 + model_index) 
    
    print(f"[Ensemble Trainer {model_index} on GPU {gpu_id}, PID {os.getpid()}] Loading dataset...")
    # Load dataset and configs within the new process
    dataset_member = LightweightGraphDataset.load_or_create(save_dir=SAVE_DIR)
    dataset_member.to(device)
    data_config_member = get_data_configs(dataset_member)
    
    model = GNNRetrievalModel(config=config, data_config=data_config_member).to(device)
    
    print(f"[Ensemble Trainer {model_index} on GPU {gpu_id}] Starting training...")
    # Using the same number of epochs as one Optuna trial
    for epoch in range(1, N_EPOCHS_PER_TRIAL + 1):
        model.train_epoch(dataset_member, config=config['training'])
        # Optional: You could add validation here and save the best model, but for simplicity, we train for a fixed number of epochs.
        print(f"[Ensemble Trainer {model_index} on GPU {gpu_id}] Epoch {epoch}/{N_EPOCHS_PER_TRIAL} complete.")

    print(f"[Ensemble Trainer {model_index} on GPU {gpu_id}] Training finished. Saving model to {save_path}")
    torch.save(model.state_dict(), save_path)

# ==============================================================================
# ===== NEW SCRIPT MODES AND HELPER FUNCTIONS ==================================
# ==============================================================================

def load_best_config(args: argparse.Namespace, dataset: LightweightGraphDataset) -> Dict[str, Any]:
    """Loads the best trial from an Optuna study and creates a model config."""
    print(f"Loading best trial from study '{args.study_name}' in '{args.db_path}'...")
    storage = optuna.storages.RDBStorage(url=f"sqlite:///{args.db_path}")
    study = optuna.load_study(study_name=args.study_name, storage=storage)
    best_trial = study.best_trial

    print("\n--- Best Trial ---")
    print(f"  Value (Best R@10): {best_trial.value:.6f}")
    print("  Parameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    data_config = get_data_configs(dataset)
    max_premise_rel = dataset.premise_edge_attr.max().item() if dataset.premise_edge_attr.numel() > 0 else -1
    max_context_rel = dataset.context_edge_attr.max().item() if dataset.context_edge_attr.numel() > 0 else -1
    n_relations = max(max_premise_rel, max_context_rel) + 1
    
    best_params = FIXED_PARAMS.copy()
    best_params.update(best_trial.params)
    
    final_config = create_config_from_params(best_params, n_relations, data_config)
    return final_config

def run_hyperparameter_search(args: argparse.Namespace):
    """Executes the Optuna hyperparameter search."""
    print(f"\nStarting Optuna study '{args.study_name}' in parallel on {len(args.gpu_ids)} GPUs: {args.gpu_ids}")
    print(f"Database will be saved to '{args.db_path}'")
    print(f"Running a total of {args.n_trials} trials, each for up to {N_EPOCHS_PER_TRIAL} epochs.")

    storage = optuna.storages.RDBStorage(url=f"sqlite:///{args.db_path}")
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )

    n_jobs = len(args.gpu_ids)
    trials_per_worker = [args.n_trials // n_jobs] * n_jobs
    for i in range(args.n_trials % n_jobs):
        trials_per_worker[i] += 1

    processes = []
    try:
        for i in range(n_jobs):
            gpu_id = args.gpu_ids[i]
            n_worker_trials = trials_per_worker[i]
            if n_worker_trials == 0: continue
            p = mp.Process(target=run_worker, args=(gpu_id, args.study_name, args.db_path, n_worker_trials))
            p.start()
            processes.append(p)
        for p in processes: p.join()
    except KeyboardInterrupt:
        print("\nStudy interrupted by user. Terminating worker processes...")
        for p in processes: p.terminate(); p.join()
        print("Workers terminated. Results so far have been saved.")

    print("\n===== Optuna Study Finished =====")


def run_ensemble_training(args: argparse.Namespace):
    """Trains N ensemble models using the best config from an Optuna study."""
    print("\n===== Starting Final Ensemble Training =====")
    
    dataset_main = LightweightGraphDataset.load_or_create(save_dir=SAVE_DIR)
    final_config = load_best_config(args, dataset_main)
    
    os.makedirs(args.ensemble_dir, exist_ok=True)
    print(f"\nTraining an ensemble of {args.n_ensemble} models using the best hyperparameters.")
    print(f"Models will be saved in '{args.ensemble_dir}/'")

    n_jobs = len(args.gpu_ids)
    checkpoint_paths = [os.path.join(args.ensemble_dir, f"model_{i}.pt") for i in range(args.n_ensemble)]

    # Iterate in chunks of size n_jobs to train one model per GPU at a time
    for i in range(0, args.n_ensemble, n_jobs):
        batch_processes = []
        batch_indices = range(i, min(i + n_jobs, args.n_ensemble))
        print(f"\n--- Starting training batch for models: {list(batch_indices)} ---")

        for model_idx_in_batch, model_idx_global in enumerate(batch_indices):
            gpu_id = args.gpu_ids[model_idx_in_batch]
            model_path = checkpoint_paths[model_idx_global]
            p = mp.Process(target=train_ensemble_member_worker, args=(gpu_id, model_idx_global, final_config, model_path))
            p.start()
            batch_processes.append(p)
        
        for p in batch_processes: p.join()
        print(f"--- Finished training batch for models: {list(batch_indices)} ---")

    print("\n===== Ensemble Training Finished =====")


def run_ensemble_evaluation(args: argparse.Namespace):
    """Loads and evaluates a trained ensemble."""
    print("\n===== Evaluating Final Ensemble Model =====")
    
    dataset_main = LightweightGraphDataset.load_or_create(save_dir=SAVE_DIR)
    final_config = load_best_config(args, dataset_main)
    
    all_checkpoint_paths = sorted([os.path.join(args.ensemble_dir, f) for f in os.listdir(args.ensemble_dir) if f.startswith('model_') and f.endswith('.pt')])
    
    if args.exclude:
        print(f"Excluding model indices: {args.exclude}")
        final_checkpoint_paths = [p for i, p in enumerate(all_checkpoint_paths) if i not in args.exclude]
    else:
        final_checkpoint_paths = all_checkpoint_paths

    if not final_checkpoint_paths:
        print("No models found to evaluate. Exiting.")
        return

    # Evaluation can be done on a single designated GPU
    eval_device = f"cuda:{args.gpu_ids[0]}"
    dataset_main.to(eval_device)

    ensemble_model = EnsembleGNNRetrievalModel(
        config=final_config,
        data_config=get_data_configs(dataset_main),
        checkpoint_paths=final_checkpoint_paths,
        device=eval_device
    )

    metrics = ensemble_model.evaluate(
        dataset=dataset_main,
        split=args.split,
        batch_size=final_config['evaluation']['batch_size']
    )
    
    print(f"\n--- Final Ensemble Metrics on '{args.split}' split ---")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nEnsemble evaluation complete.")


if __name__ == "__main__":
    """
    Main script for hyperparameter search, ensemble training, and evaluation.
    
    Usage:
    
    1. Perform Hyperparameter Search:
       python your_script_name.py --search --n-trials 50 --gpu-ids 0 2 3
    
    2. Train the Final Ensemble using the best found parameters:
       python your_script_name.py --train-ensemble --n-ensemble 6 --gpu-ids 0 2 3
       
    3. Evaluate the Trained Ensemble on the test set:
       python your_script_name.py --evaluate-ensemble --gpu-ids 0
       
    4. Evaluate, but exclude models 1 and 4 from the ensemble:
       python your_script_name.py --evaluate-ensemble --gpu-ids 0 --exclude 1 4
    """
    parser = argparse.ArgumentParser(description="GNN Retrieval Model Training and Evaluation")
    
    # Mode flags
    parser.add_argument('--search', action='store_true', help='Run Optuna hyperparameter search.')
    parser.add_argument('--train-ensemble', action='store_true', help='Train an ensemble of models using the best found config.')
    parser.add_argument('--evaluate-ensemble', action='store_true', help='Evaluate a pre-trained ensemble of models.')
    
    # Common arguments
    parser.add_argument('--gpu-ids', type=int, nargs='+', default=[0, 2, 3], help='List of GPU IDs to use.')
    parser.add_argument('--db-path', type=str, default="optuna_study.db", help='Path to the Optuna SQLite database.')
    parser.add_argument('--study-name', type=str, default="regularization-search", help='Name of the Optuna study.')
    parser.add_argument('--ensemble-dir', type=str, default="ensemble_models", help='Directory to save/load ensemble models.')
    
    # Search-specific arguments
    parser.add_argument('--n-trials', type=int, default=50, help='Number of trials for Optuna search.')
    
    # Training-specific arguments
    parser.add_argument('--n-ensemble', type=int, default=6, help='Number of models in the ensemble.')
    
    # Evaluation-specific arguments
    parser.add_argument('--exclude', type=int, nargs='*', help='List of model indices to exclude from the ensemble during evaluation.')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='Data split to evaluate on.')

    args = parser.parse_args()
    
    # Use the 'spawn' start method for CUDA safety in multiprocessing
    try:
        mp.set_start_method("spawn", force=True)
        print("Using 'spawn' start method for CUDA compatibility.")
    except RuntimeError:
        print("Could not set 'spawn' start method again (this is OK).")

    if args.search:
        run_hyperparameter_search(args)
    elif args.train_ensemble:
        run_ensemble_training(args)
    elif args.evaluate_ensemble:
        run_ensemble_evaluation(args)
    else:
        print("Please specify a mode of operation.")
        parser.print_help()