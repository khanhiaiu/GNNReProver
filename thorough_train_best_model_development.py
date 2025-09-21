import numpy as np
import torch
from torch import Tensor
from torch_geometric.nn import RGCNConv, RGATConv, GATConv, GCNConv # type: ignore
from typing import Callable, Dict, Any, List, Literal, Tuple, Optional
import json

from tqdm import tqdm

from lightweight_graph.dataset import LightweightGraphDataset

# New imports for loading Optuna results
import optuna

# NOTE: The original imports for running the search (mp, os, traceback) are no longer needed for this script.

import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# ===== ALL YOUR ORIGINAL CLASSES AND FUNCTIONS (UNCHANGED) ====================
# ==============================================================================
# Your provided code for plot_score_distributions, L2Norm, GNN, Scorer,
# LossFunction, NegativeSampler, calculate_metrics_batch, batchify_dataset,
# GNNRetrievalModel, and get_data_configs is assumed to be here.
# For brevity, it is collapsed. I will paste it in full below.

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
        
        if self.config["separate_premise_GNN"]:
            if self.use_random_embeddings:
                premise_embeddings = self.premise_gnn(self.random_initial_premise_embeddings, premise_edge_index, premise_edge_type)
                dummy_context_embeddings = torch.zeros(n_contexts, hidden_size, device=self.random_initial_premise_embeddings_for_context.device)
                all_random_embeddings_for_context = torch.cat([self.random_initial_premise_embeddings_for_context, dummy_context_embeddings], dim=0)
                all_embeddings_for_context = self.context_gnn(all_random_embeddings_for_context, all_edge_index, all_edge_type)
                context_embeddings = all_embeddings_for_context[n_premises:]
            else:
                premise_embeddings = self.premise_gnn(initial_premise_embeddings, premise_edge_index, premise_edge_type)
                all_initial_embeddings_for_context = torch.cat([initial_premise_embeddings, initial_batch_context_embeddings], dim=0)
                all_embeddings_for_context = self.context_gnn(all_initial_embeddings_for_context, all_edge_index, all_edge_type)
                context_embeddings = all_embeddings_for_context[n_premises:]
        else: # not separate_premise_GNN
            if self.use_random_embeddings:
                dummy_context_embeddings = torch.zeros(n_contexts, hidden_size, device=self.random_initial_premise_embeddings.device)
                all_random_embeddings = torch.cat([self.random_initial_premise_embeddings, dummy_context_embeddings], dim=0)
                all_embeddings = self.context_gnn(all_random_embeddings, all_edge_index, all_edge_type)
                premise_embeddings = all_embeddings[:n_premises]
                context_embeddings = all_embeddings[n_premises:]
            else:
                all_initial_embeddings = torch.cat([initial_premise_embeddings, initial_batch_context_embeddings], dim=0)
                all_embeddings = self.context_gnn(all_initial_embeddings, all_edge_index, all_edge_type)
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
            scores = self.scorer.forward(context_embeddings, premise_embeddings)
            device = scores.device

            batch_context_file_indices = batch_data["context_to_file_idx_map"]
            premise_to_file_idx_map = dataset.premise_to_file_idx_map
            file_dependency_edge_index = dataset.file_dependency_edge_index
            
            in_file_mask = batch_context_file_indices.unsqueeze(1) == premise_to_file_idx_map.unsqueeze(0)

            n_files : int = max(premise_to_file_idx_map.max().item(), file_dependency_edge_index.max().item()) + 1

            file_dependency_adj = torch.zeros((n_files, n_files), dtype=torch.bool, device=device)
            src, dst = file_dependency_edge_index
            file_dependency_adj[src, dst] = True

            batch_dependency_matrix = file_dependency_adj[batch_context_file_indices]

            imported_mask = batch_dependency_matrix[:, premise_to_file_idx_map]
            
            accessible_mask = in_file_mask | imported_mask
            
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


def get_data_configs(dataset: LightweightGraphDataset) -> Dict[str, Any]:
    data_config = {
        "n_premises": dataset.premise_embeddings.shape[0],
        "n_contexts": dataset.context_embeddings.shape[0],
        "embedding_dim": dataset.premise_embeddings.shape[1],
    }
    return data_config


# ==============================================================================
# ===== STEP 1: FUNCTION TO GET BEST PARAMETERS FROM OPTUNA DB ===============
# ==============================================================================

def load_best_params_from_study(db_path: str, study_name: str) -> Dict[str, Any]:
    """
    Loads an Optuna study from a SQLite database and returns the parameters
    of the best trial.
    """
    print(f"Loading Optuna study '{study_name}' from '{db_path}'...")
    storage = optuna.storages.RDBStorage(url=f"sqlite:///{db_path}")
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except KeyError:
        print(f"Error: Study '{study_name}' not found in '{db_path}'.")
        # Optional: Print available study names
        all_studies = optuna.get_all_study_summaries(storage=storage)
        if all_studies:
            print("Available studies are:")
            for s in all_studies:
                print(f"- {s.study_name}")
        exit()
        
    best_trial = study.best_trial
    print(f"Found best trial #{best_trial.number} with R@10 value: {best_trial.value:.6f}")
    
    return best_trial.params


# ==============================================================================
# ===== STEP 2: FUNCTION TO RECONSTRUCT THE FULL CONFIG ========================
# ==============================================================================

def create_config_from_params(params: Dict[str, Any], n_relations: int, data_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reconstructs the full, nested configuration dictionary from the flat
    parameter dictionary returned by Optuna.
    """
    # Recreate derived parameters (like gnn_hidden_size)
    gnn_hidden_size = params['gnn_head_dim'] * params['gnn_heads']
    
    # Determine weigh_by_class based on loss function
    if params['loss_function'] == 'info_nce':
        weigh_by_class = False
    else:
        weigh_by_class = params.get('weigh_by_class', False) # Default to False if not present

    # Reconstruct the nested config dictionary
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
            'residual': params['residual'],
            'normalization': params['normalization'],
            'heads': params['gnn_heads'],
        },
        'negative_sampler': {
            # In your main training, you might want a different sampler
            'sampler': 'all',  # Using 'all' for evaluation consistency
            'scorer': {
                'preprocess': {
                    'type': params['scorer_preprocess_type'],
                    'depth': params.get('scorer_nn_depth', 1), # Use .get for optional params
                },
                'heads': params['scorer_heads'],
                # Handle potential missing 'scorer_aggregation' in older trials
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
        # Add training and evaluation configs for the "normal" run
        'training': {
            'batch_size': params['batch_size'],
            'gradient_accumulation_steps': 2
        },
        'evaluation': {
            'batch_size': 2048, # Use a larger batch size for faster evaluation
        },
        'plot_score_distributions': False # Enable plotting for the final model
    }
    
    print("\n--- Constructed Full Model Configuration ---")
    print(json.dumps(config, indent=2))
    print("------------------------------------------\n")
    
    return config


# ==============================================================================
# ===== NEW MAIN BLOCK FOR NORMAL (NON-OPTUNA) TRAINING ========================
# ==============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    # --- Configuration ---
    DB_PATH = "optuna_overfit_study_r10.db"
    STUDY_NAME = "memorization-search-r10-v1"
    SAVE_DIR = "lightweight_graph/data"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    N_EPOCHS = 100
    
    print(f"Using device: {DEVICE}")

    # --- 1. Get the parameters of the best trial ---
    best_params = load_best_params_from_study(DB_PATH, STUDY_NAME)

    # --- Setup Data and Model ---
    print("Loading dataset...")
    dataset = LightweightGraphDataset.load_or_create(save_dir=SAVE_DIR)
    dataset.to(DEVICE)
    
    data_config = get_data_configs(dataset)
    
    max_premise_rel = dataset.premise_edge_attr.max().item() if dataset.premise_edge_attr.numel() > 0 else -1
    max_context_rel = dataset.context_edge_attr.max().item() if dataset.context_edge_attr.numel() > 0 else -1
    n_relations = max(max_premise_rel, max_context_rel) + 1
    print(f"Number of relations found in data: {n_relations}")

    # --- 2. Create a complete config and run the model normally ---
    best_config = create_config_from_params(best_params, n_relations, data_config)
    
    model = GNNRetrievalModel(config=best_config, data_config=data_config).to(DEVICE)

    # --- Training and Evaluation Loop ---
    for epoch in range(1, N_EPOCHS + 1):
        print(f"\n===== EPOCH {epoch}/{N_EPOCHS} =====")
        
        # Train for one epoch
        model.train_epoch(dataset, config=best_config['training'])
        
        # Evaluate on the validation set
        val_metrics = model.evaluate(dataset, 'val', batch_size=best_config['evaluation']['batch_size'])
        print(f"\nValidation metrics for Epoch {epoch}:")
        for key, value in val_metrics.items():
            print(f"  {key}: {value:.4f}")
            
    # --- Final Evaluation on Test Set ---
    print("\n===== FINAL EVALUATION ON TEST SET =====")
    test_metrics = model.evaluate(dataset, 'test', batch_size=best_config['evaluation']['batch_size'])
    print("\nFinal Test Metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")
        
    # --- Save the final trained model ---
    model_save_path = "best_model_from_optuna.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"\nModel state dictionary saved to '{model_save_path}'")