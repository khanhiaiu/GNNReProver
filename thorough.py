import numpy as np
import torch
from torch import Tensor
from torch_geometric.nn import RGCNConv, RGATConv, GATConv, GCNConv # type: ignore
from typing import Callable, Dict, Any, List, Literal, Tuple, Optional

from tqdm import tqdm

from lightweight_graph.dataset import LightweightGraphDataset

# New imports for Optuna search
import optuna
import torch.multiprocessing as mp
import os
import traceback


# TODO: when evaluating also take into account the position!

import matplotlib.pyplot as plt
import seaborn as sns

def plot_score_distributions(
    scores: Tensor,
    targets: Tensor,
    file_path: Optional[str] = "score_distribution.png",
    show_plot: bool = False
) -> None:
    # ... (code unchanged)
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
    # ... (code unchanged)
    def __init__(self):
        super(L2Norm, self).__init__() # type: ignore

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.normalize(x, p=2, dim=-1)


# ==============================================================================
# ===== MODIFIED GNN CLASS TO HANDLE DIMENSION MISMATCH ========================
# ==============================================================================
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
    # ... (code unchanged)
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
    # ... (code unchanged)
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.weigh_by_class: bool = config["weigh_by_class"]
        self.loss_function: Literal["bce", "mse", "info_nce"] = config["loss_function"]
        # TODO: pairwise loss

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
    # ... (code unchanged)
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sampler: Literal["random", "hard_top_k", "hard_sampling", "all"] = config["sampler"]
        self.scorer = Scorer(config["scorer"])

    def forward(self, context_embeddings: Tensor, premise_embeddings: Tensor, retrieved_labels : Tensor) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        # Note: retrieved_labels is a tensor of shape (2, n_correct) where the first row is the context indices and the second row is the premise indices
        # return the scores and targets
        n_contexts = context_embeddings.shape[0]
        n_premises = premise_embeddings.shape[0]

        label_premise_indices = retrieved_labels[1]
        if self.sampler == "random":
            n_negatives = self.config["n_negatives_per_context"] * n_contexts
            # randomly sample negatives without repetition
            all_indices = torch.arange(n_premises, device=premise_embeddings.device)
            negative_premise_indices = all_indices[torch.randint(0, n_premises, (n_negatives,), device=premise_embeddings.device)]
            premise_indices = torch.cat([label_premise_indices, negative_premise_indices], dim=0)
        elif self.sampler == "all":
            premise_indices = torch.arange(n_premises, device=premise_embeddings.device)
        elif self.sampler == "hard_top_k" or self.sampler == "hard_sampling":
            # calculate all scores
            with torch.no_grad():
                all_scores = self.scorer(context_embeddings, premise_embeddings) # (n_contexts, n_premises)
            k = self.config["n_negatives_per_context"]
            if self.sampler == "hard_top_k":
                _, hard_negative_indices = torch.topk(all_scores, k=k, dim=1)  # (n_contexts, k)
            elif self.sampler == "hard_sampling":
                probs = torch.softmax(all_scores, dim=1) # (n_contexts, n_premises)
                hard_negative_indices = torch.multinomial(probs, num_samples=k, replacement=False) # (n_contexts, k)
            else:
                raise RuntimeError("Unreachable")
            # flatten to a 1-D tensor of selected negative premise indices
            hard_negative_indices = hard_negative_indices.flatten()
            premise_indices = torch.cat([label_premise_indices, hard_negative_indices], dim=0)
        else:
            raise ValueError(f"Unknown sampler: {self.sampler}")
        
        # sort premise_indices and selected_premise_embeddings by premise_indices to ensure consistent ordering
        premise_indices = torch.unique(premise_indices) # remove duplicates and sort
        selected_premise_embeddings = premise_embeddings[premise_indices]
        
        scores = self.scorer(context_embeddings, selected_premise_embeddings) # (n_contexts, n_selected_premises)

        metrics = {}
        if self.sampler in ["all", "hard_sampling", "hard_top_k"]:
            with torch.no_grad():
                if "all_scores" not in locals():
                    all_scores = self.scorer(context_embeddings, premise_embeddings)
                metrics = calculate_metrics_batch(all_scores, retrieved_labels)


        # construct targets from retrieved_labels and premise_indices
        targets = torch.zeros_like(scores, dtype=torch.bool)
        context_indices = retrieved_labels[0]
        premise_indices_correct = retrieved_labels[1]
        
        # Find positions of correct premises in the selected premise_indices
        assert torch.all(torch.isin(premise_indices_correct, premise_indices)), "Some correct premise indices are not in the selected premise indices" # type: ignore
        premise_positions = torch.searchsorted(premise_indices, premise_indices_correct) # type: ignore

        # Set targets to 1.0 for correct context-premise pairs
        targets[context_indices, premise_positions] = True

        return scores, targets, metrics


def calculate_metrics_batch(all_premise_scores_batch: Tensor, batch_labels : Tensor) -> Dict[str, Tensor]:
    # ... (code unchanged)
    targets = torch.zeros_like(all_premise_scores_batch, dtype=torch.bool)
    context_indices = batch_labels[0]
    premise_indices_correct = batch_labels[1]
    targets[context_indices, premise_indices_correct] = True

    context_premise_count = targets.sum(dim=1) # (n_contexts,)

    valid_contexts_mask = context_premise_count > 0
    assert valid_contexts_mask.any()
    metrics: Dict[str, Any] = {}
    for k in [1, 10]:
        # Consider only valid contexts for evaluation
        valid_scores = all_premise_scores_batch[valid_contexts_mask]
        valid_targets = targets[valid_contexts_mask]
        valid_context_premise_count = context_premise_count[valid_contexts_mask]

        topk_indices = torch.topk(valid_scores, k=k, dim=1).indices # (n_valid_contexts, k)
        
        hits_at_k = valid_targets.gather(1, topk_indices).sum(dim=-1) # (n_valid_contexts,)
        single_R_at_k = hits_at_k / valid_context_premise_count # (n_valid_contexts,)
        metrics[f"R@{k}"] = single_R_at_k
    
    ranks = torch.argsort(torch.argsort(all_premise_scores_batch, dim=1, descending=True), dim=1) + 1 # (n_contexts, n_premises)
    
    ranks_of_correct_only = torch.where(targets, ranks.float(), torch.inf)
    
    min_ranks, _ = torch.min(ranks_of_correct_only, dim=1) # (n_contexts,)
    
    reciprocal_ranks = 1.0 / min_ranks
    
    single_MRR = reciprocal_ranks[valid_contexts_mask]
    metrics["MRR"] = single_MRR

    return metrics


def batchify_dataset(dataset: LightweightGraphDataset, split_indices: Tensor, batch_size: int):
    # ... (code unchanged)
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
        # remap context indices to local batch indices
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


# ==============================================================================
# ===== MODIFIED GNNRetrievalModel TO HANDLE DIMENSION MISMATCH ================
# ==============================================================================
class GNNRetrievalModel(torch.nn.Module):
    def __init__(self, config : Dict[str, Any], data_config: Dict[str, Any]):
        super(GNNRetrievalModel, self).__init__() # type: ignore
        self.config = config
        self.use_random_embeddings: bool = config["use_random_embeddings"]
        self.embedding_size: int = data_config["embedding_dim"]
        
        gnn_config = config["gnn"].copy()
        # Determine the input dimension for the GNN layers.
        # If using random embeddings, the input dim is the hidden dim.
        # Otherwise, it's the dimension of the pre-computed embeddings.
        if self.use_random_embeddings:
            gnn_input_dim = gnn_config["hidden_size"]
        else:
            gnn_input_dim = self.embedding_size
        gnn_config['input_size'] = gnn_input_dim

        self.context_gnn = GNN(gnn_config)
        
        if self.config["separate_premise_GNN"]:
            self.premise_gnn = GNN(gnn_config)
            if self.use_random_embeddings:
                # ALL random embeddings must be initialized with the GNN's hidden_size
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
                
                # Context embeddings must be created with the correct hidden_size, not from initial_batch_context_embeddings
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
    
    # ... (rest of GNNRetrievalModel is unchanged)
    def predict_batch(self, batch_data: Dict[str, Any]):
        context_embeddings, premise_embeddings = self.forward(batch_data)
        return self.scorer.forward(context_embeddings, premise_embeddings)

    def compute_loss(self, batch_data: Dict[str, Any]):
        context_embeddings, premise_embeddings = self.forward(batch_data)
        retrieved_labels = batch_data["retrieved_labels"]
        logits, targets, metrics = self.negative_sampler.forward(context_embeddings, premise_embeddings, retrieved_labels)
        loss = self.loss_function.compute(logits, targets)
        #import code; code.interact(local=dict(globals(), **locals()))

        # Plot score distributions for analysis
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
    
    def train_epoch(self, dataset: LightweightGraphDataset, gradient_accumulation_steps : int) -> None:
        self.train()
        self.optimizer.zero_grad()
        train_split_indices = dataset.train_mask.nonzero(as_tuple=False).view(-1)
        train_generator = batchify_dataset(dataset, train_split_indices, self.config["training"]["batch_size"])
        pbar = tqdm(enumerate(train_generator), desc="Training")

        all_batch_metrics : Dict[str, List[Any]] = {}
        for i, batch_data in pbar:
            should_update = ((i + 1) % gradient_accumulation_steps == 0) # TODO: what about last step?
            batch_metrics = self.train_batch(batch_data, should_updated=should_update)
            for key, value in batch_metrics.items():
                if key not in all_batch_metrics:
                    all_batch_metrics[key] = []
                all_batch_metrics[key].append(value)
            
            mean_metrics = {key: sum(values) / len(values) for key, values in all_batch_metrics.items()}
            pbar.set_postfix({key: f"{value:.4f}" for key, value in mean_metrics.items()})

    def overfit_first_batch(self, dataset: LightweightGraphDataset, max_steps: int, batch_size : int) -> None:
        self.train()
        train_split_indices = dataset.train_mask.nonzero(as_tuple=False).view(-1)
        # Use the entire training set as a single batch
        train_generator = batchify_dataset(dataset, train_split_indices, batch_size=batch_size)
        first_batch_data = next(train_generator)

        pbar = tqdm(range(max_steps), desc="Overfitting on first batch")
        for step in pbar:
            batch_metrics = self.train_batch(first_batch_data, should_updated=True)
            
            postfix = {key: f"{value:.4f}" for key, value in batch_metrics.items()}
            postfix["step"] = str(step)
            pbar.set_postfix(postfix)
    
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

    def evaluate(self, dataset: LightweightGraphDataset, split: str) -> Dict[str, float]:
        self.eval()
        batch_size = self.config["evaluation"]["batch_size"]
        with torch.no_grad():
            mask = getattr(dataset, f"{split}_mask", None)
            if mask is None: raise ValueError(f"Invalid split: {split}")
            split_indices = mask.nonzero(as_tuple=False).view(-1)

            eval_generator = batchify_dataset(dataset, split_indices, batch_size)

            metrics_data : list[Dict[str, Tensor]] = []
            pbar = tqdm(eval_generator, desc=f"Evaluating on {split} split")
            for batch_data in pbar:
                metrics_data.append(self.evaluate_batch(batch_data, dataset))

        # concat
        all_metrics_lists: Dict[str, list[Tensor]] = {}
        for m in metrics_data:
            for key, value in m.items():
                if key not in all_metrics_lists:
                    all_metrics_lists[key] = []
                all_metrics_lists[key].append(value)

        all_metrics_tensor = {key: torch.cat(value, dim=0) for key, value in all_metrics_lists.items()}
        # take mean
        metrics : Dict[str, float] = {}
        for key, value in all_metrics_tensor.items():
            metrics[key] = value.mean().item()

        return metrics


def get_data_configs(dataset: LightweightGraphDataset) -> Dict[str, Any]:
    # ... (code unchanged)
    data_config = {
        "n_premises": dataset.premise_embeddings.shape[0],
        "n_contexts": dataset.context_embeddings.shape[0],
        "embedding_dim": dataset.premise_embeddings.shape[1],
    }
    return data_config


def objective(trial: optuna.Trial, device: str, dataset: LightweightGraphDataset, data_config: Dict[str, Any], n_relations: int) -> float:
    # ... (code unchanged)
    # Ensure all tensors created inside this function are on the correct device
    torch.cuda.set_device(device)
    
    # --- Hyperparameter Search Space ---
    
    # GNN head dimensions and heads to ensure hidden_size is divisible by heads
    gnn_head_dim = trial.suggest_categorical("gnn_head_dim", [64, 128, 256])
    gnn_heads = trial.suggest_categorical("gnn_heads", [1, 2, 4, 8])
    gnn_hidden_size = gnn_head_dim * gnn_heads

    scorer_preprocess_type = trial.suggest_categorical("scorer_preprocess_type", ['linear', 'nn', 'cosine', 'none'])
    
    loss_function_type = trial.suggest_categorical("loss_function", ["bce", "mse", "info_nce"])
    if loss_function_type == "info_nce":
        weigh_by_class = False # Not supported
    else:
        weigh_by_class = trial.suggest_categorical("weigh_by_class", [True, False])

    config = {
        'use_random_embeddings': trial.suggest_categorical("use_random_embeddings", [True, False]),
        'separate_premise_GNN': trial.suggest_categorical("separate_premise_GNN", [True, False]),
        'gnn': {
            'layer_type': trial.suggest_categorical("layer_type", ['RGCN', 'RGAT', 'GAT', 'GCN']),
            'n_layers': trial.suggest_int("n_layers", 1, 4),
            'hidden_size': gnn_hidden_size,
            'n_relations': n_relations,
            'activation': trial.suggest_categorical("activation", ['relu', 'gelu']),
            'dropout': trial.suggest_float("dropout", 0.0, 0.5),
            'residual': trial.suggest_categorical("residual", [True, False]),
            'normalization': trial.suggest_categorical("normalization", ['batchnorm', 'layernorm', 'l2', 'none']),
            'heads': gnn_heads,
        },
        'negative_sampler': {
            'sampler': 'all',  # Fixed to 'all' for evaluating memorization capacity
            'scorer': {
                'preprocess': {
                    'type': scorer_preprocess_type,
                    'depth': trial.suggest_int("scorer_nn_depth", 1, 3) if scorer_preprocess_type == 'nn' else 1,
                },
                'heads': trial.suggest_categorical("scorer_heads", [1, 2, 4, 8]),
                'aggregation': trial.suggest_categorical("scorer_aggregation", ['logsumexp', 'mean', 'max']),
                'embedding_dim': gnn_hidden_size,  # Scorer input dim must match GNN output dim
            }
        },
        'loss': {
            'loss_function': loss_function_type,
            'weigh_by_class': weigh_by_class,
            'temperature': trial.suggest_float("temperature", 0.01, 1.0, log=True) if loss_function_type == 'info_nce' else 1.0,
        },
        'optimizer': {
            'lr': trial.suggest_float("lr", 1e-4, 5e-2, log=True),
            'weight_decay': trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        },
        'evaluation': { # Not used in objective, but needed for model init
            'batch_size': 128,
        },
        'training': {} # Not used in objective
    }
    
    model = None # Initialize to ensure it's in scope for 'finally'
    objective_value = -1.0 # Default value for failed trials
    try:
        # --- Model Setup and Training ---

        # 1. Instantiate the model
        model = GNNRetrievalModel(config, data_config).to(device)
        
        # 2. Get the single batch to overfit on
        batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048, 4096])
        train_split_indices = dataset.train_mask.nonzero(as_tuple=False).view(-1)
        train_generator = batchify_dataset(dataset, train_split_indices, batch_size=batch_size)
        first_batch_data = next(train_generator)

        # 3. Overfit on the batch for a fixed number of steps
        max_steps = 200
        
        for _ in range(max_steps):
            model.train_batch(first_batch_data, should_updated=True)
        
        # 4. Evaluate on the same batch to measure memorization
        final_metrics = model.evaluate_batch(first_batch_data, dataset)
        
        # The metric to maximize is R@10.
        objective_value = final_metrics["R@10"].mean().item()

    except Exception as e:
        # Handle potential errors like OOM, invalid hyperparameter combinations, etc.
        print(f"Trial {trial.number} failed with error: {e}")
        print(traceback.format_exc())
        # objective_value is already -1.0, so the trial will be reported as a failure.
    finally:
        # Clean up GPU memory to prevent interference between trials
        if model is not None:
            del model
        torch.cuda.empty_cache()

    return objective_value


def run_worker(rank: int, study: optuna.Study, n_trials_per_worker: int, gpu_indices: List[int], save_dir: str, n_relations: int):
    # ... (code unchanged)
    """A single worker process for the Optuna search."""
    device_id = gpu_indices[rank]
    device = f"cuda:{device_id}"
    torch.cuda.set_device(device)
    
    print(f"Worker {rank} started, using device: {device}")

    # Each worker loads its own copy of the dataset onto its assigned GPU
    dataset = LightweightGraphDataset.load_or_create(save_dir=save_dir)
    dataset.to(device)
    data_config = get_data_configs(dataset)

    # Run the optimization loop for this worker
    study.optimize(
        lambda trial: objective(trial, device, dataset, data_config, n_relations),
        n_trials=n_trials_per_worker,
        gc_after_trial=True # Garbage collect after each trial to free memory
    )
    print(f"Worker {rank} finished.")


if __name__ == "__main__":
    # ... (code unchanged)
    torch.manual_seed(42)
    # Set multiprocessing start method for CUDA
    mp.set_start_method("spawn", force=True)

    # --- Optuna Search Configuration ---
    SAVE_DIR = "lightweight_graph/data"
    GPU_INDICES = [0, 2, 3]  # Your available GPUs
    N_WORKERS = len(GPU_INDICES)
    TOTAL_TRIALS = 3000
    # Distribute trials among workers
    N_TRIALS_PER_WORKER = TOTAL_TRIALS // N_WORKERS

    STORAGE_NAME = "sqlite:///optuna_overfit_study_r10.db" # Changed DB name to reflect new objective
    STUDY_NAME = "memorization-search-r10-v1" # Changed study name
    
    print("Starting Optuna hyperparameter search for single-batch overfitting.")
    print(f"Running {TOTAL_TRIALS} trials across {N_WORKERS} GPUs: {GPU_INDICES}")
    print(f"Study results will be saved to: {STORAGE_NAME}")

    # --- Pre-computation ---
    # Load dataset on CPU once to get number of relations without occupying a GPU
    print("Pre-loading dataset on CPU to determine number of relations...")
    dataset_cpu = LightweightGraphDataset.load_or_create(save_dir=SAVE_DIR)
    max_premise_rel = dataset_cpu.premise_edge_attr.max().item() if dataset_cpu.premise_edge_attr.numel() > 0 else -1
    max_context_rel = dataset_cpu.context_edge_attr.max().item() if dataset_cpu.context_edge_attr.numel() > 0 else -1
    n_relations = max(max_premise_rel, max_context_rel) + 1
    del dataset_cpu # Free up memory
    print(f"Number of relations found in data: {n_relations}")

    # --- Create or Load Optuna Study ---
    study = optuna.create_study(
        storage=STORAGE_NAME,
        study_name=STUDY_NAME,
        direction="maximize", # We want to maximize R@10
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner() # Prune unpromising trials
    )

    # --- Run Parallel Search ---
    args = (study, N_TRIALS_PER_WORKER, GPU_INDICES, SAVE_DIR, n_relations)
    mp.spawn(run_worker, args=args, nprocs=N_WORKERS, join=True)

    # --- Print Results ---
    print("\nOptuna search finished.")
    print(f"Study '{STUDY_NAME}' results:")
    print(f"  Number of finished trials: {len(study.trials)}")
    
    best_trial = study.best_trial
    print("  Best trial:")
    print(f"    Value (R@10): {best_trial.value:.6f}")
    print("    Params: ")
    for key, value in best_trial.params.items():
        print(f"      {key}: {value}")