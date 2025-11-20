from concurrent.futures import Future, ProcessPoolExecutor, as_completed
import logging
import warnings
import torch
from torch import Tensor
from torch_geometric.nn import RGCNConv, RGATConv, GATConv, GCNConv # type: ignore
from typing import Callable, Dict, Any, List, Literal, Tuple, Optional
import copy
from tqdm import tqdm
import yaml
from lightweight_graph.dataset import LightweightGraphDataset
import optuna
from contextlib import contextmanager
import torch.multiprocessing as mp

def to_undirected(edge_index: Tensor, edge_attr: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
    undirected_edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    if edge_attr is not None: undirected_edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    else: undirected_edge_attr = None
    return undirected_edge_index, undirected_edge_attr

class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        if not (0.0 <= decay <= 1.0):
            raise ValueError("Decay must be between 0 and 1.")
        self.decay = decay
        self.model = model
        self.shadow_params = [p.clone().detach() for p in model.parameters() if p.requires_grad]
        self.updates = 0
        self.original_params: Optional[List[Tensor]] = None

    def update(self) -> None:
        self.updates += 1
        with torch.no_grad():
            for shadow_p, model_p in zip(self.shadow_params, self.model.parameters()):
                if model_p.requires_grad:
                    shadow_p.sub_((1.0 - self.decay) * (shadow_p - model_p))

    def _apply_shadow(self) -> None:
        corrected_decay = 1.0 - (self.decay ** self.updates) if self.updates > 0 else 1.0
        
        self.original_params = [p.clone().detach() for p in self.model.parameters() if p.requires_grad]

        with torch.no_grad():
            for shadow_p, model_p in zip(self.shadow_params, self.model.parameters()):
                if model_p.requires_grad:
                    model_p.copy_(shadow_p / corrected_decay)

    def _restore_original(self) -> None:
        """Restores the original model parameters."""
        if self.original_params is None:
            return
        with torch.no_grad():
            for original_p, model_p in zip(self.original_params, self.model.parameters()):
                if model_p.requires_grad:
                    model_p.copy_(original_p)
        self.original_params = None

    @contextmanager
    def average_parameters(self):
        self._apply_shadow()
        try:
            yield
        finally:
            self._restore_original()

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
            raise ValueError(f"Unknown aggregation type: {aggregation_type}")


    def forward(self, context_embeddings: Tensor, premise_embeddings: Tensor) -> Tensor:
        b, e = context_embeddings.shape
        p, e_prime = premise_embeddings.shape
        assert e == e_prime, f"Embedding dimensions must match: {e} != {e_prime}"

        context_embeddings = self.preprocess(context_embeddings)
        premise_embeddings = self.preprocess(premise_embeddings)

        context_embeddings = context_embeddings.view(b, self.heads, e // self.heads)
        premise_embeddings = premise_embeddings.view(p, self.heads, e // self.heads)
        scores = torch.einsum('bhd, phd -> bhp', context_embeddings, premise_embeddings)
        scores = self.aggregate(scores).squeeze(1)

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
            negative_premise_indices = torch.randint(0, n_premises, (n_negatives,), device=premise_embeddings.device)
            premise_indices = torch.cat([label_premise_indices, negative_premise_indices], dim=0)
        elif self.sampler == "all":
            premise_indices = torch.arange(n_premises, device=premise_embeddings.device)
        elif self.sampler == "hard_top_k" or self.sampler == "hard_sampling":
            with torch.no_grad():
                all_scores = self.scorer(context_embeddings, premise_embeddings) 
            k = self.config["n_negatives_per_context"]
            if self.sampler == "hard_top_k":
                _, hard_negative_indices = torch.topk(all_scores, k=k, dim=1)
                assert hard_negative_indices.shape == (n_contexts, k)
            elif self.sampler == "hard_sampling":
                probs = torch.softmax(all_scores, dim=1) 
                hard_negative_indices = torch.multinomial(probs, num_samples=k, replacement=False)
            else:
                raise RuntimeError("Unreachable")
            hard_negative_indices = hard_negative_indices.flatten()
            premise_indices = torch.cat([label_premise_indices, hard_negative_indices], dim=0)
        else:
            raise ValueError(f"Unknown sampler: {self.sampler}")

        premise_indices: torch.Tensor = torch.unique(premise_indices)
        premise_indices, _ = torch.sort(premise_indices)
        selected_premise_embeddings = premise_embeddings[premise_indices]
        
        scores = self.scorer(context_embeddings, selected_premise_embeddings)

        metrics: Dict[str, Tensor] = {}
        if self.sampler in ["all", "hard_sampling", "hard_top_k"]:
            with torch.no_grad():
                if "all_scores" not in locals():
                    all_scores = self.scorer(context_embeddings, premise_embeddings)
                metrics = calculate_metrics_batch(all_scores, retrieved_labels)

        targets = torch.zeros_like(scores, dtype=torch.bool)
        context_indices_correct = retrieved_labels[0]
        premise_indices_correct = retrieved_labels[1]
        
        assert torch.all(torch.isin(premise_indices_correct, premise_indices)), "Some correct premise indices are not in the selected premise indices" # type: ignore
        premise_positions = torch.searchsorted(premise_indices, premise_indices_correct) # type: ignore

        targets[context_indices_correct, premise_positions] = True

        return scores, targets, metrics

def calculate_metrics_batch(all_premise_scores_batch: Tensor, batch_labels : Tensor) -> Dict[str, Tensor]:
    targets = torch.zeros_like(all_premise_scores_batch, dtype=torch.bool)
    context_indices = batch_labels[0]
    premise_indices_correct = batch_labels[1]
    targets[context_indices, premise_indices_correct] = True

    context_premise_count = targets.sum(dim=1) 

    valid_contexts_mask = context_premise_count > 0
    if not valid_contexts_mask.any():
        warnings.warn("No valid contexts with at least one correct premise found for metrics calculation.")
        return {"R@1": torch.tensor(0.0), "R@10": torch.tensor(0.0), "MRR": torch.tensor(0.0)}

    metrics: Dict[str, Any] = {}
    for k in [1, 10]:
        valid_scores = all_premise_scores_batch[valid_contexts_mask]
        valid_targets = targets[valid_contexts_mask]
        valid_context_premise_count = context_premise_count[valid_contexts_mask]

        topk_indices = torch.topk(valid_scores, k=k, dim=1).indices 
        assert topk_indices.shape == (valid_scores.shape[0], k)
        
        hits_at_k = valid_targets.gather(1, topk_indices).sum(dim=-1)
        assert hits_at_k.shape == (valid_scores.shape[0],)
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
        
        batch_context_file_indices = dataset.context_to_file_idx_map[batch_global_indices].clone()
        batch_context_theorem_pos = dataset.context_theorem_pos[batch_global_indices].clone()
        
        batch_edge_mask = torch.isin(premise_to_context_edge_index[1], batch_global_indices)
        batch_premise_to_context_edge_index_global = premise_to_context_edge_index[:, batch_edge_mask].clone()
        batch_premise_to_context_edge_type = premise_to_context_edge_type[batch_edge_mask].clone()
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

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__() # type: ignore
    
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

    def get_scores(self, batch_data: Dict[str, Any]) -> Tensor:
        raise NotImplementedError

    def evaluate_batch(self, batch_data: Dict[str, Any], dataset: LightweightGraphDataset) -> Dict[str, Tensor]:
        self.eval()
        with torch.no_grad():
            scores = self.get_scores(batch_data)
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

class GNNRetrievalModel(Model):
    def __init__(self, config : Dict[str, Any]):
        super(GNNRetrievalModel, self).__init__() # type: ignore
        self.config = config
        self.data_config = config["data_config"]
        self.use_random_embeddings: bool = config["use_random_embeddings"]
        self.embedding_size: int = self.data_config["embedding_dim"]
        
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
                self.random_initial_premise_embeddings = torch.nn.Parameter(torch.randn(self.data_config["n_premises"], hidden_size))
                self.random_initial_premise_embeddings_for_context = torch.nn.Parameter(torch.randn(self.data_config["n_premises"], hidden_size))
        else:
            self.premise_gnn = self.context_gnn
            if self.use_random_embeddings:
                hidden_size = gnn_config["hidden_size"]
                self.random_initial_premise_embeddings = torch.nn.Parameter(torch.randn(self.data_config["n_premises"], hidden_size))
        
        self.loss_function = LossFunction(config["loss"])
        self.negative_sampler = NegativeSampler(config["negative_sampler"])
        self.scorer = self.negative_sampler.scorer

        optimizer_config = config["optimizer"]
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=optimizer_config["lr"], weight_decay=optimizer_config.get("weight_decay", 0.0))

    def forward(self, batch_data: Dict[str, Any]) -> Tuple[Tensor, Tensor]:
        initial_premise_embeddings = batch_data["premise_embeddings"]
        initial_batch_context_embeddings = batch_data["context_embeddings"]
        device = initial_premise_embeddings.device
        assert device == initial_batch_context_embeddings.device

        n_premises = initial_premise_embeddings.shape[0]
        assert n_premises == self.data_config["n_premises"]
        n_contexts = initial_batch_context_embeddings.shape[0]
        hidden_size = self.config["gnn"]["hidden_size"]
        assert initial_premise_embeddings.shape[1] == initial_batch_context_embeddings.shape[1] == self.embedding_size

        premise_edge_index = batch_data["premise_edge_index"]
        premise_edge_type = batch_data["premise_edge_type"]
        if self.config["undirected_premise_graph"]:
            premise_edge_index, premise_edge_type = to_undirected(premise_edge_index, premise_edge_type)

        premise_to_context_edge_index = batch_data["premise_to_context_edge_index"].clone()
        assert premise_to_context_edge_index[1].max() <= n_contexts
        premise_to_context_edge_index[1] += n_premises 
        premise_to_context_edge_type = batch_data["premise_to_context_edge_type"]

        all_edge_index = torch.cat([premise_edge_index, premise_to_context_edge_index], dim=1)
        all_edge_type = torch.cat([premise_edge_type, premise_to_context_edge_type], dim=0)
        
        edge_dropout_p = self.config['gnn'].get('edge_dropout', 0.0)

        # TODO: plot stuff
        if self.config["separate_premise_GNN"]:
            premise_ei_train, premise_et_train = edge_dropout(
                premise_edge_index, premise_edge_type, p=edge_dropout_p, training=self.training
            )
            all_ei_train, all_et_train = edge_dropout(
                all_edge_index, all_edge_type, p=edge_dropout_p, training=self.training
            )
            if self.use_random_embeddings:
                premise_embeddings = self.premise_gnn(self.random_initial_premise_embeddings, premise_ei_train, premise_et_train)
                dummy_context_embeddings = torch.zeros(n_contexts, hidden_size, device=device)
                all_random_embeddings_for_context = torch.cat([self.random_initial_premise_embeddings_for_context, dummy_context_embeddings], dim=0)
                all_embeddings_for_context = self.context_gnn(all_random_embeddings_for_context, all_ei_train, all_et_train)
                context_embeddings = all_embeddings_for_context[n_premises:]
            else:
                premise_embeddings = self.premise_gnn(initial_premise_embeddings, premise_ei_train, premise_et_train)
                all_initial_embeddings_for_context = torch.cat([initial_premise_embeddings, initial_batch_context_embeddings], dim=0)
                all_embeddings_for_context = self.context_gnn(all_initial_embeddings_for_context, all_ei_train, all_et_train)
                context_embeddings = all_embeddings_for_context[n_premises:]
        else:
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
        return loss, metrics

    def train_epoch(self, dataset: LightweightGraphDataset, config: Dict[str, Any], ema: Optional[EMA] = None):
        self.optimizer.zero_grad()
        training_config = config["training"]
        gradient_accumulation_steps = training_config.get("gradient_accumulation_steps", 1)
        
        train_split_indices = dataset.train_mask.nonzero(as_tuple=False).view(-1)
        train_generator = batchify_dataset(dataset, split_indices=train_split_indices, batch_size=training_config["batch_size"])
        
        num_batches = (len(train_split_indices) + training_config["batch_size"] - 1) // training_config["batch_size"]
        pbar = tqdm(enumerate(train_generator), total=num_batches, desc="Training")

        all_batch_metrics: Dict[str, List[Any]] = {}
        for i, batch_data in pbar:
            self.train()
            should_update = ((i + 1) % gradient_accumulation_steps == 0) or (i + 1 == num_batches)
            
            loss, batch_metrics_tensors = self.compute_loss(batch_data)
            memory = torch.cuda.memory_allocated(device=loss.device) / (1024 ** 3)
            loss.backward() # type: ignore

            if should_update:
                self.optimizer.step()
                if ema:
                    ema.update()
                self.optimizer.zero_grad()
            
            batch_metrics = {k: v.mean().item() for k, v in batch_metrics_tensors.items()}
            batch_metrics["loss"] = loss.item()
            batch_metrics["memory"] = memory

            for key, value in batch_metrics.items():
                if key not in all_batch_metrics:
                    all_batch_metrics[key] = []
                all_batch_metrics[key].append(value)
            
            mean_metrics = {key: sum(values) / len(values) for key, values in all_batch_metrics.items()}
            pbar.set_postfix({key: f"{value:.4f}" for key, value in mean_metrics.items()})
        
        return {
            "train_metrics": {key: sum(values) / len(values) for key, values in all_batch_metrics.items()}
        }

    def get_scores(self, batch_data: Dict[str, Any]) -> Tensor:
        context_embeddings, premise_embeddings = self.forward(batch_data)
        scores = self.scorer(context_embeddings, premise_embeddings)
        return scores

class EnsembleGNNRetrievalModel(Model):
    def __init__(self, config: Dict[str, Any], checkpoint_paths: List[str], device: str):
        super().__init__()
        self.models = torch.nn.ModuleList()
        self.device = device
        
        print(f"Loading {len(checkpoint_paths)} models for the ensemble...")
        for path in checkpoint_paths:
            model = GNNRetrievalModel(config)
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device)
            self.models.append(model)
        print("Ensemble models loaded successfully.")

    def get_scores(self, batch_data: Dict[str, Any]) -> Tensor:
        all_scores: List[Tensor] = []
        for model in self.models:
            context_embeddings, premise_embeddings = model.forward(batch_data)
            scores = model.scorer(context_embeddings, premise_embeddings)
            all_scores.append(scores)

        ensemble_scores = torch.stack(all_scores).mean(dim=0)
        return ensemble_scores

def get_data_config(dataset: LightweightGraphDataset) -> Dict[str, Any]:
    data_config = {
        "n_premises": dataset.premise_embeddings.shape[0],
        "n_contexts": dataset.context_embeddings.shape[0],
        "embedding_dim": dataset.premise_embeddings.shape[1],
    }
    return data_config

SAVE_DIR = "lightweight_graph/data_random_updated"
def load_dataset(save_dir: str=SAVE_DIR) -> LightweightGraphDataset:
    dataset = LightweightGraphDataset.load_or_create(save_dir=save_dir)
    return dataset

def run_experiment(dataset : LightweightGraphDataset, config: Dict[str, Any], gpu_id: int, save_path: Optional[str], early_stopping_metric:str = "R@10"):
    torch.cuda.set_device(gpu_id)
    device = "cuda"
    torch.manual_seed(42 + gpu_id) # type: ignore

    model = GNNRetrievalModel(config=config).to(device)
    use_ema = config.get("use_ema", False)
    ema: Optional[EMA] = None
    if use_ema:
        ema_decay = config.get("ema_decay", 0.999)
        ema = EMA(model, decay=ema_decay)
        print(f"Using EMA with decay {ema_decay}")

    best_val_metrics = None
    epoch = config["training"]["epochs"]
    assert epoch > 0, "Number of epochs must be positive"

    test_metrics = None
    for epoch in range(1, epoch + 1):
        metrics = model.train_epoch(dataset, config=config)
        train_metrics = metrics["train_metrics"]
        with torch.no_grad():
            if use_ema and ema:
                with ema.average_parameters():
                    val_metrics = model.evaluate(dataset, 'val', batch_size=config['evaluation']['batch_size'])
            else:
                val_metrics = model.evaluate(dataset, 'val', batch_size=config['evaluation']['batch_size'])
            if best_val_metrics is None or best_val_metrics[early_stopping_metric] < val_metrics[early_stopping_metric]:
                if save_path:
                    torch.save(model.state_dict(), save_path)
                    print(f"Saved best model at epoch {epoch} with {early_stopping_metric} = {val_metrics[early_stopping_metric]:.4f} to {save_path}")
                    if use_ema and ema:
                        ema_path = save_path + "_ema"
                        torch.save(ema.shadow_params, ema_path)
                        print(f"Saved EMA parameters to {ema_path}")
                best_val_metrics = val_metrics

                test_metrics = model.evaluate(dataset, 'test', batch_size=config['evaluation']['batch_size'])

    assert test_metrics is not None
    assert best_val_metrics is not None
    return {
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "train_metrics": train_metrics,
    }

def objective(trial: optuna.Trial, base_config: Dict[str, Any], base_dataset: LightweightGraphDataset) -> float:
    gpu_id = trial.user_attrs["gpu_id"]
    trial_logger = logging.LoggerAdapter(logging.getLogger(__name__), {'gpu_id': gpu_id})
    trial_logger.info(f"Starting trial {trial.number}")
    
    config = copy.deepcopy(base_config)
    
    config["gnn"]["n_layers"] = trial.suggest_int("gnn_n_layers", 1, 3)
    config["gnn"]["normalization"] = trial.suggest_categorical("gnn_normalization", ["none", "batchnorm", "layernorm", "l2"])
    config["gnn"]["activation"] = trial.suggest_categorical("gnn_activation", ["relu", "gelu"])
    config["gnn"]["dropout"] = trial.suggest_float("gnn_dropout", 0.0, 0.5)
    config["gnn"]["edge_dropout"] = trial.suggest_float("gnn_edge_dropout", 0.0, 0.5)
    config["gnn"]["residual"] = trial.suggest_categorical("gnn_residual", [True, False])
    
    config["loss"]["loss_function"] = trial.suggest_categorical("loss_function", ["bce", "mse", "info_nce"])
    
    config["negative_sampler"]["sampler"] = trial.suggest_categorical("negative_sampler", ["random", "all", "hard_top_k", "hard_sampling"])
    config["negative_sampler"]["n_negatives_per_context"] = trial.suggest_categorical("n_negatives_per_context", [3, 7, 15])
    
    config["optimizer"]["lr"] = trial.suggest_float("lr", 0.000001, 0.01, log=True)
    config["optimizer"]["weight_decay"] = trial.suggest_float("weight_decay", 0.000001, 0.01, log=True)
    
    config["training"]["batch_size"] = trial.suggest_categorical("batch_size", [16, 128, 512, 1024])
    config["training"]["gradient_accumulation_steps"] = trial.suggest_categorical("gradient_accumulation_steps", [1, 2, 4])

    config["undirected_premise_graph"] = trial.suggest_categorical("undirected_premise_graph", [True, False])
    
    dataset = copy.deepcopy(base_dataset).to(device=f"cuda:{gpu_id}")
    try:
        metrics = run_experiment(dataset, config, gpu_id, None)
        test_metrics = metrics["test_metrics"]
        val_metrics = metrics["best_val_metrics"]
        train_metrics = metrics["train_metrics"]

        trial.set_user_attr("R@10 (Test)", test_metrics["R@10"])
        trial.set_user_attr("R@10 (Val)", val_metrics["R@10"])

        trial.set_user_attr("R@10 (Train)", train_metrics["R@10"])
        trial.set_user_attr("R@1 (Train)", train_metrics["R@1"])
        trial.set_user_attr("MRR (Train)", train_metrics["MRR"])

        trial_logger.info(f"Trial {trial.number} completed - R@10 (Val): {val_metrics['R@10']:.4f}")
        return val_metrics["R@10"]
    except Exception as e:
        trial_logger.error(f"Error occurred for trial {trial.number}: {e}", exc_info=True)
        raise e

def optune(base_config: Dict[str, Any], base_dataset: LightweightGraphDataset, gpu_ids: List[int], storage: str, study_name: str):
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Optuna study: {study_name}")
    logger.info(f"Using GPUs: {gpu_ids}")

    data_config = get_data_config(base_dataset)
    base_config["data_config"] = data_config
    
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True
    )

    num_experiments_per_gpu = 1
    max_workers = num_experiments_per_gpu * len(gpu_ids)
    
    # Set multiprocessing start method to 'spawn' to avoid CUDA initialization errors
    mp_context = mp.get_context('spawn')
    executor = ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context)

    try:
        while True:
            futures_to_trial: Dict[Future[Any], optuna.Trial] = {}
            for _ in range(num_experiments_per_gpu):
                for gpu_id in gpu_ids:
                    trial = study.ask()
                    trial.set_user_attr("gpu_id", gpu_id)
                    future = executor.submit(objective, trial, base_config, base_dataset)
                    futures_to_trial[future] = trial
            
            for future in as_completed(futures_to_trial):
                trial = futures_to_trial[future]
                try:
                    result = future.result()
                    study.tell(trial, result)
                except Exception as e:
                    logger.error(f"Error occurred for trial {trial.number}: {e}", exc_info=True)
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Terminating all running trials...")
        executor.shutdown(wait=False)
        logger.info("Optimization terminated.")
        logger.info("Optimization interrupted by user.")

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    dataset = load_dataset()
    config = load_config("config.yaml")
    config["data_config"] = get_data_config(dataset)
    study_name = "random_embedding_study"
    optune(config, dataset, gpu_ids=[1, 2, 3], storage=f"sqlite:///{study_name}.db", study_name=study_name)