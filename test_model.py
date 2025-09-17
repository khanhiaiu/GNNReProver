#!/usr/bin/env python

import os
import sys
import torch
from lightweight_graph.dataset import LightweightGraphDataset
from tqdm import tqdm
from typing import Dict, Generator, Tuple
import optuna
import multiprocessing

# --- Project Root and Path Setup ---
project_root = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) if "scripts" in os.getcwd() or "notebooks" in os.getcwd() else os.getcwd()
if project_root not in sys.path:
    sys.path.insert(0, project_root)

SAVE_DIR = "lightweight_graph/data"

# --- Batching Logic (Unchanged) ---
def batchify_contexts(
    dataset: LightweightGraphDataset,
    split_indices: torch.Tensor,
    batch_size: int
) -> Generator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
    n_premises = dataset.premise_embeddings.size(0)

    split_edge_mask = torch.isin(dataset.context_edge_index[1], split_indices)
    split_context_edge_index = dataset.context_edge_index[:, split_edge_mask]
    split_context_edge_attr = dataset.context_edge_attr[split_edge_mask]

    split_label_mask = torch.isin(dataset.context_premise_labels[0], split_indices)
    split_context_premise_labels = dataset.context_premise_labels[:, split_label_mask]

    for start in range(0, len(split_indices), batch_size):
        end = min(start + batch_size, len(split_indices))
        batch_global_indices = split_indices[start:end]

        batch_global_to_local_map = torch.full(
            (batch_global_indices.max() + 1,), -1, dtype=torch.long, device=split_indices.device
        )
        batch_global_to_local_map[batch_global_indices] = torch.arange(
            len(batch_global_indices), device=split_indices.device
        )

        batch_context_embeddings = dataset.context_embeddings[batch_global_indices]
        batch_context_file_indices = dataset.context_to_file_idx_map[batch_global_indices]

        batch_edge_mask = torch.isin(split_context_edge_index[1], batch_global_indices)
        batch_context_edge_index_global = split_context_edge_index[:, batch_edge_mask]
        batch_context_edge_attr = split_context_edge_attr[batch_edge_mask]

        batch_label_mask = torch.isin(split_context_premise_labels[0], batch_global_indices)
        batch_labels_global = split_context_premise_labels[:, batch_label_mask]

        batch_context_edge_index = batch_context_edge_index_global.clone()
        batch_context_edge_index[1] = batch_global_to_local_map[batch_context_edge_index[1]] + n_premises

        batch_labels = batch_labels_global.clone()
        batch_labels[0] = batch_global_to_local_map[batch_labels[0]]

        all_batch_embeddings = torch.cat([dataset.premise_embeddings, batch_context_embeddings], dim=0)
        all_batch_edge_index = torch.cat([dataset.premise_edge_index, batch_context_edge_index], dim=1)
        all_batch_edge_attr = torch.cat([dataset.premise_edge_attr, batch_context_edge_attr], dim=0)

        yield all_batch_embeddings, all_batch_edge_index, all_batch_edge_attr, batch_labels, batch_context_file_indices

# --- Base Model Class (train_epoch removed) ---
class Model:
    def train_batch(self, batch_embeddings: torch.Tensor, batch_edge_index: torch.Tensor, batch_edge_attr: torch.Tensor, batch_labels: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        raise NotImplementedError("Subclasses must implement the training step.")

    def get_predictions(self, batch_embeddings: torch.Tensor, batch_edge_index: torch.Tensor, batch_edge_attr: torch.Tensor, num_batch_contexts: int, n_premises: int) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement the prediction logic.")

    @torch.no_grad()
    def eval_batch(self, batch_embeddings: torch.Tensor, batch_edge_index: torch.Tensor, batch_edge_attr: torch.Tensor, batch_labels: torch.Tensor, batch_context_file_indices: torch.Tensor, dataset: LightweightGraphDataset) -> Dict[str, float]:
        n_premises = dataset.premise_embeddings.shape[0]
        num_batch_contexts = batch_embeddings.shape[0] - n_premises
        scores = self.get_predictions(batch_embeddings, batch_edge_index, batch_edge_attr, num_batch_contexts, n_premises)

        accessible_mask = torch.zeros_like(scores, dtype=torch.bool)
        for i in range(num_batch_contexts):
            context_file_idx = batch_context_file_indices[i].item()
            in_file_mask = (dataset.premise_to_file_idx_map == context_file_idx)
            dependency_file_indices = dataset.file_dependency_edge_index[1, dataset.file_dependency_edge_index[0] == context_file_idx]
            imported_mask = torch.isin(dataset.premise_to_file_idx_map, dependency_file_indices)
            accessible_mask[i] = in_file_mask | imported_mask

        scores.masked_fill_(~accessible_mask, -torch.inf)

        gt_mask = torch.zeros_like(scores, dtype=torch.bool)
        gt_mask[batch_labels[0], batch_labels[1]] = True
        num_positives = gt_mask.sum(dim=1)
        valid_contexts = num_positives > 0
        if not valid_contexts.any(): return {'R@1': 0.0, 'R@10': 0.0, 'MRR': 0.0}

        top_10_indices = scores.topk(k=10, dim=1).indices
        top_10_hits = gt_mask.gather(1, top_10_indices)

        recall_at_1 = (top_10_hits[:, 0][valid_contexts] / num_positives[valid_contexts]).mean().item()
        recall_at_10 = (top_10_hits.sum(dim=1)[valid_contexts] / num_positives[valid_contexts]).mean().item()

        sorted_indices = scores.argsort(dim=1, descending=True)
        sorted_gt = gt_mask.gather(1, sorted_indices)
        first_hit_rank = torch.argmax(sorted_gt[valid_contexts].int(), dim=1) + 1
        mrr = (1.0 / first_hit_rank).mean().item()

        return {'R@1': recall_at_1, 'R@10': recall_at_10, 'MRR': mrr}


# --- Your Custom Model Classes ---
from numpy import negative
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from typing import Literal

def calculate_metrics(scores: torch.Tensor, gt_mask: torch.Tensor) -> float:
    num_positives = gt_mask.sum(dim=1)
    valid_contexts = num_positives > 0
    if not valid_contexts.any():
        return {"R@1": 0.0, "R@10": 0.0, "MRR": 0.0}

    top_10_indices = scores.topk(k=10, dim=1).indices
    top_10_hits = gt_mask.gather(1, top_10_indices)

    recall_at_1 = (top_10_hits[:, 0][valid_contexts] / num_positives[valid_contexts]).mean().item()
    recall_at_10 = (top_10_hits.sum(dim=1)[valid_contexts] / num_positives[valid_contexts]).mean().item()
    return {"R@1": recall_at_1, "R@10": recall_at_10}


class HeadAttentionScoring(nn.Module):
    # This class is unchanged
    def __init__(self, embedding_dim: int, num_heads: int, aggregation: Literal["logsumexp", "mean", "max", "gated"], w_depth: int = 1):
        super(HeadAttentionScoring, self).__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.aggregation = aggregation
        self.w_depth = w_depth

        if self.w_depth > 1:
            self.score_W = nn.ModuleList()
            for _ in range(self.w_depth):
                self.score_W.append(nn.Linear(embedding_dim, embedding_dim, bias=False))
        else:
            self.score_W = nn.Linear(embedding_dim, embedding_dim, bias=False)

        if aggregation == "gated":
            self.gate_W = nn.Linear(embedding_dim, num_heads)

    def forward(self, premise_embs: torch.Tensor, context_embs: torch.Tensor) -> torch.Tensor:
        batch_size = context_embs.size(0)
        n_premises = premise_embs.size(0)
        if self.w_depth > 1:
            initial_context_embs = context_embs
            for i, layer in enumerate(self.score_W):
                context_embs = layer(context_embs)
                if i < self.w_depth - 1:
                    context_embs = F.relu(context_embs)
            context_embs = context_embs + initial_context_embs
        else:
            context_embs = self.score_W(context_embs)
        context_embs = context_embs.view(batch_size, self.num_heads, self.embedding_dim // self.num_heads)
        premise_embs = premise_embs.view(n_premises, self.num_heads, self.embedding_dim // self.num_heads)
        scores = torch.einsum('bhd, phd -> bhp', context_embs, premise_embs)
        scores = scores.permute(0, 2, 1)
        if self.aggregation == "max": scores, _ = scores.max(dim=-1)
        elif self.aggregation == "mean": scores = scores.mean(dim=-1)
        elif self.aggregation == "logsumexp": scores = torch.logsumexp(scores, dim=-1)
        elif self.aggregation == "gated":
            score_gates = self.gate_W(context_embs.mean(dim=1))
            score_gates = F.softmax(score_gates, dim=-1)
            scores = (scores * score_gates.unsqueeze(1)).sum(dim=-1)
        else: raise ValueError(f"Unknown aggregation method: {self.aggregation}")
        return scores

class TestModel(Model, nn.Module):
    def __init__(
        self,
        dataset: LightweightGraphDataset,
        hidden_dim: int,
        aggregation: Literal["mean", "max", "logsumexp", "gated"],
        n_heads: int,
        lr: float = 1e-4,
        loss: Literal["bce", "mse"] = "mse",
        w_depth: int = 1,
        use_scheduler: bool = True
    ):
        Model.__init__(self)
        nn.Module.__init__(self)
        
        self.premise_embeddings_shape = dataset.premise_embeddings.shape
        self.embedding_dim = dataset.premise_embeddings.shape[1]
        self.hidden_dim = hidden_dim
        self.num_relations = len(dataset.edge_types_map)

        self.random_premise_embeds = nn.Embedding(dataset.premise_embeddings.shape[0], self.hidden_dim)
        self.random_premise_embed_for_context = nn.Embedding(dataset.premise_embeddings.shape[0], self.embedding_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = None
        if use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=100, min_lr=1e-6
            )
        self.loss = loss

        self.rgcn = RGCNConv(in_channels=self.embedding_dim, out_channels=self.hidden_dim, num_relations=2)
        self.scoring = HeadAttentionScoring(embedding_dim=self.hidden_dim, num_heads=n_heads, aggregation=aggregation, w_depth=w_depth)

    def forward(self, batch_embeddings: torch.Tensor, batch_edge_index: torch.Tensor, batch_edge_attr: torch.Tensor, n_premises: int) -> Tuple[torch.Tensor, torch.Tensor]:
        expected_dtype = torch.float32
        initial_premise_embs = self.random_premise_embeds.weight.to(expected_dtype)
        initial_context_embs = torch.zeros_like(batch_embeddings[n_premises:]).to(expected_dtype)
        initial_premise_emb_for_context = self.random_premise_embed_for_context.weight.to(expected_dtype)
        batch_embeddings_for_context = torch.cat([initial_premise_emb_for_context, initial_context_embs], dim=0)
        refined_context_embs = self.rgcn(batch_embeddings_for_context, batch_edge_index, batch_edge_attr)
        return initial_premise_embs, refined_context_embs[n_premises:]

    def get_predictions(self, batch_embeddings: torch.Tensor, batch_edge_index: torch.Tensor, batch_edge_attr: torch.Tensor, num_batch_contexts: int, n_premises: int, squash01: bool = True) -> torch.Tensor:
        final_premise_embs, final_context_embs = self.forward(batch_embeddings, batch_edge_index, batch_edge_attr, n_premises)
        scores = self.scoring(final_premise_embs, final_context_embs)
        if squash01: scores = torch.sigmoid(scores)
        return scores

    def train_batch(self, batch_embeddings: torch.Tensor, batch_edge_index: torch.Tensor, batch_edge_attr: torch.Tensor, batch_labels: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        self.train()
        self.optimizer.zero_grad()
        
        n_premises = self.premise_embeddings_shape[0]
        num_batch_contexts = batch_embeddings.shape[0] - n_premises
        logits_tensor = self.get_predictions(batch_embeddings, batch_edge_index, batch_edge_attr, num_batch_contexts, n_premises, squash01=False)

        targets_tensor = torch.zeros_like(logits_tensor)
        targets_tensor[batch_labels[0], batch_labels[1]] = 1.0

        metrics = calculate_metrics(logits_tensor.detach(), targets_tensor)

        n_negative = (targets_tensor == 0).sum().item()
        n_positive = (targets_tensor == 1).sum().item()

        weights = torch.ones_like(logits_tensor)
        if n_positive > 0 and n_negative > 0:
            weights[targets_tensor == 1] = n_negative / n_positive
        
        if self.loss == "mse":
            unweighted_loss = F.mse_loss(torch.sigmoid(logits_tensor), targets_tensor, reduction='none')
        else: # bce
            unweighted_loss = F.binary_cross_entropy_with_logits(logits_tensor, targets_tensor, reduction='none')
        
        weighted_loss = (unweighted_loss * weights).mean()

        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()

        return weighted_loss, metrics

    def overfit_single_batch(
        self,
        batch_data: Tuple,
        num_steps: int,
        scheduler_step_interval: int
    ) -> None:
        batch_embeddings, batch_edge_index, batch_edge_attr, batch_labels, _ = batch_data
        
        pbar = tqdm(range(num_steps), desc="Overfitting Single Batch", leave=False)
        interval_loss = 0.0
        
        for step_num in pbar:
            loss, metrics = self.train_batch(batch_embeddings, batch_edge_index, batch_edge_attr, batch_labels)
            
            current_loss = loss.item()
            interval_loss += current_loss

            log_dict = {"loss": f"{current_loss:.6f}"}
            log_dict.update(metrics)
            pbar.set_postfix(log_dict)

            if self.scheduler and (step_num + 1) % scheduler_step_interval == 0:
                self.scheduler.step(interval_loss / scheduler_step_interval)
                interval_loss = 0.0

# ==============================================================================
# ===== NEW CODE FOR HYPERPARAMETER SEARCH (USING overfit_single_batch) ======
# ==============================================================================

# --- Configuration for the Hyperparameter Search ---
GPUS_TO_USE = [1, 2] # Your available GPU indices
N_TOTAL_TRIALS = 50   # Total number of combinations to test
OVERFIT_STEPS = 2000  # Number of steps to overfit on the single batch
SCHEDULER_INTERVAL = 10
BATCH_SIZE = 1024
DB_FILENAME = "overfitting_aggregation_search.db"
STUDY_NAME = "rgcn_overfitting_aggregation_search"

optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_experiment(trial, device):
    """
    Runs a single experiment: overfits on one batch and evaluates on that same batch.
    """
    params = {
        'hidden_dim': trial.suggest_categorical("hidden_dim", [2048]),
        'lr': trial.suggest_categorical("lr", [1e-2]),
        'loss': trial.suggest_categorical("loss", ["bce"]),
        'w_depth': trial.suggest_categorical("w_depth", [2]),
        'use_scheduler': trial.suggest_categorical("use_scheduler", [False]),
        'aggregation': trial.suggest_categorical("aggregation", ["mean", "gated", "logsumexp", "max"]),
    }

    dataset = LightweightGraphDataset.load_or_create(save_dir=SAVE_DIR)
    dataset.to(device)

    model = TestModel(
        dataset=dataset,
        hidden_dim=params['hidden_dim'], lr=params['lr'], aggregation=params['aggregation'], n_heads=8,
        w_depth=params['w_depth'], loss=params['loss'], use_scheduler=params['use_scheduler']
    )
    model.to(device)

    # Get the first batch of training data
    train_indices = dataset.train_mask.nonzero(as_tuple=True)[0]
    train_generator = batchify_contexts(dataset, train_indices, BATCH_SIZE)
    first_batch = next(iter(train_generator), None)
    if first_batch is None:
        print("No training data available. Skipping trial.")
        return 0.0
    
    # Run the overfitting process on this single batch
    print(f"Trial {trial.number}: Overfitting with params {params} on {device}")
    model.overfit_single_batch(first_batch, OVERFIT_STEPS, SCHEDULER_INTERVAL)

    # After overfitting, evaluate performance on that same batch
    print(f"Trial {trial.number}: Final evaluation on the overfitted batch...")
    final_metrics = model.eval_batch(*first_batch, dataset)
    train_r_at_1 = final_metrics.get('R@1', 0.0)
    
    print(f"Trial {trial.number} finished. Final Overfitted Batch R@1: {train_r_at_1:.4f}")
    return train_r_at_1


def objective(trial, gpu_id):
    device = torch.device(f"cuda:{gpu_id}")
    try:
        return run_experiment(trial, device)
    except Exception as e:
        print(f"!!! Trial {trial.number} failed with error on {device}: {e}")
        return 0.0

def run_worker(gpu_id):
    print(f"Worker started for GPU {gpu_id}")
    study = optuna.load_study(study_name=STUDY_NAME, storage=f"sqlite:///{DB_FILENAME}")
    
    n_workers = len(GPUS_TO_USE)
    n_trials_per_worker = N_TOTAL_TRIALS // n_workers
    if gpu_id == GPUS_TO_USE[-1]: n_trials_per_worker += N_TOTAL_TRIALS % n_workers

    study.optimize(lambda trial: objective(trial, gpu_id), n_trials=n_trials_per_worker)
    print(f"Worker for GPU {gpu_id} finished.")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    study = optuna.create_study(
        study_name=STUDY_NAME, storage=f"sqlite:///{DB_FILENAME}",
        direction="maximize", load_if_exists=True
    )

    print(f"Starting hyperparameter search with {N_TOTAL_TRIALS} trials across GPUs: {GPUS_TO_USE}.")
    print(f"Objective: Find params that maximize R@1 on a single batch after {OVERFIT_STEPS} steps.")
    print(f"To view progress, run: optuna-dashboard sqlite:///{DB_FILENAME}")
    
    with multiprocessing.Pool(processes=len(GPUS_TO_USE)) as pool:
        pool.map(run_worker, GPUS_TO_USE)

    print("\n--- Hyperparameter Search for Overfitting Finished ---")
    print(f"Number of finished trials: {len(study.trials)}")
    print("\n--- Best Trial for Overfitting ---")
    best_trial = study.best_trial
    print(f"  Value (Overfitted Batch R@1): {best_trial.value:.6f}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")