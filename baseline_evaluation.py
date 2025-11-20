#!/usr/bin/env python3
"""
Baseline evaluation script that evaluates the performance using only the initial
embeddings from the language model without any training.

This script provides a baseline to compare against trained models.
"""

import torch
from torch import Tensor
import argparse
from typing import Dict, Any, Tuple
from tqdm import tqdm
import json
import os

from lightweight_graph.dataset import LightweightGraphDataset


class BaselineScorer:
    """Simple scorer that uses cosine similarity between embeddings."""
    
    def __init__(self, normalize: bool = True):
        self.normalize = normalize
    
    def forward(self, context_embeddings: Tensor, premise_embeddings: Tensor) -> Tensor:
        """
        Compute cosine similarity scores between contexts and premises.
        
        Args:
            context_embeddings: (batch_size, embedding_dim)
            premise_embeddings: (n_premises, embedding_dim)
            
        Returns:
            scores: (batch_size, n_premises)
        """
        if self.normalize:
            context_embeddings = torch.nn.functional.normalize(context_embeddings, p=2, dim=-1)
            premise_embeddings = torch.nn.functional.normalize(premise_embeddings, p=2, dim=-1)
        
        # Compute cosine similarity as dot product of normalized vectors
        scores = torch.matmul(context_embeddings, premise_embeddings.t())
        return scores


def calculate_metrics_batch(all_premise_scores_batch: Tensor, batch_labels: Tensor) -> Dict[str, Tensor]:
    """Calculate retrieval metrics for a batch."""
    targets = torch.zeros_like(all_premise_scores_batch, dtype=torch.bool)
    context_indices = batch_labels[0]
    premise_indices_correct = batch_labels[1]
    targets[context_indices, premise_indices_correct] = True

    context_premise_count = targets.sum(dim=1) 

    valid_contexts_mask = context_premise_count > 0
    if not valid_contexts_mask.any():
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
    """Generate batches from the dataset."""
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
        
        retrieved_labels_mask = torch.isin(dataset.context_premise_labels[0], batch_global_indices)
        retrieved_labels_global = dataset.context_premise_labels[:, retrieved_labels_mask]
        retrieved_labels = retrieved_labels_global.clone()
        retrieved_labels[0] = batch_global_to_local_map[retrieved_labels[0]]

        data: Dict[str, Any] = {
            "context_embeddings": batch_context_embeddings.float(),
            "context_to_file_idx_map": batch_context_file_indices,
            "context_theorem_pos": batch_context_theorem_pos,
            "premise_embeddings": dataset.premise_embeddings.float(),
            "retrieved_labels": retrieved_labels
        }
        yield data


class BaselineModel:
    """Baseline model that uses raw embeddings with cosine similarity."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.scorer = BaselineScorer(normalize=True)
    
    def evaluate_batch(self, batch_data: Dict[str, Any], dataset: LightweightGraphDataset) -> Dict[str, Tensor]:
        """Evaluate a single batch and return metrics."""
        with torch.no_grad():
            context_embeddings = batch_data["context_embeddings"]
            premise_embeddings = batch_data["premise_embeddings"]
            
            # Compute scores using cosine similarity
            scores = self.scorer.forward(context_embeddings, premise_embeddings)
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

    def evaluate(self, dataset: LightweightGraphDataset, split: str, batch_size: int = 2048) -> Dict[str, float]:
        """Evaluate the baseline model on a given split."""
        print(f"Evaluating baseline model on {split} split...")
        
        with torch.no_grad():
            mask = getattr(dataset, f"{split}_mask", None)
            if mask is None:
                raise ValueError(f"Invalid split: {split}")
            
            split_indices = mask.nonzero(as_tuple=False).view(-1)
            eval_generator = batchify_dataset(dataset, split_indices, batch_size)

            metrics_data: list[Dict[str, Tensor]] = []
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
        
        metrics: Dict[str, float] = {}
        for key, value in all_metrics_tensor.items():
            metrics[key] = value.mean().item()

        return metrics


def main():
    parser = argparse.ArgumentParser(description="Baseline evaluation using raw embeddings")
    parser.add_argument('--dataset-dir', type=str, default="lightweight_graph/data_novel_updated", 
                       help='Directory containing the dataset')
    parser.add_argument('--splits', nargs='+', default=['val', 'test'], 
                       choices=['train', 'val', 'test'], help='Splits to evaluate on')
    parser.add_argument('--batch-size', type=int, default=2048, help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default="cuda", help='Device to use for evaluation')
    parser.add_argument('--output-file', type=str, default="baseline_results.json", 
                       help='File to save results to')
    
    args = parser.parse_args()
    
    print(f"Loading dataset from: {args.dataset_dir}")
    dataset = LightweightGraphDataset.load_or_create(save_dir=args.dataset_dir)
    
    # Move dataset to device
    dataset.to(args.device)
    
    # Print dataset info
    print(f"\nDataset Statistics:")
    print(f"  Number of premises: {dataset.premise_embeddings.shape[0]}")
    print(f"  Number of contexts: {dataset.context_embeddings.shape[0]}")
    print(f"  Embedding dimension: {dataset.premise_embeddings.shape[1]}")
    print(f"  Train contexts: {dataset.train_mask.sum().item()}")
    print(f"  Val contexts: {dataset.val_mask.sum().item()}")
    print(f"  Test contexts: {dataset.test_mask.sum().item()}")
    
    # Create baseline model
    baseline_model = BaselineModel(device=args.device)
    
    # Evaluate on requested splits
    all_results = {}
    for split in args.splits:
        print(f"\n{'='*50}")
        print(f"Evaluating on {split.upper()} split")
        print(f"{'='*50}")
        
        metrics = baseline_model.evaluate(dataset, split, args.batch_size)
        all_results[split] = metrics
        
        print(f"\nBaseline Results for {split.upper()} split:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    
    # Save results
    print(f"\nSaving results to: {args.output_file}")
    with open(args.output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\nBaseline evaluation completed!")
    
    # Print summary
    print(f"\n{'='*60}")
    print("BASELINE SUMMARY")
    print(f"{'='*60}")
    for split in args.splits:
        results = all_results[split]
        print(f"{split.upper()}: R@1={results['R@1']:.4f}, R@10={results['R@10']:.4f}, MRR={results['MRR']:.4f}")


if __name__ == "__main__":
    main()