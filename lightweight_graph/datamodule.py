# lightweight_graph/datamodule.py

import torch
from torch.utils.data import Dataset
from typing import List, Dict
from functools import partial

# Import the main data store
from .dataset import LightweightGraphDataset

class ContextualGraphDataset(Dataset):
    """
    A PyTorch Dataset wrapper for our LightweightGraphDataset.

    This class provides a standard interface for a DataLoader. It considers each
    context in a given split as a single data item.
    The __getitem__ method returns the global index of a context.
    """
    def __init__(self, full_dataset: LightweightGraphDataset, split: str):
        super().__init__()
        self.full_dataset = full_dataset
        
        if split == "train":
            mask = self.full_dataset.train_mask
        elif split == "val":
            mask = self.full_dataset.val_mask
        elif split == "test":
            mask = self.full_dataset.test_mask
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")
            
        # self.split_indices stores the GLOBAL indices of the contexts for this split
        self.split_indices = mask.nonzero(as_tuple=True)[0]

    def __len__(self) -> int:
        return len(self.split_indices)

    def __getitem__(self, idx: int) -> int:
        """Returns the global context index for the given local dataset index."""
        return self.split_indices[idx].item()


def graph_collate_fn(batch_global_indices: List[int], full_dataset: LightweightGraphDataset) -> Dict[str, torch.Tensor]:
    """
    A custom collate function to construct a batch subgraph for the GNN.

    This function takes a list of global context indices, fetches the necessary
    data from the full dataset, and constructs the batch graph with shifted indices.
    """
    n_premises = full_dataset.premise_embeddings.size(0)
    device = full_dataset.premise_embeddings.device # Work on the same device as the dataset
    
    batch_global_indices = torch.tensor(batch_global_indices, dtype=torch.long, device=device)
    
    # --- Map global context indices to local (0 to batch_size-1) indices ---
    batch_global_to_local_map = torch.full(
        (batch_global_indices.max() + 1,), -1, dtype=torch.long, device=device
    )
    batch_global_to_local_map[batch_global_indices] = torch.arange(
        len(batch_global_indices), device=device
    )

    # --- Slice data for the batch ---
    batch_context_embeddings = full_dataset.context_embeddings[batch_global_indices]
    batch_context_file_indices = full_dataset.context_to_file_idx_map[batch_global_indices]
    
    # --- Filter edges and labels for this specific batch ---
    edge_mask = torch.isin(full_dataset.context_edge_index[1], batch_global_indices)
    batch_context_edge_index_global = full_dataset.context_edge_index[:, edge_mask]
    batch_context_edge_attr = full_dataset.context_edge_attr[edge_mask]
    
    label_mask = torch.isin(full_dataset.context_premise_labels[0], batch_global_indices)
    batch_labels_global = full_dataset.context_premise_labels[:, label_mask]

    # --- Shift indices for the subgraph ---
    batch_context_edge_index = batch_context_edge_index_global.clone()
    batch_context_edge_index[1] = batch_global_to_local_map[batch_context_edge_index[1]] + n_premises
    
    batch_labels = batch_labels_global.clone()
    batch_labels[0] = batch_global_to_local_map[batch_labels[0]]

    # --- Construct the final batch tensors ---
    all_batch_embeddings = torch.cat([full_dataset.premise_embeddings, batch_context_embeddings], dim=0)
    all_batch_edge_index = torch.cat([full_dataset.premise_edge_index, batch_context_edge_index], dim=1)
    all_batch_edge_attr = torch.cat([full_dataset.premise_edge_attr, batch_context_edge_attr], dim=0)

    return {
        "embeddings": all_batch_embeddings,
        "edge_index": all_batch_edge_index,
        "edge_attr": all_batch_edge_attr,
        "labels": batch_labels,
        "context_file_indices": batch_context_file_indices
    }