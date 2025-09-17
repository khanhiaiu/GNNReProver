import os
import torch
from torch import Tensor
from loguru import logger
from typing import Optional, Tuple, Dict, Any, List


class LightweightGraphDataset:
    """
    A lightweight, fast-loading graph dataset container for GNN training.

    This class holds the pre-processed graph data as simple torch Tensors,
    avoiding complex data loading logic during training.

    Attributes:
        premise_embeddings (Tensor): Embeddings for all premises.
        premise_edge_index (Tensor): Directed adjacency list for the premise graph.
        premise_edge_attr (Tensor): Edge types for the premise graph.
        context_embeddings (Tensor): Embeddings for all unique contexts.
        context_edge_index (Tensor): Directed edges from premises to contexts.
        context_edge_attr (Tensor): Edge types for the premise-to-context edges.
        context_premise_labels (Tensor): Ground-truth links from contexts to correct answer premises.
        train_mask, val_mask, test_mask (Tensor): Boolean masks for data splits.
        premise_to_file_idx_map (Tensor): Maps a premise index to its file index.
        context_to_file_idx_map (Tensor): Maps a context index to its file index.
        file_dependency_edge_index (Tensor): Transitive import graph between files.
        file_idx_to_path_map (List[str]): Maps a file index to its string path.
        premise_idx_to_name_map (List[str]): Maps a premise index to its full name string.
        edge_types_map (Dict[str, int]): Maps edge type names to integer IDs.
    """

    def __init__(
        self,
        premise_embeddings: Tensor,
        premise_edge_index: Tensor,
        premise_edge_attr: Tensor, # <-- ADDED
        context_embeddings: Tensor,
        context_edge_index: Tensor,
        context_edge_attr: Tensor, # <-- ADDED
        context_premise_labels: Tensor,
        train_mask: Tensor, val_mask: Tensor, test_mask: Tensor,
        premise_to_file_idx_map: Tensor,
        context_to_file_idx_map: Tensor,
        file_dependency_edge_index: Tensor,
        file_idx_to_path_map: List[str],
        premise_idx_to_name_map: List[str],
        edge_types_map: Dict[str, int], # <-- ADDED
    ):
        self.premise_embeddings = premise_embeddings
        self.premise_edge_index = premise_edge_index
        self.premise_edge_attr = premise_edge_attr # <-- ADDED
        self.context_embeddings = context_embeddings
        self.context_edge_index = context_edge_index
        self.context_edge_attr = context_edge_attr # <-- ADDED
        self.context_premise_labels = context_premise_labels
        self.train_mask, self.val_mask, self.test_mask = train_mask, val_mask, test_mask
        self.premise_to_file_idx_map = premise_to_file_idx_map
        self.context_to_file_idx_map = context_to_file_idx_map
        self.file_dependency_edge_index = file_dependency_edge_index
        self.file_idx_to_path_map = file_idx_to_path_map
        self.premise_idx_to_name_map = premise_idx_to_name_map
        self.edge_types_map = edge_types_map # <-- ADDED

    def to(self, device: torch.device) -> "LightweightGraphDataset":
        """Moves all tensor attributes to the specified device."""
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, Tensor):
                setattr(self, attr_name, attr_value.to(device))
        return self

    @classmethod
    def load_or_create(
        cls,
        save_dir: str,
        data_path: Optional[str] = None,
        corpus_path: Optional[str] = None,
        retriever_ckpt_path: Optional[str] = None,
        gnn_config: Optional[Dict[str, Any]] = None,
    ) -> "LightweightGraphDataset":
        os.makedirs(save_dir, exist_ok=True)
        filenames = {
            "premise_embeddings.pt", "premise_edge_index.pt", "premise_edge_attr.pt",
            "context_embeddings.pt", "context_edge_index.pt", "context_edge_attr.pt",
            "context_premise_labels.pt", "train_mask.pt", "val_mask.pt", "test_mask.pt",
            "premise_to_file_idx_map.pt", "context_to_file_idx_map.pt",
            "file_dependency_edge_index.pt", "file_idx_to_path_map.pt",
            "premise_idx_to_name_map.pt", "edge_types_map.pt",
        }

        if all(os.path.exists(os.path.join(save_dir, fn)) for fn in filenames):
            logger.info(f"Loading lightweight graph dataset from {save_dir}...")
            data_dict = { k.replace('.pt', ''): torch.load(os.path.join(save_dir, k)) for k in filenames }
            return cls(**data_dict)
        else:
            logger.info("Cached lightweight dataset not found. Creating from source...")
            if not all([data_path, corpus_path, retriever_ckpt_path, gnn_config]):
                raise ValueError("All source data paths and config must be provided for first-time creation.")
            data_dict = cls._create_from_source(data_path, corpus_path, retriever_ckpt_path, gnn_config, save_dir)
            return cls(**data_dict)

    @staticmethod
    def _create_from_source(
        data_path: str, corpus_path: str, retriever_ckpt_path: str, gnn_config: Dict[str, Any], save_dir: str
    ) -> Dict[str, Any]:
        # This import is only used for the one-time data creation process.
        from retrieval.gnn_datamodule import GNNDataModule, RetrievalDataset
        logger.info("Instantiating GNNDataModule to process source data...")
        datamodule = GNNDataModule(
            data_path=data_path, corpus_path=corpus_path, retriever_ckpt_path=retriever_ckpt_path,
            batch_size=128, eval_batch_size=128, num_workers=4,
            graph_dependencies_config=gnn_config,
            attributes={}, negative_mining={"strategy" : "random", "num_negatives": 0, "num_in_file_negatives": 0}
        )
        datamodule.setup(stage="predict")
        corpus = datamodule.corpus
        logger.info("Data processing complete.")

        # Premise Info
        premise_embeddings = datamodule.node_features.clone()
        premise_edge_index = corpus.premise_dep_graph.edge_index.clone()
        premise_edge_attr = corpus.premise_dep_graph.edge_attr.clone()
        premise_idx_to_name_map = [p.full_name for p in corpus.all_premises]
        edge_types_map = corpus.edge_types_map

        # File Info
        logger.info("Extracting file graph information...")
        file_paths = sorted(list(corpus.transitive_dep_graph.nodes()))
        path_to_file_idx = {path: i for i, path in enumerate(file_paths)}
        premise_to_file_idx_map = torch.tensor([path_to_file_idx[p.path] for p in corpus.all_premises], dtype=torch.long)
        dep_src, dep_dst = zip(*corpus.transitive_dep_graph.edges()) if corpus.transitive_dep_graph.edges() else ([], [])
        file_dependency_edge_index = torch.tensor([[path_to_file_idx[u] for u in dep_src], [path_to_file_idx[v] for v in dep_dst]], dtype=torch.long)

        # Context Info & Masks
        logger.info("Building unified context list, masks, and file map...")
        all_examples = datamodule.ds_pred.data
        unique_contexts = sorted(list(datamodule.context_embeddings.keys()))
        context_to_idx = {ctx: i for i, ctx in enumerate(unique_contexts)}
        num_contexts = len(unique_contexts)
        context_embeddings = torch.stack([datamodule.context_embeddings[s] for s in unique_contexts])
        train_mask, val_mask, test_mask = (torch.zeros(num_contexts, dtype=torch.bool) for _ in range(3))
        context_to_file_idx_map = torch.full((num_contexts,), -1, dtype=torch.long)

        for ex in all_examples:
            ctx_idx = context_to_idx.get(ex["context"].serialize())
            if ctx_idx is not None:
                context_to_file_idx_map[ctx_idx] = path_to_file_idx.get(ex["context"].path, -1)
        for split, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
            split_ds = RetrievalDataset([os.path.join(data_path, f"{split}.json")], corpus, 0, 0, 0, None, False, gnn_config)
            for ex in split_ds.data:
                ctx_idx = context_to_idx.get(ex["context"].serialize())
                if ctx_idx is not None: mask[ctx_idx] = True

        # Edge Info
        logger.info("Building context edges with attributes and ground-truth labels...")
        src_edges, dst_edges, attr_edges = [], [], []
        label_ctx_indices, label_p_indices = [], []
        lctx_id = edge_types_map.get('signature_lctx')
        goal_id = edge_types_map.get('signature_goal')
        for ex in all_examples:
            ctx_idx = context_to_idx.get(ex["context"].serialize())
            if ctx_idx is not None:
                for p_name in ex["lctx_premises"]:
                    p_idx = corpus.name2idx.get(p_name)
                    if p_idx is not None and lctx_id is not None: src_edges.append(p_idx); dst_edges.append(ctx_idx); attr_edges.append(lctx_id)
                for p_name in ex["goal_premises"]:
                    p_idx = corpus.name2idx.get(p_name)
                    if p_idx is not None and goal_id is not None: src_edges.append(p_idx); dst_edges.append(ctx_idx); attr_edges.append(goal_id)
                for pos_premise in ex["all_pos_premises"]:
                    p_idx = corpus.name2idx.get(pos_premise.full_name)
                    if p_idx is not None: label_ctx_indices.append(ctx_idx); label_p_indices.append(p_idx)
        
        context_edge_index = torch.tensor([src_edges, dst_edges], dtype=torch.long)
        context_edge_attr = torch.tensor(attr_edges, dtype=torch.long)
        context_premise_labels = torch.tensor([label_ctx_indices, label_p_indices], dtype=torch.long)
        
        # Save Data
        logger.info(f"Saving processed data to {save_dir}...")
        data_to_save = {
            "premise_embeddings": premise_embeddings, "premise_edge_index": premise_edge_index, "premise_edge_attr": premise_edge_attr,
            "context_embeddings": context_embeddings, "context_edge_index": context_edge_index, "context_edge_attr": context_edge_attr,
            "context_premise_labels": context_premise_labels, "train_mask": train_mask, "val_mask": val_mask, "test_mask": test_mask,
            "premise_to_file_idx_map": premise_to_file_idx_map, "context_to_file_idx_map": context_to_file_idx_map,
            "file_dependency_edge_index": file_dependency_edge_index, "file_idx_to_path_map": file_paths,
            "premise_idx_to_name_map": premise_idx_to_name_map, "edge_types_map": edge_types_map,
        }
        for name, data in data_to_save.items():
            torch.save(data, os.path.join(save_dir, f"{name}.pt"))
        return data_to_save