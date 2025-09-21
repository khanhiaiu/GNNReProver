import os
import torch
from torch import Tensor
from loguru import logger
from tqdm import tqdm
from typing import Optional, Dict, Any, List

from torch import LongTensor, FloatTensor


class LightweightGraphDataset:
    """
    A lightweight, fast-loading graph dataset container for GNN training.

    This class holds the pre-processed graph data as simple torch Tensors,
    avoiding complex data loading logic during training. The "context" nodes in
    this dataset correspond to individual tactic instances, preserving duplicates
    to match the original ReProver evaluation methodology.

    Attributes:
        premise_embeddings (Tensor): Embeddings for all premises.
        premise_edge_index (Tensor): Directed adjacency list for the premise graph.
        premise_edge_attr (Tensor): Edge types for the premise graph.
        context_embeddings (Tensor): Embeddings for all tactic instances.
        context_edge_index (Tensor): Directed edges from premises to tactic instances.
        context_edge_attr (Tensor): Edge types for the premise-to-instance edges.
        context_premise_labels (Tensor): Ground-truth links from instances to correct answer premises.
        train_mask, val_mask, test_mask (Tensor): Boolean masks for data splits, applied to instances.
        premise_to_file_idx_map (Tensor): Maps a premise index to its file index.
        context_to_file_idx_map (Tensor): Maps a tactic instance index to its file index.
        file_dependency_edge_index (Tensor): Transitive import graph between files.
        file_idx_to_path_map (List[str]): Maps a file index to its string path.
        premise_idx_to_name_map (List[str]): Maps a premise index to its full name string.
        edge_types_map (Dict[str, int]): Maps edge type names to integer IDs.
    """

    def __init__(
        self,
        premise_embeddings: Tensor,
        premise_edge_index: LongTensor,
        premise_edge_attr: LongTensor,
        context_embeddings: Tensor,
        context_edge_index: LongTensor,
        context_edge_attr: LongTensor,
        context_premise_labels: LongTensor,
        train_mask: Tensor, val_mask: Tensor, test_mask: Tensor,
        premise_to_file_idx_map: LongTensor,
        context_to_file_idx_map: LongTensor,
        file_dependency_edge_index: Tensor,
        file_idx_to_path_map: List[str],
        premise_idx_to_name_map: List[str],
        edge_types_map: Dict[str, int],
    ):
        self.premise_embeddings = premise_embeddings
        self.premise_edge_index = premise_edge_index
        self.premise_edge_attr = premise_edge_attr
        self.context_embeddings = context_embeddings
        self.context_edge_index = context_edge_index
        self.context_edge_attr = context_edge_attr
        self.context_premise_labels = context_premise_labels
        self.train_mask, self.val_mask, self.test_mask = train_mask, val_mask, test_mask
        self.premise_to_file_idx_map = premise_to_file_idx_map
        self.context_to_file_idx_map = context_to_file_idx_map
        self.file_dependency_edge_index = file_dependency_edge_index
        self.file_idx_to_path_map = file_idx_to_path_map
        self.premise_idx_to_name_map = premise_idx_to_name_map
        self.edge_types_map = edge_types_map

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
        """
        Loads or creates the dataset where each "context" node corresponds to a
        tactic instance, not a unique proof state. This matches the original
        ReProver evaluation methodology.
        """
        instance_save_dir = save_dir + "_instances"
        os.makedirs(instance_save_dir, exist_ok=True)
        filenames = {
            "premise_embeddings.pt", "premise_edge_index.pt", "premise_edge_attr.pt",
            "context_embeddings.pt", "context_edge_index.pt", "context_edge_attr.pt",
            "context_premise_labels.pt", "train_mask.pt", "val_mask.pt", "test_mask.pt",
            "premise_to_file_idx_map.pt", "context_to_file_idx_map.pt",
            "file_dependency_edge_index.pt", "file_idx_to_path_map.pt",
            "premise_idx_to_name_map.pt", "edge_types_map.pt",
        }

        if all(os.path.exists(os.path.join(instance_save_dir, fn)) for fn in filenames):
            logger.info(f"Loading instance-based graph dataset from {instance_save_dir}...")
            data_dict = { k.replace('.pt', ''): torch.load(os.path.join(instance_save_dir, k)) for k in filenames }
            return cls(**data_dict)
        else:
            logger.info("Cached instance-based dataset not found. Creating from source...")
            if not all([data_path, corpus_path, retriever_ckpt_path, gnn_config]):
                raise ValueError("All source data paths and config must be provided for first-time creation.")
            data_dict = cls._create_from_source(data_path, corpus_path, retriever_ckpt_path, gnn_config, instance_save_dir)
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
        all_examples = datamodule.ds_pred.data
        num_instances = len(all_examples)
        logger.info(f"Data processing complete. Found {num_instances} total tactic instances.")

        # --- Premise and File Info ---
        premise_embeddings = datamodule.node_features.clone()
        premise_edge_index = corpus.premise_dep_graph.edge_index.clone()
        premise_edge_attr = corpus.premise_dep_graph.edge_attr.clone()
        premise_idx_to_name_map = [p.full_name for p in corpus.all_premises]
        edge_types_map = corpus.edge_types_map

        file_paths = sorted(list(corpus.transitive_dep_graph.nodes()))
        path_to_file_idx = {path: i for i, path in enumerate(file_paths)}
        premise_to_file_idx_map = torch.tensor([path_to_file_idx[p.path] for p in corpus.all_premises], dtype=torch.long)
        dep_src, dep_dst = zip(*corpus.transitive_dep_graph.edges()) if corpus.transitive_dep_graph.edges() else ([], [])
        file_dependency_edge_index = torch.tensor([[path_to_file_idx[u] for u in dep_src], [path_to_file_idx[v] for v in dep_dst]], dtype=torch.long)

        # --- Tactic Instance Info ---
        logger.info("Building tensors based on all tactic instances...")
        embedding_dim = datamodule.node_features.shape[1]
        context_embeddings_dtype = next(iter(datamodule.context_embeddings.values())).dtype
        context_embeddings = torch.zeros((num_instances, embedding_dim), dtype=context_embeddings_dtype)
        context_to_file_idx_map = torch.full((num_instances,), -1, dtype=torch.long)

        # For edges and labels
        src_edges, dst_edges, attr_edges = [], [], []
        label_ctx_indices, label_p_indices = [], []
        lctx_id = edge_types_map.get('signature_lctx')
        goal_id = edge_types_map.get('signature_goal')

        for i, ex in enumerate(tqdm(all_examples, desc="Building instance tensors")):
            # 1. Get the pre-computed embedding for this instance's context string
            context_str = ex["context"].serialize()
            context_embeddings[i] = datamodule.context_embeddings[context_str]

            # 2. Map this instance to its file index
            context_to_file_idx_map[i] = path_to_file_idx.get(ex["context"].path, -1)

            # 3. Build context edges (p -> instance)
            for p_name in ex["lctx_premises"]:
                p_idx = corpus.name2idx.get(p_name)
                if p_idx is not None and lctx_id is not None:
                    src_edges.append(p_idx); dst_edges.append(i); attr_edges.append(lctx_id)
            for p_name in ex["goal_premises"]:
                p_idx = corpus.name2idx.get(p_name)
                if p_idx is not None and goal_id is not None:
                    src_edges.append(p_idx); dst_edges.append(i); attr_edges.append(goal_id)

            # 4. Build ground-truth labels (instance -> p)
            for pos_premise in ex["all_pos_premises"]:
                p_idx = corpus.name2idx.get(pos_premise.full_name)
                if p_idx is not None:
                    label_ctx_indices.append(i); label_p_indices.append(p_idx)

        context_edge_index = torch.tensor([src_edges, dst_edges], dtype=torch.long)
        context_edge_attr = torch.tensor(attr_edges, dtype=torch.long)
        context_premise_labels = torch.tensor([label_ctx_indices, label_p_indices], dtype=torch.long)

        # --- Masks ---
        logger.info("Building train/val/test masks for instances...")
        train_mask, val_mask, test_mask = (torch.zeros(num_instances, dtype=torch.bool) for _ in range(3))
        # Create a unique key for each tactic instance to map it back
        instance_key_to_idx = {
            (ex["file_path"], ex["full_name"], tuple(ex["start"]), ex["tactic_idx"]): i
            for i, ex in enumerate(all_examples)
        }

        for split, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
            # We need to reload the split data to get the original list of examples for that split
            split_json_path = os.path.join(data_path, f"{split}.json")
            if os.path.exists(split_json_path):
                split_ds = RetrievalDataset([split_json_path], corpus, 0, 0, 0, None, False, gnn_config)
                for ex in split_ds.data:
                    key = (ex["file_path"], ex["full_name"], tuple(ex["start"]), ex["tactic_idx"])
                    instance_idx = instance_key_to_idx.get(key)
                    if instance_idx is not None:
                        mask[instance_idx] = True

        # --- Save Data ---
        logger.info(f"Saving processed instance-based data to {save_dir}...")
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