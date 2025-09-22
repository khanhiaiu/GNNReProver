import os
import torch
from torch import Tensor
from loguru import logger
from tqdm import tqdm
from typing import Optional, Dict, Any, List

from torch import LongTensor, FloatTensor


class LightweightGraphDataset:
    """
    Lightweight graph dataset for GNN training on formal theorem proving tasks.
    
    Dataset dimensions:
        n_premises: Number of mathematical premises (theorems, definitions, lemmas)
        n_contexts: Number of tactic instances (individual proof steps)
        n_files: Number of source files in the corpus
        embedding_dim: Dimension of node embeddings

    Attributes:
        premise_embeddings (Tensor): ReProver embeddings for all premises. Shape: (n_premises, embedding_dim)
        premise_edge_index (LongTensor): Directed adjacency list for the premise graph. Shape: (2, n_premise_edges)
        premise_edge_attr (LongTensor): Edge types for the premise graph. Shape: (n_premise_edges,)
        context_embeddings (Tensor): ReProver embeddings for all tactic instances. Shape: (n_contexts, embedding_dim)
        context_edge_index (LongTensor): Directed edges from premises to tactic instances (bipartite graph). Shape: (2, n_context_edges)
        context_edge_attr (LongTensor): Edge types for the premise-to-instance edges. Shape: (n_context_edges,)
        context_premise_labels (LongTensor): Ground-truth links from instances to correct answer premises. Shape: (2, n_labels)
        train_mask, val_mask, test_mask (Tensor): Boolean masks for data splits, applied to instances. Shape: (n_contexts,)
        premise_pos (Tensor): Start and end positions for each premise [start_line, start_col, end_line, end_col]. Shape: (n_premises, 4)
        context_theorem_pos (Tensor): Theorem position for each tactic instance [line, column]. Shape: (n_contexts, 2)
        premise_to_file_idx_map (LongTensor): Maps premise index to its file index. Shape: (n_premises,)
        context_to_file_idx_map (LongTensor): Maps tactic instance index to its file index. Shape: (n_contexts,)
        file_dependency_edge_index (Tensor): Transitive import graph between files. Shape: (2, n_file_edges)
        file_idx_to_path_map (List[str]): Maps file index to its string path. Length: n_files
        premise_idx_to_name_map (List[str]): Maps premise index to its full name string. Length: n_premises
        edge_types_map (Dict[str, int]): Maps edge type names to integer IDs (e.g., 'signature_lctx', 'signature_goal')
        
    Notes:
        - All edge indices follow PyTorch Geometric format: source nodes in row 0, target nodes in row 1
        - Premise and context nodes are separately indexed starting from 0
        - Labels tensor format: [context_indices, premise_indices] for positive premise retrieval
        - File paths are sorted alphabetically for consistent indexing
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
        premise_pos: Tensor,
        context_theorem_pos: Tensor,
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
        self.premise_pos = premise_pos
        self.context_theorem_pos = context_theorem_pos

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
            "premise_idx_to_name_map.pt", "edge_types_map.pt", "premise_pos.pt",
            "context_theorem_pos.pt",
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
        from retrieval.gnn_datamodule import GNNDataModule, RetrievalDataset
        logger.info("Instantiating GNNDataModule to process source data...")
        datamodule = GNNDataModule(
            data_path=data_path, corpus_path=corpus_path, retriever_ckpt_path=retriever_ckpt_path,
            batch_size=4, eval_batch_size=4, num_workers=4,
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
        premise_pos_data = [[p.start.line_nb, p.start.column_nb, p.end.line_nb, p.end.column_nb] for p in corpus.all_premises]
        premise_pos = torch.tensor(premise_pos_data, dtype=torch.long)
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
        context_theorem_pos = torch.full((num_instances, 2), -1, dtype=torch.long)

        # For edges and labels
        src_edges, dst_edges, attr_edges = [], [], []
        label_ctx_indices, label_p_indices = [], []
        lctx_id = edge_types_map.get('signature_lctx')
        goal_id = edge_types_map.get('signature_goal')

        for i, ex in enumerate(tqdm(all_examples, desc="Building instance tensors")):
            # 1. Get the pre-computed embedding for this instance's context string
            context_str = ex["context"].serialize()
            context_embeddings[i] = datamodule.context_embeddings[context_str]

            # 2. Map this instance to its file index and theorem position
            context_to_file_idx_map[i] = path_to_file_idx.get(ex["context"].path, -1)
            context_theorem_pos[i] = torch.tensor((ex["context"].theorem_pos.line_nb, ex["context"].theorem_pos.column_nb), dtype=torch.long)

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
        instance_key_to_idx = {
            (ex["file_path"], ex["full_name"], tuple(ex["start"]), ex["tactic_idx"]): i
            for i, ex in enumerate(all_examples)
        }

        for split, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
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
            "premise_pos": premise_pos, "context_theorem_pos": context_theorem_pos,
        }
        for name, data in data_to_save.items():
            torch.save(data, os.path.join(save_dir, f"{name}.pt"))
        return data_to_save