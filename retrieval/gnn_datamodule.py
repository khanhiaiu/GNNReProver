import os
import pickle
import torch

# Monkeypatch torch.load before Lightning/DeepSpeed calls it
_orig_load = torch.load
def patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)

torch.load = patched_load


import functools
import pytorch_lightning as pl
from loguru import logger
from typing import Optional, List, Dict, Any
from torch.utils.data import DataLoader

from common import Corpus, Batch
from retrieval.model import PremiseRetriever
from retrieval.datamodule import RetrievalDataset

# Global variable to hold the retriever in each worker process.
_WORKER_RETRIEVER = None

# Standalone collate function to avoid pickling the DataModule instance.
@torch.no_grad()
def gnn_collate_fn(
    examples: List[Dict[str, Any]],
    corpus: Corpus,
    node_features: torch.Tensor,
    retriever_ckpt_path: str,
    num_negatives: int,
) -> Batch:
    global _WORKER_RETRIEVER
    
    # Lazy initialization of the retriever in the worker process.
    if _WORKER_RETRIEVER is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing retriever in DataLoader worker on device: {device}")
        _WORKER_RETRIEVER = PremiseRetriever.load_hf(
            retriever_ckpt_path, 2048, device
        )

    batch = {}
    batch["node_features"] = node_features
    batch["edge_index"] = corpus.premise_dep_graph.edge_index
    batch["edge_attr"] = corpus.premise_dep_graph.edge_attr 

    # Get context embeddings using the worker-local retriever
    contexts = [ex["context"] for ex in examples]
    tokenized_contexts = _WORKER_RETRIEVER.tokenizer(
        [c.serialize() for c in contexts],
        padding="longest",
        max_length=_WORKER_RETRIEVER.max_seq_len,
        truncation=True,
        return_tensors="pt",
    )
    
    input_ids = tokenized_contexts.input_ids.to(_WORKER_RETRIEVER.encoder.device)
    attention_mask = tokenized_contexts.attention_mask.to(_WORKER_RETRIEVER.encoder.device)
    
    # THE FIX: Move the GPU-generated tensor back to the CPU before returning.
    batch["context_features"] = _WORKER_RETRIEVER._encode(input_ids, attention_mask).cpu()

    lctx_neighbor_indices = []
    for ex in examples:
        indices = [
            corpus.name2idx[name]
            for name in ex["lctx_premises"]
            if name in corpus.name2idx
        ]
        lctx_neighbor_indices.append(torch.tensor(indices, dtype=torch.long))
    batch["lctx_neighbor_indices"] = lctx_neighbor_indices

    goal_neighbor_indices = []
    for ex in examples:
        indices = [
            corpus.name2idx[name]
            for name in ex["goal_premises"]
            if name in corpus.name2idx
        ]
        goal_neighbor_indices.append(torch.tensor(indices, dtype=torch.long))
    batch["goal_neighbor_indices"] = goal_neighbor_indices

    pos_premise_indices = [
        corpus.name2idx[ex["pos_premise"].full_name] for ex in examples
    ]
    batch["pos_premise_indices"] = torch.tensor(pos_premise_indices, dtype=torch.long)

    batch["neg_premises_indices"] = []
    for i in range(num_negatives):
        neg_indices = [
            corpus.name2idx[ex["neg_premises"][i].full_name] for ex in examples
        ]
        batch["neg_premises_indices"].append(torch.tensor(neg_indices, dtype=torch.long))

    batch_size = len(examples)
    label = torch.zeros(batch_size, batch_size * (1 + num_negatives))

    for j in range(batch_size):
        all_pos_premises_names = {p.full_name for p in examples[j]["all_pos_premises"]}
        for k in range(batch_size * (1 + num_negatives)):
            if k < batch_size:
                p_name_k = examples[k]["pos_premise"].full_name
            else:
                p_name_k = examples[k % batch_size]["neg_premises"][
                    k // batch_size - 1
                ].full_name
            label[j, k] = float(p_name_k in all_pos_premises_names)

    batch["label"] = label
    return batch


class GNNDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        corpus_path: str,
        embeddings_path: str,
        retriever_ckpt_path: str,
        num_negatives: int,
        num_in_file_negatives: int,
        batch_size: int,
        eval_batch_size: int,
        num_workers: int,
        # Replace old params with the new config dict
        graph_dependencies: Dict[str, Any],
        context_neighbor_verbosity: str,
        attributes: Dict[str, Any], # Keep for future use
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.num_negatives = num_negatives
        self.num_in_file_negatives = num_in_file_negatives
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.context_neighbor_verbosity = context_neighbor_verbosity #
        
        # Pass graph construction params to Corpus
        self.corpus = Corpus(
            corpus_path,
            graph_dependencies,
        )
        
        with open(embeddings_path, "rb") as f:
            indexed_corpus = pickle.load(f)
        
        self.node_features = indexed_corpus.embeddings
        self.retriever_ckpt_path = retriever_ckpt_path
    
    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.ds_train = RetrievalDataset(
                [os.path.join(self.data_path, "train.json")],
                self.corpus,
                self.num_negatives,
                self.num_in_file_negatives,
                max_seq_len=0,
                tokenizer=None,
                is_train=True,
                context_neighbor_verbosity=self.context_neighbor_verbosity, # Pass param
            )
    
    def train_dataloader(self) -> DataLoader:
        collate_fn = functools.partial(
            gnn_collate_fn,
            corpus=self.corpus,
            node_features=self.node_features,
            retriever_ckpt_path=self.retriever_ckpt_path,
            num_negatives=self.num_negatives,
        )
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )