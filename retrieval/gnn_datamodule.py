# retrieval/gnn_datamodule.py

import os
import torch
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
from tqdm import tqdm
from typing import Optional, List, Dict, Any
from torch.utils.data import DataLoader, Dataset

from common import Corpus, Batch
from retrieval.model import PremiseRetriever
from retrieval.datamodule import RetrievalDataset


# The collate function is now simpler. It receives pre-computed embeddings.
def gnn_collate_fn(
    examples: List[Dict[str, Any]],
    corpus: Corpus,
    node_features: torch.Tensor,
    context_embeddings: Dict[str, torch.Tensor], # Receives pre-computed embeddings
    num_negatives: int,
) -> Batch:
    batch = {}
    batch["node_features"] = node_features
    batch["edge_index"] = corpus.premise_dep_graph.edge_index
    batch["edge_attr"] = corpus.premise_dep_graph.edge_attr 
    
    # Look up pre-computed embeddings instead of generating them.
    batch["context_features"] = torch.stack(
        [context_embeddings[ex["context"].serialize()] for ex in examples]
    )

    # The rest of the function remains the same...
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
        premise_embeddings_path: str,
        retriever_ckpt_path: str,
        num_negatives: int,
        num_in_file_negatives: int,
        batch_size: int,
        eval_batch_size: int,
        num_workers: int,
        graph_dependencies: Dict[str, Any],
        context_neighbor_verbosity: str,
        attributes: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.num_negatives = num_negatives
        self.num_in_file_negatives = num_in_file_negatives
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.context_neighbor_verbosity = context_neighbor_verbosity
        self.retriever_ckpt_path = retriever_ckpt_path

        self.corpus = Corpus(corpus_path, graph_dependencies)
        # Expose edge_types_map at datamodule level for CLI linking
        self.edge_types_map = self.corpus.edge_types_map
        
        with open(premise_embeddings_path, "rb") as f:
            indexed_corpus = pickle.load(f)
        
        self.node_features = indexed_corpus.embeddings
        
        # Define the path for the context embeddings cache.
        retriever_name = self.retriever_ckpt_path.replace("/", "_")
        self.context_embeddings_path = os.path.join(self.data_path, f"context_embeddings_{retriever_name}.pt")
        self.context_embeddings = None

    def setup(self, stage: Optional[str] = None) -> None:
        # Load or generate context embeddings.
        if os.path.exists(self.context_embeddings_path):
            logger.info(f"Loading cached context embeddings from {self.context_embeddings_path}")
            self.context_embeddings = torch.load(self.context_embeddings_path)
        else:
            logger.info("Cached context embeddings not found. Generating them now...")
            self._generate_and_cache_context_embeddings()
            
        if stage in (None, "fit"):
            self.ds_train = RetrievalDataset(
                [os.path.join(self.data_path, "train.json")],
                self.corpus,
                self.num_negatives,
                self.num_in_file_negatives,
                max_seq_len=0,
                tokenizer=None,
                is_train=True,
                context_neighbor_verbosity=self.context_neighbor_verbosity,
            )
            
    def _generate_and_cache_context_embeddings(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        retriever = PremiseRetriever.load_hf(self.retriever_ckpt_path, 2048, device)
        
        # Temporarily create a dataset with all splits to find unique contexts.
        full_dataset = RetrievalDataset(
            [os.path.join(self.data_path, f"{split}.json") for split in ("train", "val", "test")],
            self.corpus, 0, 0, 0, None, is_train=False
        )

        unique_contexts = {ex["context"].serialize(): None for ex in full_dataset.data}
        context_list = list(unique_contexts.keys())
        logger.info(f"Found {len(context_list)} unique contexts to embed.")

        self.context_embeddings = {}
        batch_size = self.eval_batch_size

        with torch.no_grad():
            for i in tqdm(range(0, len(context_list), batch_size), desc="Preembedding contexts"):
                batch_contexts = context_list[i : i + batch_size]
                tokenized = retriever.tokenizer(
                    batch_contexts,
                    padding="longest",
                    max_length=retriever.max_seq_len,
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = tokenized.input_ids.to(device)
                attention_mask = tokenized.attention_mask.to(device)
                
                embeddings = retriever._encode(input_ids, attention_mask).cpu()

                for j, context_str in enumerate(batch_contexts):
                    self.context_embeddings[context_str] = embeddings[j]
        
        logger.info(f"Saving context embeddings to {self.context_embeddings_path}")
        torch.save(self.context_embeddings, self.context_embeddings_path)

    def train_dataloader(self) -> DataLoader:
        collate_fn_with_cache = functools.partial(
            gnn_collate_fn,
            corpus=self.corpus,
            node_features=self.node_features,
            context_embeddings=self.context_embeddings,
            num_negatives=self.num_negatives,
        )
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn_with_cache,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )