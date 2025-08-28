import os
import pickle
import torch
import pytorch_lightning as pl
from typing import Optional, List, Dict, Any
from torch.utils.data import DataLoader

from common import Corpus, Batch
from retrieval.model import PremiseRetriever
from retrieval.datamodule import RetrievalDataset


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
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.num_negatives = num_negatives
        self.num_in_file_negatives = num_in_file_negatives
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.corpus = Corpus(corpus_path)
        
        with open(embeddings_path, "rb") as f:
            indexed_corpus = pickle.load(f)
        
        # =================== THE FIX ===================
        # Access the embeddings tensor using dot notation for a dataclass
        self.node_features = indexed_corpus.embeddings
        # ===============================================

        self.retriever_ckpt_path = retriever_ckpt_path
        self.retriever = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.retriever is None:
            # Use load_hf here since we are loading from a Hugging Face model identifier
            self.retriever = PremiseRetriever.load_hf(
                self.retriever_ckpt_path, 2048, "cpu"
            )

        if stage in (None, "fit"):
            self.ds_train = RetrievalDataset(
                [os.path.join(self.data_path, "train.json")],
                self.corpus,
                self.num_negatives,
                self.num_in_file_negatives,
                max_seq_len=0,  # Not used for GNN
                tokenizer=None,  # Not used for GNN
                is_train=True,
            )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    @torch.no_grad()
    def collate(self, examples: List[Dict[str, Any]]) -> Batch:
        batch = {}

        # Static graph data (same for all batches)
        batch["node_features"] = self.node_features
        batch["edge_index"] = self.corpus.premise_dep_graph.edge_index

        # Get context embeddings on the fly using the frozen text encoder
        contexts = [ex["context"] for ex in examples]
        tokenized_contexts = self.retriever.tokenizer(
            [c.serialize() for c in contexts],
            padding="longest",
            max_length=self.retriever.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        batch["context_features"] = self.retriever._encode(
            tokenized_contexts.input_ids, tokenized_contexts.attention_mask
        )

        # Get neighbor indices for ghost nodes from `before_premises`
        neighbor_indices = []
        for ex in examples:
            indices = [
                self.corpus.name2idx[name]
                for name in ex["before_premises"]
                if name in self.corpus.name2idx
            ]
            neighbor_indices.append(torch.tensor(indices, dtype=torch.long))
        batch["neighbor_indices"] = neighbor_indices

        # Get positive and negative premise indices.
        pos_premise_indices = [
            self.corpus.name2idx[ex["pos_premise"].full_name] for ex in examples
        ]
        batch["pos_premise_indices"] = torch.tensor(pos_premise_indices, dtype=torch.long)

        batch["neg_premises_indices"] = []
        for i in range(self.num_negatives):
            neg_indices = [
                self.corpus.name2idx[ex["neg_premises"][i].full_name] for ex in examples
            ]
            batch["neg_premises_indices"].append(torch.tensor(neg_indices, dtype=torch.long))

        # Re-creating the label logic from `RetrievalDataset.collate`.
        batch_size = len(examples)
        label = torch.zeros(batch_size, batch_size * (1 + self.num_negatives))

        for j in range(batch_size):
            all_pos_premises_names = {p.full_name for p in examples[j]["all_pos_premises"]}
            for k in range(batch_size * (1 + self.num_negatives)):
                if k < batch_size:
                    p_name_k = examples[k]["pos_premise"].full_name
                else:
                    p_name_k = examples[k % batch_size]["neg_premises"][
                        k // batch_size - 1
                    ].full_name
                label[j, k] = float(p_name_k in all_pos_premises_names)

        batch["label"] = label
        return batch