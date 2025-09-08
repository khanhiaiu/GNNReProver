"""Ligihtning module for the premise retriever."""

import os
import torch

# Monkeypatch torch.load before Lightning/DeepSpeed calls it
_orig_load = torch.load
def patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)

torch.load = patched_load


import pickle
import numpy as np
from tqdm import tqdm
from lean_dojo import Pos
from loguru import logger
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple, Union
from transformers import AutoModelForTextEncoding, AutoTokenizer

from common import (
    Premise,
    Context,
    Corpus,
    get_optimizers,
    load_checkpoint,
    zip_strict,
    cpu_checkpointing_enabled,
)
from retrieval.gnn_model import GNNRetriever


torch.set_float32_matmul_precision("medium")


class PremiseRetriever(pl.LightningModule):
    # Add a class-level counter for logging
    _predict_log_counter = 0
    _max_predict_logs = 5

    def __init__(
        self,
        model_name: str,
        lr: float,
        warmup_steps: int,
        max_seq_len: int,
        num_retrieved: int = 100,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.num_retrieved = num_retrieved
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModelForTextEncoding.from_pretrained(model_name)
        self.embeddings_staled = True

        self.gnn_model: Optional[GNNRetriever] = None
        self.premise_layer_embeddings: Optional[List[torch.FloatTensor]] = None

    @classmethod
    def load(cls, ckpt_path: str, device, freeze: bool) -> "PremiseRetriever":
        return load_checkpoint(cls, ckpt_path, device, freeze)

    @classmethod
    def load_hf(
        cls, ckpt_path: str, max_seq_len: int, device: int, dtype=None
    ) -> "PremiseRetriever":
        model = PremiseRetriever(ckpt_path, 0.0, 0, max_seq_len, 100).to(device).eval()
        if dtype is not None:
            return model.to(dtype)
        elif (
            model.dtype == torch.float32
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] >= 8
        ):
            return model.to(torch.bfloat16)
        else:
            return model

    def load_corpus(self, path_or_corpus: Union[str, Corpus], graph_config: Optional[Dict[str, Any]] = None) -> None:
        """Associate the retriever with a corpus."""
        if isinstance(path_or_corpus, Corpus):
            self.corpus = path_or_corpus
            self.corpus_embeddings = None
            self.embeddings_staled = True
            return

        path = path_or_corpus
        if path.endswith(".jsonl"):  # A raw corpus without embeddings.
            if graph_config is None:
                # Provide a default config if none is given, to avoid crashing simple scripts.
                logger.warning("graph_config not provided to load_corpus. Using a default configuration.")
                graph_config = {
                    'mode': 'custom',
                    'use_proof_dependencies': True,
                    'signature_and_state': {'verbosity': 'clickable', 'distinguish_lctx_goal': True}
                }
            self.corpus = Corpus(path, graph_config)
            self.corpus_embeddings = None
            self.embeddings_staled = True
        else:  # A corpus with pre-computed embeddings.
            indexed_corpus = pickle.load(open(path, "rb"))
            self.corpus = indexed_corpus.corpus
            self.corpus_embeddings = indexed_corpus.embeddings
            self.embeddings_staled = False

    @property
    def embedding_size(self) -> int:
        """Return the size of the feature vector produced by ``encoder``."""
        return self.encoder.config.hidden_size

    def _encode(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor
    ) -> torch.FloatTensor:
        """Encode a premise or a context into a feature vector."""
        logger.debug(f"PremiseRetriever._encode called. encoder.device: {self.encoder.device}")
        logger.debug(f"Input_ids device: {input_ids.device}")
        logger.debug(f"Attention_mask device: {attention_mask.device}")
        if cpu_checkpointing_enabled(self):
            hidden_states = torch.utils.checkpoint.checkpoint(
                self.encoder, input_ids, attention_mask, use_reentrant=False
            )[0]
        else:
            hidden_states = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            ).last_hidden_state

        # Masked average.
        lens = attention_mask.sum(dim=1)
        features = (hidden_states * attention_mask.unsqueeze(2)).sum(
            dim=1
        ) / lens.unsqueeze(1)

        # Normalize the feature vector to have unit norm.
        return F.normalize(features, dim=1)

    def forward(
        self,
        context_ids: torch.LongTensor,
        context_mask: torch.LongTensor,
        pos_premise_ids: torch.LongTensor,
        pos_premise_mask: torch.LongTensor,
        neg_premises_ids: torch.LongTensor,
        neg_premises_mask: torch.LongTensor,
        label: torch.LongTensor,
    ) -> torch.FloatTensor:
        """Compute the contrastive loss for premise retrieval."""
        # Encode the query and positive/negative documents.
        context_emb = self._encode(context_ids, context_mask)
        pos_premise_emb = self._encode(pos_premise_ids, pos_premise_mask)
        neg_premise_embs = [
            self._encode(ids, mask)
            for ids, mask in zip_strict(neg_premises_ids, neg_premises_mask)
        ]
        all_premise_embs = torch.cat([pos_premise_emb, *neg_premise_embs], dim=0)

        # Cosine similarities for unit-norm vectors are just inner products.
        similarity = torch.mm(context_emb, all_premise_embs.t())
        assert -1 <= similarity.min() <= similarity.max() <= 1
        loss = F.mse_loss(similarity, label)
        return loss

    ############
    # Training #
    ############

    def on_fit_start(self) -> None:
        if self.logger is not None:
            self.logger.log_hyperparams(self.hparams)
            logger.info(f"Logging to {self.trainer.log_dir}")

        self.corpus = self.trainer.datamodule.corpus
        self.corpus_embeddings = None
        self.embeddings_staled = True

    def training_step(self, batch: Dict[str, Any], _) -> torch.Tensor:
        loss = self(
            batch["context_ids"],
            batch["context_mask"],
            batch["pos_premise_ids"],
            batch["pos_premise_mask"],
            batch["neg_premises_ids"],
            batch["neg_premises_mask"],
            batch["label"],
        )
        self.log(
            "loss_train", loss, on_epoch=True, sync_dist=True, batch_size=len(batch)
        )
        return loss

    def on_train_batch_end(self, outputs, batch, _) -> None:
        """Mark the embeddings as staled after a training batch."""
        self.embeddings_staled = True

    def configure_optimizers(self) -> Dict[str, Any]:
        return get_optimizers(
            self.parameters(), self.trainer, self.lr, self.warmup_steps
        )

    ##############
    # Validation #
    ##############

    @torch.no_grad()
    def reindex_corpus(self, batch_size: int) -> None:
        """Re-index the retrieval corpus using the up-to-date encoder."""
        if not self.embeddings_staled:
            return
        logger.info("Re-indexing the retrieval corpus")
        #raise RuntimeError("Should not reindex when using gnn (this is a debug error to prevent reindexing)")

        self.corpus_embeddings = torch.zeros(
            len(self.corpus.all_premises),
            self.embedding_size,
            dtype=self.encoder.dtype,
            device=self.device,
        )

        for i in tqdm(range(0, len(self.corpus), batch_size)):
            batch_premises = self.corpus.all_premises[i : i + batch_size]
            tokenized_premises = self.tokenizer(
                [p.serialize() for p in batch_premises],
                padding="longest",
                max_length=self.max_seq_len,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            self.corpus_embeddings[i : i + batch_size] = self._encode(
                tokenized_premises.input_ids, tokenized_premises.attention_mask
            )

        self.embeddings_staled = False

    def on_validation_start(self) -> None:
        self.reindex_corpus(self.trainer.datamodule.eval_batch_size)

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Retrieve premises and calculate metrics such as Recall@K and MRR."""
        # Retrieval.
        context_emb = self._encode(batch["context_ids"], batch["context_mask"])
        assert not self.embeddings_staled
        retrieved_premises, _ = self.corpus.get_nearest_premises(
            self.corpus_embeddings,
            batch["context"],
            context_emb,
            self.num_retrieved,
        )

        # Evaluation & logging.
        recall = [[] for _ in range(self.num_retrieved)]
        MRR = []
        num_with_premises = 0

        for i, (all_pos_premises, premises) in enumerate(
            zip_strict(batch["all_pos_premises"], retrieved_premises)
        ):
            all_pos_premises = set(all_pos_premises)
            if len(all_pos_premises) == 0:
                continue
            else:
                num_with_premises += 1
            first_match_found = False

            for j in range(self.num_retrieved):
                TP = len(all_pos_premises.intersection(premises[: (j + 1)]))
                recall[j].append(float(TP) / len(all_pos_premises))
                if premises[j] in all_pos_premises and not first_match_found:
                    MRR.append(1.0 / (j + 1))
                    first_match_found = True
            if not first_match_found:
                MRR.append(0.0)

        recall = [100 * np.mean(_) for _ in recall]

        for j in range(self.num_retrieved):
            self.log(
                f"Recall@{j+1}_val",
                recall[j],
                on_epoch=True,
                sync_dist=True,
                batch_size=num_with_premises,
            )

        self.log(
            "MRR",
            np.mean(MRR),
            on_epoch=True,
            sync_dist=True,
            batch_size=num_with_premises,
        )

    ##############
    # Prediction #
    ##############

    def on_predict_start(self) -> None:
        # Reset the counter at the beginning of prediction
        PremiseRetriever._predict_log_counter = 0
        
        # 1. Get initial embeddings from the base retriever (ByT5).
        # We always re-index to ensure the correct base embeddings are used.
        logger.info("Re-indexing corpus with base retriever before GNN refinement.")
        self.corpus = self.trainer.datamodule.corpus
        self.corpus_embeddings = None
        self.embeddings_staled = True
        self.reindex_corpus(self.trainer.datamodule.eval_batch_size)

        self.premise_layer_embeddings = []

        # 2. If a GNN is present, refine premise embeddings and cache each layer's output.
        if self.gnn_model is not None:
            logger.info("GNN model found. Caching premise embeddings for each GNN layer...")
            
            # Ensure the GNN is on the correct device and get its expected dtype
            device = self.corpus_embeddings.device
            self.gnn_model.to(device)
            gnn_dtype = next(self.gnn_model.parameters()).dtype

            # Explicitly cast the initial embeddings to match the GNN's dtype
            initial_premise_embeddings = self.corpus_embeddings.to(gnn_dtype)
            logger.info(f"Casting initial premise embeddings from {self.corpus_embeddings.dtype} to {gnn_dtype} to match GNN.")
            
            edge_index = self.corpus.premise_dep_graph.edge_index.to(device)
            edge_attr = getattr(self.corpus.premise_dep_graph, "edge_attr", None)
            if edge_attr is not None:
                edge_attr = edge_attr.to(device)

            self.premise_layer_embeddings = self.gnn_model.compute_premise_layer_embeddings(
                initial_premise_embeddings, edge_index, edge_attr # Use the casted tensor
            )
            final_premise_embs = self.premise_layer_embeddings[-1]

            # Overwrite self.corpus_embeddings with the final GNN-refined embeddings for KNN search.
            if self.gnn_model.hparams.concat_with_original_embeddings:
                original_embs_norm = F.normalize(self.corpus_embeddings, p=2, dim=1)
                final_gnn_embs_norm = F.normalize(final_premise_embs, p=2, dim=1)
                # Ensure dtypes and devices match for concat
                original_embs_norm = original_embs_norm.to(final_gnn_embs_norm.device, final_gnn_embs_norm.dtype)
                concatenated_embs = torch.cat([final_gnn_embs_norm, original_embs_norm], dim=1)
                self.corpus_embeddings = F.normalize(concatenated_embs, p=2, dim=1)
            else:
                self.corpus_embeddings = F.normalize(final_premise_embs, p=2, dim=1)
            
            logger.info(f"Cached {len(self.premise_layer_embeddings)} sets of premise embeddings.")

        self.predict_step_outputs = []

    def predict_step(self, batch: Dict[str, Any], _):
        if self.gnn_model is not None:
            # EFFICIENT DYNAMIC GNN-AWARE RETRIEVAL
            initial_context_emb = self._encode(batch["context_ids"], batch["context_mask"])

            # --- START OF NEW LOGGING CODE ---
            if PremiseRetriever._predict_log_counter < PremiseRetriever._max_predict_logs:
                logger.info("\n" + "="*80)
                logger.info(f"--- PREDICTION SAMPLE LOG #{PremiseRetriever._predict_log_counter + 1} ---")
                
                # Log the proof state
                context_text = batch["context"][0].serialize()
                logger.info(f"Proof State:\n{context_text}")
                
                # Log Local Context (lctx) Premises
                lctx_premise_names = batch["lctx_premises"][0]
                if lctx_premise_names:
                    logger.info("  Local Context (lctx) Premises:")
                    for name in lctx_premise_names:
                        logger.info(f"    - {name}")
                else:
                    logger.info("  Local Context (lctx) Premises: [None]")

                # Log Goal Premises
                goal_premise_names = batch["goal_premises"][0]
                if goal_premise_names:
                    logger.info("  Goal Premises:")
                    for name in goal_premise_names:
                        logger.info(f"    - {name}")
                else:
                    logger.info("  Goal Premises: [None]")
                
                logger.info("="*80 + "\n")
                PremiseRetriever._predict_log_counter += 1
            # --- END OF NEW LOGGING CODE ---
            
            # --- START OF FIX ---
            # Explicitly cast the initial context embeddings to match the GNN's dtype.
            gnn_dtype = next(self.gnn_model.parameters()).dtype
            initial_context_emb_casted = initial_context_emb.to(gnn_dtype)
            # --- END OF FIX ---

            batch_lctx_neighbor_indices = [
                torch.tensor([self.corpus.name2idx[n] for n in names if n in self.corpus.name2idx], dtype=torch.long)
                for names in batch["lctx_premises"]
            ]
            batch_goal_neighbor_indices = [
                torch.tensor([self.corpus.name2idx[n] for n in names if n in self.corpus.name2idx], dtype=torch.long)
                for names in batch["goal_premises"]
            ]

            context_emb = self.gnn_model.get_dynamic_context_embedding(
                initial_context_embs=initial_context_emb_casted, # Use the casted tensor
                batch_lctx_neighbor_indices=batch_lctx_neighbor_indices,
                batch_goal_neighbor_indices=batch_goal_neighbor_indices,
                premise_layer_embeddings=self.premise_layer_embeddings,
            )
            
            # Retrieve against the pre-computed final premise embeddings.
            retrieved_premises, scores = self.corpus.get_nearest_premises(
                self.corpus_embeddings,
                batch["context"],
                context_emb,
                self.num_retrieved,
            )

        else:
            # ORIGINAL STATIC RETRIEVAL
            context_emb = self._encode(batch["context_ids"], batch["context_mask"])
            assert not self.embeddings_staled
            retrieved_premises, scores = self.corpus.get_nearest_premises(
                self.corpus_embeddings,
                batch["context"],
                context_emb,
                self.num_retrieved,
            )

        batch_outputs = []
        
        for (
            url,
            commit,
            file_path,
            full_name,
            start,
            tactic_idx,
            ctx,
            pos_premises,
            lctx_premises,
            goal_premises,
            premises,
            s,
        ) in zip_strict(
            batch["url"],
            batch["commit"],
            batch["file_path"],
            batch["full_name"],
            batch["start"],
            batch["tactic_idx"],
            batch["context"],
            batch["all_pos_premises"],
            batch["lctx_premises"],
            batch["goal_premises"],
            retrieved_premises,
            scores,
        ):
            prediction_item ={
                "url": url,
                "commit": commit,
                "file_path": file_path,
                "full_name": full_name,
                "start": start,
                "tactic_idx": tactic_idx,
                "context": ctx,
                "all_pos_premises": pos_premises,
                "lctx_premises": lctx_premises,
                "goal_premises": goal_premises,
                "retrieved_premises": premises,
                "scores": s,
            }
            self.predict_step_outputs.append(prediction_item) # Keep this for backward compatibility
            batch_outputs.append(prediction_item)  


        return batch_outputs

    def on_predict_epoch_end(self) -> None:

        if self.trainer.log_dir is not None:
            # We save separately in main.py now
            #path = os.path.join(self.trainer.log_dir, "predictions.pickle")
            #with open(path, "wb") as oup:
            #    pickle.dump(self.predict_step_outputs, oup)
            #logger.info(f"Retrieval predictions saved to {path}")

            self.predict_step_outputs.clear()

    @torch.no_grad()
    def retrieve(
        self,
        state: str,
        file_name: str,
        theorem_full_name: str,
        theorem_pos: Pos,
        k: int,
    ) -> Tuple[List[Premise], List[float]]:
        """Retrieve ``k`` premises from ``corpus`` using ``state`` and ``tactic_prefix`` as context."""
        self.reindex_corpus(batch_size=32)

        ctx = Context(file_name, theorem_full_name, theorem_pos, state)
        ctx_tokens = self.tokenizer(
            [ctx.serialize()],
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        context_emb = self._encode(
            ctx_tokens.input_ids.to(self.device),
            ctx_tokens.attention_mask.to(self.device),
        )

        if self.corpus_embeddings.device != context_emb.device:
            self.corpus_embeddings = self.corpus_embeddings.to(context_emb.device)
        if self.corpus_embeddings.dtype != context_emb.dtype:
            self.corpus_embeddings = self.corpus_embeddings.to(context_emb.dtype)

        retrieved_premises, scores = self.corpus.get_nearest_premises(
            self.corpus_embeddings,
            [ctx],
            context_emb,
            k,
        )
        assert len(retrieved_premises) == len(scores) == 1
        return retrieved_premises[0], scores[0]