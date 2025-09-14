# retrieval/analyze_gate_activations.py

import os
import torch

# Monkeypatch torch.load to handle older checkpoints
_orig_load = torch.load
def patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = patched_load

import argparse
import yaml
import numpy as np
import functools
from loguru import logger
import pytorch_lightning as pl
from typing import Dict, Any, List
from torch.utils.data import DataLoader

from retrieval.model import PremiseRetriever
from retrieval.gnn_model import GNNRetriever
from retrieval.gnn_datamodule import GNNDataModule, gnn_collate_fn
from retrieval.datamodule import RetrievalDataset
from retrieval.gnn_modules.GatedRGCNConv import ResidualGatedRGCNConv as GatedRGCNConv

class GateAnalyzer(pl.LightningModule):
    """
    A wrapper LightningModule to perform gate activation analysis using the Trainer.
    """
    def __init__(self, retriever: PremiseRetriever):
        super().__init__()
        self.retriever = retriever
        self.gnn_model = retriever.gnn_model

    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> List[Dict[str, Any]]:
        # This step computes the final context embeddings, which triggers the GNN forward pass
        # and populates the `last_gate_activation` attribute we need.
        gnn_dtype = next(self.gnn_model.parameters()).dtype
        initial_context_emb_casted = batch["context_features"].to(gnn_dtype)

        batch_lctx_indices = [torch.tensor([self.retriever.corpus.name2idx[n] for n in names if n in self.retriever.corpus.name2idx], dtype=torch.long) for names in batch["lctx_premises"]]
        batch_goal_indices = [torch.tensor([self.retriever.corpus.name2idx[n] for n in names if n in self.retriever.corpus.name2idx], dtype=torch.long) for names in batch["goal_premises"]]

        # This call runs the modified gnn_utils function internally
        _ = self.gnn_model.get_dynamic_context_embedding(
            initial_context_embs=initial_context_emb_casted,
            batch_lctx_neighbor_indices=batch_lctx_indices,
            batch_goal_neighbor_indices=batch_goal_indices,
            premise_layer_embeddings=self.retriever.premise_layer_embeddings,
        )

        # Now, access the stored gate activations from each layer
        batch_results = []
        num_layers = self.gnn_model.hparams.num_layers
        
        for i in range(len(batch["context"])): # Iterate through items in the batch
            result = {"context_str": batch["context"][i].serialize()}
            for layer_idx, layer in enumerate(self.gnn_model.layers):
                if isinstance(layer, GatedRGCNConv) and hasattr(layer, "last_gate_activation"):
                    # The gate activation tensor has shape (batch_size, feature_size)
                    # We take the activation for the current item `i` and calculate its mean.
                    gate_activations_for_item = layer.last_gate_activation[i]
                    avg_activation = gate_activations_for_item.mean().item()
                    result[f"layer_{layer_idx}_avg_gate_activation"] = avg_activation
            batch_results.append(result)
            
        return batch_results

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze gate activations of a trained GatedRGCN model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the hparams.yaml/config.yaml from GNN training.")
    parser.add_argument("--retriever_ckpt_path", type=str, required=True, help="Path to the base ByT5 retriever (HF name or local path).")
    parser.add_argument("--gnn_ckpt_path", type=str, required=True, help="Path to the trained GNN checkpoint (.ckpt).")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data directory (e.g., data/leandojo_benchmark_4/random/).")
    # --- MODIFICATION: Add splits argument ---
    parser.add_argument("--splits", nargs='+', default=['train', 'val', 'test'], help="Which data splits to analyze (e.g., --splits val test).")
    # --- END MODIFICATION ---
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for analysis.")
    parser.add_argument("--num_workers", type=int, default=10, help="Number of workers for the dataloader.")
    args = parser.parse_args()
    logger.info(f"Starting gate activation analysis with arguments: {args}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    with open(args.config, 'r') as f:
        hparams = yaml.safe_load(f)
    data_hparams = hparams['data']

    retriever = PremiseRetriever.load_hf(args.retriever_ckpt_path, 2048, device)
    gnn_model = GNNRetriever.load(args.gnn_ckpt_path, device, freeze=True)
    retriever.gnn_model = gnn_model
    
    # This datamodule is now primarily used for setup: building the corpus and caching embeddings
    datamodule = GNNDataModule(
        data_path=args.data_path,
        corpus_path=data_hparams['corpus_path'],
        retriever_ckpt_path=args.retriever_ckpt_path,
        batch_size=args.batch_size, eval_batch_size=args.batch_size,
        num_workers=args.num_workers,
        graph_dependencies_config=data_hparams['graph_dependencies_config'],
        attributes=data_hparams.get('attributes', {}),
        negative_mining=data_hparams.get('negative_mining', {}),
    )
    # This caches premise and all context embeddings.
    datamodule.setup(stage='predict') 

    # Use a mock trainer to leverage on_predict_start for embedding setup
    mock_trainer = pl.Trainer(accelerator="auto", devices=1, logger=False)
    mock_trainer.datamodule = datamodule
    retriever.trainer = mock_trainer
    retriever.on_predict_start()

    analyzer_module = GateAnalyzer(retriever)
    
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto", # Automatically use available GPUs
        logger=False,
    )
    
    # --- MODIFICATION: Loop over the requested splits ---
    for split in args.splits:
        logger.info(f"\n{'='*20} Analyzing split: {split.upper()} {'='*20}")
        split_path = os.path.join(args.data_path, f"{split}.json")
        if not os.path.exists(split_path):
            logger.warning(f"Split file not found, skipping: {split_path}")
            continue

        # Manually create a dataset and dataloader for the specific split
        split_dataset = RetrievalDataset(
            data_paths=[split_path],
            corpus=datamodule.corpus,
            num_negatives=0, num_in_file_negatives=0, max_seq_len=0, tokenizer=None,
            is_train=False,
            graph_dependencies_config=datamodule.graph_dependencies_config,
        )

        collate_fn_with_cache = functools.partial(
            gnn_collate_fn,
            corpus=datamodule.corpus,
            node_features=datamodule.node_features,
            context_embeddings=datamodule.context_embeddings,
            is_train=False,
        )

        split_dataloader = DataLoader(
            split_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collate_fn_with_cache,
            shuffle=False,
        )

        logger.info(f"Starting analysis loop for {split} split...")
        list_of_batch_results = trainer.predict(analyzer_module, dataloaders=split_dataloader)
        
        # Flatten results from all GPUs/workers for this split
        all_results = [item for batch in list_of_batch_results for item in batch]
        logger.info(f"Analyzed {len(all_results)} contexts in the {split} split.")

        # Aggregate and print statistics for this split
        if not all_results:
            logger.warning(f"No results to analyze for the {split} split.")
            continue

        num_layers = gnn_model.hparams.num_layers
        for layer_idx in range(num_layers):
            metric_key = f"layer_{layer_idx}_avg_gate_activation"
            if metric_key in all_results[0]:
                activations = [r[metric_key] for r in all_results]
                avg = np.mean(activations)
                std = np.std(activations)
                p_gt_half = np.mean([1 if a > 0.5 else 0 for a in activations]) * 100
                
                logger.info("-" * 50)
                logger.info(f"Gate Activation Statistics for Layer {layer_idx} ({split.upper()} split)")
                logger.info(f"  - Overall Mean: {avg:.4f}")
                logger.info(f"  - Std. Dev:     {std:.4f}")
                logger.info(f"  - Percentage of contexts with gate > 0.5: {p_gt_half:.2f}%")
                logger.info("-" * 50)
    # --- END MODIFICATION ---

if __name__ == "__main__":
    main()