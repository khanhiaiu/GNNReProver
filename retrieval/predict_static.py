#!/usr/bin/env python3
"""
Script for running static (baseline) premise retrieval without GNN.

This script performs the same retrieval as the original LeanDojo retriever,
using only the pre-trained text encoder without any GNN enhancement.
"""

import os
import pickle
import argparse
from loguru import logger
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

import torch
# Monkeypatch torch.load before Lightning/DeepSpeed calls it to handle older checkpoints
_orig_load = torch.load
def patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = patched_load

from retrieval.model import PremiseRetriever
from retrieval.datamodule import RetrievalDataModule


def main():
    parser = argparse.ArgumentParser(description="Static premise retrieval (baseline, no GNN)")
    parser.add_argument("--retriever_ckpt_path", type=str, required=True, 
                       help="Path to the base ByT5 retriever checkpoint (Hugging Face model name or local path)")
    parser.add_argument("--indexed_corpus_path", type=str, required=True, 
                       help="Path to the pre-computed indexed corpus pickle file")
    parser.add_argument("--data_path", type=str, required=True, 
                       help="Path to the data directory (e.g., data/leandojo_benchmark_4/random/)")
    parser.add_argument("--output_path", type=str, required=True, 
                       help="Path to save the prediction results")
    parser.add_argument("--batch_size", type=int, default=64, 
                       help="Batch size for prediction")
    parser.add_argument("--num_workers", type=int, default=10, 
                       help="Number of workers for data loading")
    parser.add_argument("--max_seq_len", type=int, default=1024, 
                       help="Maximum sequence length")
    
    args = parser.parse_args()
    logger.info(f"Running static prediction with args: {args}")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_int = 0  # Default to GPU 0 for CUDA, or CPU device index
    logger.info(f"Using device: {device}")
    
    # Load the retriever model
    logger.info(f"Loading retriever from {args.retriever_ckpt_path}")
    model = PremiseRetriever.load_hf(
        ckpt_path=args.retriever_ckpt_path, 
        max_seq_len=args.max_seq_len,
        device=device_int
    )
    
    # Load the indexed corpus (this sets corpus_embeddings and marks embeddings as not staled)
    logger.info(f"Loading indexed corpus from {args.indexed_corpus_path}")
    model.load_corpus(args.indexed_corpus_path, graph_config={})
    corpus = model.corpus
    
    # Create data module for prediction
    logger.info(f"Setting up data module for {args.data_path}")
    datamodule = RetrievalDataModule(
        data_path=args.data_path,
        corpus_path=None,  # Not needed since we're loading indexed corpus
        model_name=args.retriever_ckpt_path,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,  # Use same batch size for eval
        num_workers=args.num_workers,
        num_negatives=7,  # Default value, not used in prediction
        num_in_file_negatives=0,  # Default value, not used in prediction
        graph_dependencies_config={},  # Empty config for static prediction
    )
    
    # Set corpus and setup for prediction stage
    datamodule.corpus = corpus
    datamodule.setup(stage="predict")
    
    # Set up trainer for prediction
    logger.info("Setting up Lightning trainer")
    log_dir = os.path.dirname(args.output_path)
    os.makedirs(log_dir, exist_ok=True)
    
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=CSVLogger(save_dir=log_dir, name="static_prediction"),
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
    )
    
    # Run prediction
    logger.info("Starting prediction...")
    predictions = trainer.predict(model, datamodule=datamodule)
    
    # Flatten results from all batches
    all_predictions = []
    if predictions:
        for batch_results in predictions:
            if batch_results:
                all_predictions.extend(batch_results)
    
    # Save predictions
    logger.info(f"Saving {len(all_predictions)} predictions to {args.output_path}")
    with open(args.output_path, "wb") as f:
        pickle.dump(all_predictions, f)
    
    logger.info("Static prediction completed successfully!")


if __name__ == "__main__":
    main()
