# retrieval/predict_dynamic.py
import os
import torch

# Monkeypatch torch.load before Lightning/DeepSpeed calls it to handle older checkpoints
_orig_load = torch.load
def patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)

torch.load = patched_load


import pickle
import argparse
import yaml
from loguru import logger
import pytorch_lightning as pl

from retrieval.model import PremiseRetriever
from retrieval.gnn_model import GNNRetriever
from retrieval.gnn_datamodule import GNNDataModule


def main() -> None:
    """
    Main script for running dynamic, GNN-based premise retrieval.

    This script loads a trained GNN model and uses its saved hyperparameters
    to ensure that the data processing and graph construction for prediction
    are identical to those used during training.
    """
    parser = argparse.ArgumentParser(description="Script for dynamic GNN-based retrieval using saved training config.")
    parser.add_argument("--config", type=str, required=True, help="Path to the hparams.yaml file saved during GNN training.")
    parser.add_argument("--retriever_ckpt_path", type=str, required=True, help="Path to the base ByT5 retriever checkpoint (Hugging Face model name or local path).")
    parser.add_argument("--gnn_ckpt_path", type=str, required=True, help="Path to the trained GNN checkpoint (e.g., last.ckpt).")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data to predict on (e.g., data/leandojo_benchmark_4/random/).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the final prediction pickle file.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for prediction.")
    parser.add_argument("--num_workers", type=int, default=10, help="Number of workers for the dataloader.")
    args = parser.parse_args()
    logger.info(f"Starting dynamic prediction with arguments: {args}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load the exact hyperparameters saved during the GNN training run
    logger.info(f"Loading training configuration from {args.config}")
    with open(args.config, 'r') as f:
        hparams = yaml.safe_load(f)

    # The hparams file contains the 'data' and 'model' sections we need.
    # We primarily use the 'data' section to configure the DataModule correctly.
    data_hparams = hparams['data']
    logger.info(f"Using saved data hyperparameters: {data_hparams}")

    # --- Step 1: Setup the retriever and GNN model ---
    # Load the base text encoder (ByT5)
    retriever = PremiseRetriever.load_hf(args.retriever_ckpt_path, 2048, device)
    
    # Load the trained GNN model
    gnn_model = GNNRetriever.load(args.gnn_ckpt_path, device, freeze=True)
    
    # Attach the GNN to the retriever. The retriever's predict_step will use it.
    retriever.gnn_model = gnn_model
    logger.info("Successfully loaded and attached GNN model to the retriever.")

    graph_dependencies_config=data_hparams['graph_dependencies_config']
    logger.info(f"Using graph_dependencies_config for prediction: {graph_dependencies_config}")

    sig_cfg = graph_dependencies_config.get('signature_and_state', {})
    context_neighbor_verbosity = sig_cfg.get('verbosity', 'verbose')
    logger.info(f"Context neighbor verbosity set to: {context_neighbor_verbosity}")

    # --- Step 2: Setup the datamodule with the exact configuration from training ---
    # This is the crucial step for consistency. We use the loaded hyperparameters
    # to ensure the Corpus and data processing match the training environment.
    datamodule = GNNDataModule(
        data_path=args.data_path,  # Use data_path from args to allow predicting on different splits (e.g., test)
        corpus_path=data_hparams['corpus_path'],
        retriever_ckpt_path=args.retriever_ckpt_path,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=args.num_workers,
        graph_dependencies_config=graph_dependencies_config,
        # These are needed by the constructor but not used for prediction.
        # We get them from the config for consistency, with fallbacks.
        attributes=data_hparams.get('attributes', {}),
        negative_mining=data_hparams.get('negative_mining', {'strategy': 'random', 'num_negatives': 0, 'num_in_file_negatives': 0}),
    )

    # --- Step 3: Run prediction using the PyTorch Lightning Trainer ---
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        logger=False, # No need to log during prediction
    )
    
    logger.info("Starting prediction loop...")
    list_of_batch_results = trainer.predict(retriever, datamodule=datamodule)

    # The `predict` method returns a list of lists (one list of batch outputs per worker).
    # We need to flatten this into a single list of prediction dictionaries.
    all_predictions = [item for batch in list_of_batch_results for item in batch]
    logger.info(f"Generated a total of {len(all_predictions)} predictions.")

    # --- Step 4: Save the results ---
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "wb") as f:
        pickle.dump(all_predictions, f)
    logger.info(f"Dynamic retrieval predictions saved to {args.output_path}")


if __name__ == "__main__":
    main()