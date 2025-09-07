# retrieval/predict_dynamic.py
import os
import torch


# Monkeypatch torch.load before Lightning/DeepSpeed calls it
_orig_load = torch.load
def patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)

torch.load = patched_load


import pickle
import argparse
from loguru import logger
import pytorch_lightning as pl

from common import IndexedCorpus
from retrieval.model import PremiseRetriever
from retrieval.gnn_model import GNNRetriever
from retrieval.datamodule import RetrievalDataModule


def main() -> None:
    parser = argparse.ArgumentParser(description="Script for dynamic GNN-based or static retrieval.")
    parser.add_argument("--retriever_ckpt_path", type=str, required=True, help="Path to the ByT5 retriever checkpoint.")
    parser.add_argument("--gnn_ckpt_path", type=str, default=None, help="Optional path to the GNN checkpoint. If not provided, performs static retrieval.")
    parser.add_argument("--indexed_corpus_path", type=str, required=True, help="Path to the indexed corpus.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data (e.g., data/leandojo_benchmark_4/random).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the prediction pickle file.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    logger.info(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    retriever = PremiseRetriever.load_hf(args.retriever_ckpt_path, 2048, device)
    
    logger.info(f"Loading GNN from {args.gnn_ckpt_path} for dynamic retrieval.")
    gnn_model = GNNRetriever.load(args.gnn_ckpt_path, device, freeze=True)
    retriever.gnn_model = gnn_model

    datamodule = RetrievalDataModule(
        data_path=args.data_path,
        corpus_path=None, # Not needed, corpus is in retriever
        model_name=args.retriever_ckpt_path, # For tokenizer
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=args.num_workers,
        # Dummy values for training-specific args
        num_negatives=0,
        num_in_file_negatives=0,
        max_seq_len=2048,
    )
    datamodule.corpus = retriever.corpus
    datamodule.setup("predict")  # Explicitly setup datasets now that corpus is available

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
    )
    list_of_batch_results = trainer.predict(retriever, datamodule=datamodule)

    final_count = len(retriever.predict_step_outputs)
    logger.info("--- FINAL CHECK IN MAIN PROCESS ---")
    logger.info(f"The number of predictions available in the main process is: {final_count}")

    # We need to flatten this list of lists into a single list of predictions
    all_predictions = [item for batch in list_of_batch_results for item in batch]

    with open(args.output_path, "wb") as f:
        pickle.dump(all_predictions, f)
    logger.info(f"Dynamic retrieval predictions saved to {args.output_path}")


if __name__ == "__main__":
    main()