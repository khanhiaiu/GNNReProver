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
    parser = argparse.ArgumentParser(description="Script for dynamic GNN-based retrieval.")
    parser.add_argument("--retriever_ckpt_path", type=str, required=True, help="Path to the ByT5 retriever checkpoint.")
    parser.add_argument("--gnn_ckpt_path", type=str, required=True, help="Path to the GNN checkpoint.")
    parser.add_argument("--indexed_corpus_path", type=str, required=True, help="Path to the GNN-refined indexed corpus.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data (e.g., data/leandojo_benchmark_4/random).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the prediction pickle file.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    logger.info(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load the GNN-refined corpus (contains static premise embeddings).
    with open(args.indexed_corpus_path, "rb") as f:
        indexed_corpus = pickle.load(f)

    # 2. Load the main PremiseRetriever (the text encoder).
    # We load it via HF because it's just a frozen encoder for this task.
    retriever = PremiseRetriever.load_hf(args.retriever_ckpt_path, 2048, device)
    retriever.corpus = indexed_corpus.corpus
    retriever.corpus_embeddings = indexed_corpus.embeddings.to(device)
    retriever.embeddings_staled = False

    # 3. Load the GNN model and attach it to the retriever.
    gnn_model = GNNRetriever.load(args.gnn_ckpt_path, device, freeze=True)
    retriever.gnn_model = gnn_model

    # 4. Set up the datamodule to feed data to the retriever.
    # Note: We need a dummy model_name for the datamodule's tokenizer,
    # but the retriever already has its own correct tokenizer loaded.
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

    # 5. Run prediction.
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
    )
    predictions = trainer.predict(retriever, datamodule=datamodule)

    # The predictions are stored in retriever.predict_step_outputs
    with open(args.output_path, "wb") as f:
        pickle.dump(retriever.predict_step_outputs, f)
    logger.info(f"Dynamic retrieval predictions saved to {args.output_path}")


if __name__ == "__main__":
    main()