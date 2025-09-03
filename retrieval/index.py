import torch

# Monkeypatch torch.load before Lightning/DeepSpeed calls it
_orig_load = torch.load
def patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)

torch.load = patched_load

import pickle
import argparse
import torch.nn.functional as F
from loguru import logger

from common import IndexedCorpus
from retrieval.model import PremiseRetriever
from retrieval.gnn_model import GNNRetriever


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Script for training the BM25 premise retriever."
    )
    parser.add_argument(
        "--ckpt_path", type=str, required=True, help="Path to the ByT5 retriever checkpoint."
    )
    parser.add_argument(
        "--gnn_ckpt_path", type=str, help="Optional path to the GNN checkpoint."
    )
    parser.add_argument("--corpus-path", type=str, required=True)
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    logger.info(args)

    if not torch.cuda.is_available():
        logger.warning("Indexing the corpus using CPU can be very slow.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # 1. Get initial embeddings with the frozen text encoder.
    retriever = PremiseRetriever.load_hf(args.ckpt_path, 2048, device)
    retriever.load_corpus(args.corpus_path)
    retriever.reindex_corpus(batch_size=args.batch_size)
    initial_embeddings = retriever.corpus_embeddings.to(device)
    corpus = retriever.corpus

    # 2. Refine embeddings with the GNN if provided.
    if args.gnn_ckpt_path:
        logger.info(f"Loading GNN from {args.gnn_ckpt_path}")
        gnn_model = GNNRetriever.load(args.gnn_ckpt_path, device, freeze=True)
        edge_index = corpus.premise_dep_graph.edge_index.to(device)

        # GNN forward pass for inference
        with torch.no_grad():
            # Log dtype information for debugging
            logger.info(f"Initial embeddings dtype: {initial_embeddings.dtype}")
            logger.info(f"GNN model parameter dtype: {next(gnn_model.parameters()).dtype}")
            
            # Ensure dtype compatibility between initial embeddings and GNN model
            x = initial_embeddings.to(next(gnn_model.parameters()).dtype)
            logger.info(f"Converted embeddings dtype: {x.dtype}")
            
            for i, layer in enumerate(gnn_model.layers):
                x = layer(x, edge_index)
                if i < len(gnn_model.layers) - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=gnn_model.dropout_p, training=False)

            final_embeddings = x
    else:
        print("NO GNN provided, using initial embeddings.")
        final_embeddings = initial_embeddings

    # 3. Save the final indexed corpus.
    pickle.dump(
        IndexedCorpus(corpus, final_embeddings.to(torch.float32).cpu()),
        open(args.output_path, "wb"),
    )
    logger.info(f"Indexed corpus saved to {args.output_path}")


if __name__ == "__main__":
    main()