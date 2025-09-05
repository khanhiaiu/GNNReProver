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
from transformers import AutoConfig

from common import Corpus, IndexedCorpus
from retrieval.model import PremiseRetriever
from retrieval.gnn_model import GNNRetriever


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Script for creating an indexed corpus for retrieval."
    )
    parser.add_argument(
        "--ckpt_path", type=str, required=True, help="Path to the ByT5 retriever checkpoint. Used for config even in random mode."
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
    parser.add_argument(
        "--random-embeddings",
        action="store_true",
        help="If set, creates an index with random embeddings instead of using a model.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    logger.info(args)

    if not torch.cuda.is_available():
        logger.warning("Using CPU can be very slow for non-random indexing.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    default_graph_config = {
        "mode": "custom",
        "use_proof_dependencies": True,
        "signature_and_state": {
            "verbosity": "verbose",  # Use the most detailed verbosity for the index
            "distinguish_lctx_goal": True,
        },
    }

    corpus = Corpus(args.corpus_path, default_graph_config)
    final_embeddings = None

    if args.random_embeddings:
        logger.info("Generating RANDOM embeddings for the corpus.")
        
        # We still need ckpt_path to know the correct embedding dimension.
        config = AutoConfig.from_pretrained(args.ckpt_path)
        embedding_dim = config.hidden_size
        num_premises = len(corpus)
        
        logger.info(f"Corpus size: {num_premises}, Embedding dim: {embedding_dim}")

        # Generate and normalize random embeddings. Normalization is crucial
        # so that dot product is equivalent to cosine similarity.
        random_embeds = torch.randn(num_premises, embedding_dim)
        final_embeddings = F.normalize(random_embeds, p=2, dim=1)

    else:
        # 1. Get initial embeddings with the frozen text encoder.
        retriever = PremiseRetriever.load_hf(args.ckpt_path, 2048, device)
        retriever.load_corpus(corpus) # Pass the already loaded corpus object
        retriever.reindex_corpus(batch_size=args.batch_size)
        initial_embeddings = retriever.corpus_embeddings.to(device)

        # 2. Refine embeddings with the GNN if provided.
        if args.gnn_ckpt_path:
            logger.info(f"Loading GNN from {args.gnn_ckpt_path}")
            gnn_model = GNNRetriever.load(args.gnn_ckpt_path, device, freeze=True)
            edge_index = corpus.premise_dep_graph.edge_index.to(device)

            with torch.no_grad():
                logger.info(f"Initial embeddings dtype: {initial_embeddings.dtype}")
                logger.info(f"GNN model parameter dtype: {next(gnn_model.parameters()).dtype}")
                
                x = initial_embeddings.to(next(gnn_model.parameters()).dtype)
                logger.info(f"Converted embeddings dtype: {x.dtype}")
                
                final_embeddings = gnn_model.forward_embeddings(x, edge_index)
        else:
            logger.info("NO GNN provided, using initial text-encoder embeddings.")
            final_embeddings = initial_embeddings

    # 3. Save the final indexed corpus.
    pickle.dump(
        IndexedCorpus(corpus, final_embeddings.cpu()),
        open(args.output_path, "wb"),
    )
    logger.info(f"Indexed corpus saved to {args.output_path}")


if __name__ == "__main__":
    main()