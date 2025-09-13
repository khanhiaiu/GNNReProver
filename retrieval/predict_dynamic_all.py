# retrieval/predict_dynamic_all.py
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
from tqdm import tqdm
from common import zip_strict

from retrieval.model import PremiseRetriever
from retrieval.gnn_model import GNNRetriever
from retrieval.gnn_datamodule import GNNDataModule


def main() -> None:
    """
    Main script for running dynamic, GNN-based premise retrieval in a single batch.

    This script constructs one giant graph for all contexts in the dataset,
    runs the GNN once, and then performs retrieval. This is useful for
    offline evaluation and interpretation as it mirrors the training setup.
    """
    parser = argparse.ArgumentParser(description="Script for single-batch dynamic GNN-based retrieval.")
    parser.add_argument("--config", type=str, required=True, help="Path to the hparams.yaml file saved during GNN training.")
    parser.add_argument("--retriever_ckpt_path", type=str, required=True, help="Path to the base ByT5 retriever checkpoint (HF name or local path).")
    parser.add_argument("--gnn_ckpt_path", type=str, required=True, help="Path to the trained GNN checkpoint (e.g., last.ckpt).")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data to predict on (e.g., data/leandojo_benchmark_4/random/).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the final prediction pickle file.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for the final scoring phase (not for GNN execution).")

    args = parser.parse_args()
    logger.info(f"Starting single-batch dynamic prediction with arguments: {args}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- 1. Load Configs and Set Up Models & Data ---
    logger.info(f"Loading training configuration from {args.config}")
    with open(args.config, 'r') as f:
        hparams = yaml.safe_load(f)
    data_hparams = hparams['data']

    # Load base retriever and GNN
    retriever = PremiseRetriever.load_hf(args.retriever_ckpt_path, 2048, device)
    gnn_model = GNNRetriever.load(args.gnn_ckpt_path, device, freeze=True)

    # Setup datamodule - this will also build the corpus and cache all embeddings
    datamodule = GNNDataModule(
        data_path=args.data_path,
        corpus_path=data_hparams['corpus_path'],
        retriever_ckpt_path=args.retriever_ckpt_path,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=0, # Not using dataloader for this script
        graph_dependencies_config=data_hparams['graph_dependencies_config'],
        attributes=data_hparams.get('attributes', {}),
        negative_mining=data_hparams.get('negative_mining', {'strategy': 'random', 'num_negatives': 0, 'num_in_file_negatives': 0}),
    )
    datamodule.setup(stage="predict")
    
    # --- 2. Get Final Premise Embeddings ---
    # We can reuse the retriever's on_predict_start logic for this part.
    # It will compute initial embeddings and then run the GNN to get final premise embeddings.
    logger.info("Computing final GNN-refined premise embeddings...")
    retriever.gnn_model = gnn_model
    trainer_mock = pl.Trainer(accelerator="auto", devices=1, logger=False)
    trainer_mock.datamodule = datamodule
    retriever.trainer = trainer_mock
    retriever.on_predict_start()
    final_premise_embs = retriever.corpus_embeddings.to(device) # Keep on GPU for scoring
    logger.info(f"Final premise embeddings computed. Shape: {final_premise_embs.shape}")

    # --- 3. Prepare Data for Giant Graph ---
    # The datamodule has already cached the initial context embeddings.
    all_dataset_examples = datamodule.ds_pred.data
    
    # Map each unique context string to its data
    context_str_to_data = {ex["context"].serialize(): {
        "lctx_indices": torch.tensor([datamodule.corpus.name2idx[n] for n in ex["lctx_premises"] if n in datamodule.corpus.name2idx], dtype=torch.long),
        "goal_indices": torch.tensor([datamodule.corpus.name2idx[n] for n in ex["goal_premises"] if n in datamodule.corpus.name2idx], dtype=torch.long),
    } for ex in all_dataset_examples}

    unique_context_strs = list(context_str_to_data.keys())
    unique_context_map = {s: i for i, s in enumerate(unique_context_strs)}

    logger.info(f"Found {len(unique_context_strs)} unique contexts across the dataset.")

    # Get initial context embeddings from the datamodule cache
    initial_context_embs_tensor = torch.stack([datamodule.context_embeddings[s] for s in unique_context_strs]).to(device)
    
    # Get neighbor info for each unique context
    all_lctx_neighbors = [context_str_to_data[s]["lctx_indices"] for s in unique_context_strs]
    all_goal_neighbors = [context_str_to_data[s]["goal_indices"] for s in unique_context_strs]
    
    # --- 4. Run GNN on Giant Graph to get all Context Embeddings ---
    logger.info("Constructing and running GNN on the giant augmented graph...")
    initial_premise_embs_tensor = datamodule.node_features.to(device)
    edge_index = datamodule.corpus.premise_dep_graph.edge_index.to(device)
    edge_attr = getattr(datamodule.corpus.premise_dep_graph, "edge_attr", None)
    if edge_attr is not None:
        edge_attr = edge_attr.to(device)

    final_context_embs = gnn_model.get_all_dynamic_context_embeddings(
        initial_premise_embs=initial_premise_embs_tensor,
        initial_context_embs=initial_context_embs_tensor,
        edge_index=edge_index,
        edge_attr=edge_attr,
        all_lctx_neighbor_indices=all_lctx_neighbors,
        all_goal_neighbor_indices=all_goal_neighbors,
    )
    logger.info(f"Final context embeddings computed. Shape: {final_context_embs.shape}")
    
    # --- 5. Perform Retrieval and Save Predictions ---
    logger.info("Performing nearest neighbor search for all examples...")
    all_predictions = []
    
    for i in tqdm(range(0, len(all_dataset_examples), args.batch_size), desc="Scoring"):
        batch_examples = all_dataset_examples[i : i + args.batch_size]
        
        # Look up the pre-computed final embeddings for this batch of contexts
        context_indices = [unique_context_map[ex["context"].serialize()] for ex in batch_examples]
        batch_context_embs = final_context_embs[context_indices]
        
        # Perform retrieval using the retriever's helper method
        retrieved_premises, scores = datamodule.corpus.get_nearest_premises(
            final_premise_embs,
            [ex["context"] for ex in batch_examples],
            batch_context_embs,
            retriever.num_retrieved,
        )
        
        # Format and collect results
        for (
            ex,
            premises,
            s,
        ) in zip_strict(
            batch_examples,
            retrieved_premises,
            scores,
        ):
            prediction_item = {
                "url": ex["url"],
                "commit": ex["commit"],
                "file_path": ex["file_path"],
                "full_name": ex["full_name"],
                "start": ex["start"],
                "tactic_idx": ex["tactic_idx"],
                "context": ex["context"],
                "all_pos_premises": ex["all_pos_premises"],
                "lctx_premises": ex["lctx_premises"],
                "goal_premises": ex["goal_premises"],
                "retrieved_premises": premises,
                "scores": s,
            }
            all_predictions.append(prediction_item)

    # --- 6. Save the results ---
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "wb") as f:
        pickle.dump(all_predictions, f)
    logger.info(f"Single-batch dynamic retrieval predictions saved to {args.output_path}")

if __name__ == "__main__":
    main()