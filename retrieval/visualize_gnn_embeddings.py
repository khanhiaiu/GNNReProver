# retrieval/visualize_gnn_embeddings.py

import os
import torch

# Monkeypatch torch.load before Lightning/DeepSpeed calls it to handle older checkpoints
_orig_load = torch.load
def patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)

torch.load = patched_load

import random
import argparse
import yaml
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS
from collections import Counter

from retrieval.model import PremiseRetriever
from retrieval.gnn_model import GNNRetriever
from retrieval.datamodule import RetrievalDataModule
from common import Corpus

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize the effect of GNN on a random sample of premise embeddings using MDS.")
    parser.add_argument("--gnn-config", type=str, required=True, help="Path to the hparams.yaml/config.yaml from GNN training.")
    parser.add_argument("--gnn-ckpt-path", type=str, required=True, help="Path to the trained GNN checkpoint (.ckpt).")
    parser.add_argument("--retriever-ckpt-path", type=str, required=True, help="Path to the base ByT5 retriever (HF name or local path).")
    # --- FIXED: Added explicit --data-path argument ---
    parser.add_argument("--data-path", type=str, required=True, help="Path to the data split directory (e.g., data/leandojo_benchmark_4/random/).")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the output plots.")
    parser.add_argument("--num-points", type=int, default=2000, help="Number of premises to randomly sample for the visualization.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 1. Load models and data using the GNN's training config for consistency ---
    logger.info(f"Loading GNN training configuration from {args.gnn_config}")
    with open(args.gnn_config, 'r') as f:
        hparams = yaml.safe_load(f)
    
    data_hparams = hparams.get('data', hparams)
    
    # --- FIXED: Use the command-line data_path for the DataModule ---
    datamodule = RetrievalDataModule(
        data_path=args.data_path, # Use the correct, user-provided path
        corpus_path=data_hparams['corpus_path'],
        model_name=args.retriever_ckpt_path,
        batch_size=1, eval_batch_size=1, num_workers=0, max_seq_len=2048,
        graph_dependencies_config=data_hparams['graph_dependencies_config'],
        negative_mining={'strategy': 'random', 'num_negatives': 0, 'num_in_file_negatives': 0}
    )
    datamodule.setup('predict')
    corpus = datamodule.corpus

    # --- 2. Generate "Before" and "After" Embeddings ---
    logger.info(f"Generating initial text embeddings using {args.retriever_ckpt_path}...")
    retriever = PremiseRetriever.load_hf(args.retriever_ckpt_path, 2048, device)
    retriever.load_corpus(corpus, graph_config=data_hparams['graph_dependencies_config'])
    with torch.no_grad():
        retriever.reindex_corpus(batch_size=4)
    initial_premise_embs = retriever.corpus_embeddings.float().cpu().numpy()

    logger.info(f"Loading trained GNN from {args.gnn_ckpt_path} to generate final embeddings...")
    gnn_model = GNNRetriever.load(args.gnn_ckpt_path, device, freeze=True)
    
    gnn_dtype = next(gnn_model.parameters()).dtype
    initial_embs_tensor = torch.from_numpy(initial_premise_embs).to(device, gnn_dtype)
    
    edge_index = corpus.premise_dep_graph.edge_index.to(device)
    edge_attr = getattr(corpus.premise_dep_graph, "edge_attr", None)
    if edge_attr is not None:
        edge_attr = edge_attr.to(device)
        
    final_premise_embs_tensor = gnn_model.forward_embeddings(initial_embs_tensor, edge_index, edge_attr)
    final_premise_embs = final_premise_embs_tensor.cpu().numpy()

    # --- 3. Take a Random Sample and Prepare for Plotting ---
    logger.info(f"Randomly sampling {args.num_points} premises for visualization...")
    num_total_premises = len(corpus.all_premises)
    if args.num_points > num_total_premises:
        logger.warning(f"num_points ({args.num_points}) is greater than total premises ({num_total_premises}). Using all premises.")
        args.num_points = num_total_premises
        
    indices_to_plot = random.sample(range(num_total_premises), args.num_points)
    
    module_labels = [os.path.basename(corpus.all_premises[i].path) for i in indices_to_plot]
    
    most_common_modules = [module for module, count in Counter(module_labels).most_common(10)]
    plot_labels = [label if label in most_common_modules else 'other' for label in module_labels]

    # --- 4. Perform MDS and Plot ---
    logger.info(f"Running MDS on {len(indices_to_plot)} premises...")
    mds = MDS(n_components=2, random_state=42, n_init=4, normalized_stress='auto', verbose=1, n_jobs=-1)

    initial_subset_embs = initial_premise_embs[indices_to_plot, :]
    initial_embs_2d = mds.fit_transform(initial_subset_embs)

    final_subset_embs = final_premise_embs[indices_to_plot, :]
    final_embs_2d = mds.fit_transform(final_subset_embs)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 14))

    for ax, embs_2d, title in [(ax1, initial_embs_2d, "Before GNN (Based on Text Similarity)"), 
                               (ax2, final_embs_2d, "After GNN (Graph-Refined)")]:
        
        sns.scatterplot(
            x=embs_2d[:, 0], y=embs_2d[:, 1], hue=plot_labels,
            palette="viridis", s=15, alpha=0.6, ax=ax, legend='auto'
        )
        
        ax.set_title(title, fontsize=20)
        ax.set_xlabel("MDS Dimension 1")
        ax.set_ylabel("MDS Dimension 2")
        ax.legend(title='Module', bbox_to_anchor=(1.05, 1), loc='upper left')

    fig.tight_layout(rect=[0, 0, 0.9, 1])
    output_path = os.path.join(args.output_dir, "gnn_embedding_visualization.png")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved combined visualization to {output_path}")


if __name__ == "__main__":
    main()