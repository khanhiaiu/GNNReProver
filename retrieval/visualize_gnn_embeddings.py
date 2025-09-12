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
from sklearn.manifold import MDS, TSNE
from collections import Counter

# UMAP is an optional dependency
from umap import UMAP
HAS_UMAP = True


from retrieval.model import PremiseRetriever
from retrieval.gnn_model import GNNRetriever
from retrieval.datamodule import RetrievalDataModule
from common import Corpus

def run_and_plot_reduction(
    method_name: str,
    reducer,
    initial_embs: np.ndarray,
    final_embs: np.ndarray,
    plot_labels: list,
    output_dir: str,
) -> None:
    """
    Runs a dimensionality reduction method and saves the resulting plot.

    Args:
        method_name (str): Name of the method (e.g., "MDS", "t-SNE", "UMAP").
        reducer: The instantiated dimensionality reduction object (e.g., MDS(), TSNE()).
        initial_embs (np.ndarray): The embeddings before GNN.
        final_embs (np.ndarray): The embeddings after GNN.
        plot_labels (list): List of labels for coloring the points.
        output_dir (str): Directory to save the plot.
    """
    logger.info(f"Running {method_name} on {initial_embs.shape[0]} premises...")

    # Run reduction on initial embeddings
    initial_embs_2d = reducer.fit_transform(initial_embs)

    # Run reduction on final embeddings
    final_embs_2d = reducer.fit_transform(final_embs)

    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 14))
    fig.suptitle(f'Premise Embeddings Visualized with {method_name}', fontsize=24)

    for ax, embs_2d, title in [(ax1, initial_embs_2d, "Before GNN (Based on Text Similarity)"),
                               (ax2, final_embs_2d, "After GNN (Graph-Refined)")]:

        sns.scatterplot(
            x=embs_2d[:, 0], y=embs_2d[:, 1], hue=plot_labels,
            palette="viridis", s=15, alpha=0.6, ax=ax, legend='auto'
        )

        ax.set_title(title, fontsize=20)
        ax.set_xlabel(f"{method_name} Dimension 1")
        ax.set_ylabel(f"{method_name} Dimension 2")
        ax.legend(title='Module', bbox_to_anchor=(1.05, 1), loc='upper left')

    fig.tight_layout(rect=[0, 0, 0.9, 0.95])  # Adjust rect for suptitle
    output_path = os.path.join(output_dir, f"{method_name.lower()}_embedding_visualization.png")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved {method_name} visualization to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize the effect of GNN on a random sample of premise embeddings using MDS, t-SNE, and UMAP.")
    parser.add_argument("--gnn-config", type=str, required=True, help="Path to the hparams.yaml/config.yaml from GNN training.")
    parser.add_argument("--gnn-ckpt-path", type=str, required=True, help="Path to the trained GNN checkpoint (.ckpt).")
    parser.add_argument("--retriever-ckpt-path", type=str, required=True, help="Path to the base ByT5 retriever (HF name or local path).")
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
    
    datamodule = RetrievalDataModule(
        data_path=args.data_path,
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
    
    initial_subset_embs = initial_premise_embs[indices_to_plot, :]
    final_subset_embs = final_premise_embs[indices_to_plot, :]

    # --- 4. Perform Dimensionality Reduction and Plot for each method ---

    # Method 1: MDS
    mds_reducer = MDS(n_components=2, random_state=42, n_init=4, normalized_stress='auto', verbose=1, n_jobs=-1)
    run_and_plot_reduction(
        "MDS", mds_reducer, initial_subset_embs, final_subset_embs, plot_labels, args.output_dir
    )

    # Method 2: t-SNE
    # Note: t-SNE can be sensitive to perplexity. 30 is a common default.
    tsne_reducer = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, verbose=1, n_jobs=-1)
    run_and_plot_reduction(
        "t-SNE", tsne_reducer, initial_subset_embs, final_subset_embs, plot_labels, args.output_dir
    )

    # Method 3: UMAP
    if HAS_UMAP:
        # Common defaults: n_neighbors=15, min_dist=0.1
        umap_reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1, verbose=True)
        run_and_plot_reduction(
            "UMAP", umap_reducer, initial_subset_embs, final_subset_embs, plot_labels, args.output_dir
        )


if __name__ == "__main__":
    main()