# Lightweight Graph Subproject

This subproject provides a streamlined, fast-loading graph dataset for GNN experiments, decoupled from the main project's complex data loading logic.

## Purpose

The main `retrieval` project involves intricate data processing steps, including on-the-fly text tokenization and embedding generation, which can be slow and memory-intensive. This subproject addresses that by:

1.  **Preprocessing:** Performing the heavy data processing once to extract the core graph structure.
2.  **Caching:** Saving the essential graph components—premise embeddings, context embeddings, edge indices, and data splits—into simple `torch.Tensor` files.
3.  **Fast Loading:** Providing a `LightweightGraphDataset` class that loads these tensors directly into memory, making subsequent GNN training and experimentation significantly faster.

## Usage

The `main.py` script demonstrates how to create or load the dataset. On the first run, it will process the original dataset and save the lightweight version to a specified directory.

```bash
# Make sure you are in the root directory of the ReProver project
# and your environment is activated.

# First run: Create and cache the lightweight dataset
python lightweight_graph/main.py \
    --data_path data/leandojo_benchmark_4/random \
    --corpus_path data/leandojo_benchmark_4/corpus.jsonl \
    --retriever_ckpt_path kaiyuy/leandojo-lean4-retriever-byt5-small \
    --save_dir lightweight_graph/data

# Subsequent runs: Load the cached dataset instantly
python lightweight_graph/main.py --save_dir lightweight_graph/data
```
