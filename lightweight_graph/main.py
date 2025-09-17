import argparse
from loguru import logger
from dataset import LightweightGraphDataset

def main() -> None:
    parser = argparse.ArgumentParser(description="Create or load the LightweightGraphDataset.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save/load the cached lightweight dataset.")
    parser.add_argument("--data_path", type=str, default="data/leandojo_benchmark_4/random", help="Path to the original dataset (e.g., .../random). Required for first run.")
    parser.add_argument("--corpus_path", type=str, default="data/leandojo_benchmark_4/corpus.jsonl", help="Path to the corpus.jsonl file. Required for first run.")
    parser.add_argument("--retriever_ckpt_path", type=str, default="kaiyuy/leandojo-lean4-retriever-byt5-small", help="Path to the retriever model. Required for first run.")
    args = parser.parse_args()

    gnn_config = {
        'mode': 'custom', 'use_proof_dependencies': False,
        'signature_and_state': {'verbosity': 'clickable', 'distinguish_lctx_goal': True}
    }

    dataset = LightweightGraphDataset.load_or_create(
        save_dir=args.save_dir, data_path=args.data_path, corpus_path=args.corpus_path,
        retriever_ckpt_path=args.retriever_ckpt_path, gnn_config=gnn_config,
    )

    logger.info("Dataset loaded successfully. Statistics:")
    logger.info(f"  - Premise Embeddings Shape: {dataset.premise_embeddings.shape}")
    logger.info(f"  - Premise Edge Index Shape: {dataset.premise_edge_index.shape}, Attr Shape: {dataset.premise_edge_attr.shape}")
    logger.info(f"  - Context Embeddings Shape: {dataset.context_embeddings.shape}")
    logger.info(f"  - Context Edge Index Shape: {dataset.context_edge_index.shape}, Attr Shape: {dataset.context_edge_attr.shape}")
    logger.info(f"  - Context->Premise Labels Shape: {dataset.context_premise_labels.shape}")
    logger.info(f"  - Num Train Contexts: {dataset.train_mask.sum().item()}")
    logger.info(f"  - Num Val Contexts:   {dataset.val_mask.sum().item()}")
    logger.info(f"  - Num Test Contexts:  {dataset.test_mask.sum().item()}")
    logger.info(f"  - Edge Types Map: {dataset.edge_types_map}")
    logger.info(f"  - File Graph Info:")
    logger.info(f"    - Num Files: {len(dataset.file_idx_to_path_map)}")
    logger.info(f"    - Premise-to-File Map Shape: {dataset.premise_to_file_idx_map.shape}")
    logger.info(f"    - Context-to-File Map Shape: {dataset.context_to_file_idx_map.shape}")
    logger.info(f"    - File Dependency Edge Index Shape: {dataset.file_dependency_edge_index.shape}")

if __name__ == "__main__":
    main()