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
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple

from retrieval.model import PremiseRetriever
from retrieval.gnn_model import GNNRetriever
from retrieval.datamodule import RetrievalDataModule
from common import Premise, Corpus


def find_gnn_wins(gnn_preds: List[Dict[str, Any]], baseline_preds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Given predictions from GNN and a baseline, find entries where GNN's top-1 is correct
    but the baseline's is not.
    """
    # Create maps for efficient lookup
    key_fn = lambda p: (p["file_path"], p["full_name"], tuple(p["start"]), p["tactic_idx"])
    gnn_map = {key_fn(p): p for p in gnn_preds}
    baseline_map = {key_fn(p): p for p in baseline_preds}

    wins = []
    for key, gnn_pred in gnn_map.items():
        if key not in baseline_map:
            continue

        baseline_pred = baseline_map[key]
        
        # Ensure both predictions have retrieved premises and positive premises exist
        if not gnn_pred["retrieved_premises"] or not baseline_pred["retrieved_premises"] or not gnn_pred["all_pos_premises"]:
            continue

        gnn_top_1 = gnn_pred["retrieved_premises"][0]
        baseline_top_1 = baseline_pred["retrieved_premises"][0]
        positive_premises = set(gnn_pred["all_pos_premises"])

        is_gnn_correct = gnn_top_1 in positive_premises
        is_baseline_correct = baseline_top_1 in positive_premises

        if is_gnn_correct and not is_baseline_correct:
            wins.append(gnn_pred)
            
    return wins


def perform_loo_attribution(
    pred_item: Dict[str, Any],
    retriever: PremiseRetriever,
    datamodule: RetrievalDataModule,
    baseline_preds_map: Dict[Any, Dict[str, Any]],
) -> Tuple[str, List[Tuple[str, str, float]], Premise, Premise]:
    """
    Performs leave-one-out edge attribution for a given query (ghost node).
    """
    corpus = datamodule.corpus
    gnn_model = retriever.gnn_model
    device = next(gnn_model.parameters()).device
    gnn_dtype = next(gnn_model.parameters()).dtype

    # 1. Identify query, neighbors, and baseline's incorrect top-1 premise
    context = pred_item["context"]
    key = (context.path, context.theorem_full_name, tuple(pred_item["start"]), pred_item["tactic_idx"])
    baseline_pred = baseline_preds_map[key]
    baseline_top_1_premise = baseline_pred["retrieved_premises"][0]
    gnn_top_1_premise = pred_item["retrieved_premises"][0]

    all_neighbors = (
        [(p_name, "lctx") for p_name in pred_item["lctx_premises"]] +
        [(p_name, "goal") for p_name in pred_item["goal_premises"]]
    )

    # 2. Get GNN-refined embedding of the baseline's top-1 premise
    # This was pre-computed in `on_predict_start` and is stored in `corpus_embeddings`
    baseline_top_1_idx = corpus.name2idx[baseline_top_1_premise.full_name]
    baseline_top_1_emb = retriever.corpus_embeddings[baseline_top_1_idx].unsqueeze(0)

    # 3. Compute the original score with all neighbors present
    initial_context_emb = retriever._encode(
        *retriever.tokenizer([context.serialize()], return_tensors="pt", truncation=True).to_device(device).values()
    ).to(gnn_dtype)

    lctx_indices = [torch.tensor([corpus.name2idx[n] for n in pred_item["lctx_premises"] if n in corpus.name2idx])]
    goal_indices = [torch.tensor([corpus.name2idx[n] for n in pred_item["goal_premises"] if n in corpus.name2idx])]
    
    original_context_emb = gnn_model.get_dynamic_context_embedding(
        initial_context_embs=initial_context_emb,
        batch_lctx_neighbor_indices=lctx_indices,
        batch_goal_neighbor_indices=goal_indices,
        premise_layer_embeddings=retriever.premise_layer_embeddings,
    )
    original_score = F.cosine_similarity(original_context_emb, baseline_top_1_emb)

    # 4. Iterate through neighbors, removing one at a time
    results = []
    for i, (neighbor_name, r_type) in enumerate(all_neighbors):
        temp_lctx_names = list(pred_item["lctx_premises"])
        temp_goal_names = list(pred_item["goal_premises"])

        if r_type == "lctx":
            temp_lctx_names.remove(neighbor_name)
        else:
            temp_goal_names.remove(neighbor_name)
        
        # Recompute context embedding with the edge removed
        loo_lctx_indices = [torch.tensor([corpus.name2idx[n] for n in temp_lctx_names if n in corpus.name2idx])]
        loo_goal_indices = [torch.tensor([corpus.name2idx[n] for n in temp_goal_names if n in corpus.name2idx])]

        new_context_emb = gnn_model.get_dynamic_context_embedding(
            initial_context_embs=initial_context_emb,
            batch_lctx_neighbor_indices=loo_lctx_indices,
            batch_goal_neighbor_indices=loo_goal_indices,
            premise_layer_embeddings=retriever.premise_layer_embeddings,
        )
        new_score = F.cosine_similarity(new_context_emb, baseline_top_1_emb)
        
        delta = original_score - new_score
        results.append((neighbor_name, r_type, delta.item()))

    sorted_results = sorted(results, key=lambda item: item[2], reverse=True)
    return context.serialize(), sorted_results, gnn_top_1_premise, baseline_top_1_premise


def main() -> None:
    parser = argparse.ArgumentParser(description="Interpret GNN retriever predictions using Leave-One-Out attribution.")
    parser.add_argument("--gnn-config", type=str, required=True, help="Path to the hparams.yaml/config.yaml from GNN training.")
    parser.add_argument("--gnn-ckpt-path", type=str, required=True, help="Path to the trained GNN checkpoint (.ckpt).")
    parser.add_argument("--retriever-ckpt-path", type=str, required=True, help="Path to the base ByT5 retriever (HF name or local path).")
    parser.add_argument("--gnn-preds-path", type=str, required=True, help="Path to the GNN's prediction pickle file.")
    parser.add_argument("--baseline-preds-path", type=str, required=True, help="Path to the baseline model's prediction pickle file.")
    parser.add_argument("--num-interpret", type=int, default=5, help="Number of examples to interpret.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- 1. Load models and data using the GNN's training config for consistency ---
    logger.info(f"Loading GNN training configuration from {args.gnn_config}")
    with open(args.gnn_config, 'r') as f:
        hparams = yaml.safe_load(f)
    
    data_hparams = hparams.get('data', hparams) # Handle both old and new config formats
    logger.info(f"Instantiating DataModule with saved config: {data_hparams['graph_dependencies_config']}")
    
    # We need a dummy datamodule to build the corpus correctly.
    datamodule = RetrievalDataModule(
        data_path=os.path.dirname(data_hparams['corpus_path']),
        corpus_path=data_hparams['corpus_path'],
        model_name=args.retriever_ckpt_path,
        batch_size=1, eval_batch_size=1, num_workers=0, max_seq_len=2048,
        graph_dependencies_config=data_hparams['graph_dependencies_config'],
        num_negatives=0, num_in_file_negatives=0,
    )
    datamodule.setup('predict')

    # Load base retriever and GNN
    retriever = PremiseRetriever.load_hf(args.retriever_ckpt_path, 2048, device)
    gnn_model = GNNRetriever.load(args.gnn_ckpt_path, device, freeze=True)
    retriever.gnn_model = gnn_model
    
    # Pre-compute GNN-refined premise embeddings
    logger.info("Pre-computing GNN-refined premise embeddings for the entire corpus...")
    trainer_mock = pl.Trainer(accelerator="auto", devices=1, logger=False)
    trainer_mock.datamodule = datamodule
    retriever.trainer = trainer_mock
    retriever.on_predict_start()
    
    # --- 2. Load predictions and find cases where the GNN wins ---
    logger.info(f"Loading GNN predictions from {args.gnn_preds_path}")
    with open(args.gnn_preds_path, "rb") as f:
        gnn_preds = pickle.load(f)

    logger.info(f"Loading baseline predictions from {args.baseline_preds_path}")
    with open(args.baseline_preds_path, "rb") as f:
        baseline_preds = pickle.load(f)

    gnn_wins = find_gnn_wins(gnn_preds, baseline_preds)
    logger.info(f"Found {len(gnn_wins)} cases where GNN's top-1 is correct and baseline's is not.")

    key_fn = lambda p: (p["file_path"], p["full_name"], tuple(p["start"]), p["tactic_idx"])
    baseline_preds_map = {key_fn(p): p for p in baseline_preds}

    # --- 3. Perform and print LOO attribution analysis ---
    num_to_interpret = min(args.num_interpret, len(gnn_wins))
    if num_to_interpret == 0:
        logger.warning("No cases found to interpret. Exiting.")
        return

    logger.info(f"Performing Leave-One-Out attribution on {num_to_interpret} examples...")
    for i, pred_item in enumerate(gnn_wins[:num_to_interpret]):
        context_text, loo_results, gnn_top_1, baseline_top_1 = perform_loo_attribution(
            pred_item, retriever, datamodule, baseline_preds_map
        )
        
        print("\n" + "="*80)
        print(f"INTERPRETATION EXAMPLE #{i+1}")
        print("="*80)
        print(f"Proof State:\n{context_text}\n")
        print(f"  - GNN Correct Prediction: {gnn_top_1.full_name}")
        print(f"  - Baseline Incorrect Prediction: {baseline_top_1.full_name}\n")
        print("Leave-One-Out Attribution (Edges sorted by impact on score for baseline's incorrect prediction):")
        print(f"{'Neighbor Premise':<60} {'Relation':<10} {'Score Drop (Δ)':<20}")
        print(f"{'-'*60:<60} {'-'*10:<10} {'-'*20:<20}")
        for neighbor, r_type, delta in loo_results:
            print(f"{neighbor:<60} {r_type:<10} {delta:
<20.6f}")
        print("="*80)


if __name__ == "__main__":
    main()