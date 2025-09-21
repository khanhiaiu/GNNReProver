import os
import sys
import torch
from lightweight_graph.dataset import LightweightGraphDataset
from tqdm import tqdm
from typing import Dict, Generator, Tuple
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from typing import Literal

project_root = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) if "scripts" in os.getcwd() or "notebooks" in os.getcwd() else os.getcwd()
if project_root not in sys.path:
    sys.path.insert(0, project_root)

SAVE_DIR = "lightweight_graph/data"
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

dataset = LightweightGraphDataset.load_or_create(save_dir=SAVE_DIR)
dataset.to(DEVICE)

print(f"Dataset loaded to {DEVICE}. Training contexts: {dataset.train_mask.sum().item()}")

def batchify_contexts(
    dataset: LightweightGraphDataset,
    split_indices: torch.Tensor,
    batch_size: int
) -> Generator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
    n_premises = dataset.premise_embeddings.size(0)
    
    split_edge_mask = torch.isin(dataset.context_edge_index[1], split_indices)
    split_context_edge_index = dataset.context_edge_index[:, split_edge_mask]
    split_context_edge_attr = dataset.context_edge_attr[split_edge_mask]
    
    split_label_mask = torch.isin(dataset.context_premise_labels[0], split_indices)
    split_context_premise_labels = dataset.context_premise_labels[:, split_label_mask]

    for start in range(0, len(split_indices), batch_size):
        end = min(start + batch_size, len(split_indices))
        batch_global_indices = split_indices[start:end]
        
        batch_global_to_local_map = torch.full(
            (batch_global_indices.max() + 1,), -1, dtype=torch.long, device=split_indices.device
        )
        batch_global_to_local_map[batch_global_indices] = torch.arange(
            len(batch_global_indices), device=split_indices.device
        )
        
        batch_context_embeddings = dataset.context_embeddings[batch_global_indices]
        batch_context_file_indices = dataset.context_to_file_idx_map[batch_global_indices]
        
        batch_edge_mask = torch.isin(split_context_edge_index[1], batch_global_indices)
        batch_context_edge_index_global = split_context_edge_index[:, batch_edge_mask]
        batch_context_edge_attr = split_context_edge_attr[batch_edge_mask]
        
        batch_label_mask = torch.isin(split_context_premise_labels[0], batch_global_indices)
        batch_labels_global = split_context_premise_labels[:, batch_label_mask]

        batch_context_edge_index = batch_context_edge_index_global.clone()
        batch_context_edge_index[1] = batch_global_to_local_map[batch_context_edge_index[1]] + n_premises
        
        batch_labels = batch_labels_global.clone()
        batch_labels[0] = batch_global_to_local_map[batch_labels[0]]

        all_batch_embeddings = torch.cat([dataset.premise_embeddings, batch_context_embeddings], dim=0)
        all_batch_edge_index = torch.cat([dataset.premise_edge_index, batch_context_edge_index], dim=1)
        all_batch_edge_attr = torch.cat([dataset.premise_edge_attr, batch_context_edge_attr], dim=0)

        yield all_batch_embeddings, all_batch_edge_index, all_batch_edge_attr, batch_labels, batch_context_file_indices


def compute_retrieval_metrics(scores: torch.Tensor, gt_mask: torch.Tensor) -> Dict[str, float]:
    num_positives = gt_mask.sum(dim=1)
    valid_contexts = num_positives > 0
    if not valid_contexts.any():
        return {'R@1': 0.0, 'R@10': 0.0, 'MRR': 0.0}

    scores = scores[valid_contexts]
    gt_mask = gt_mask[valid_contexts]
    num_positives = num_positives[valid_contexts]

    top_10_indices = scores.topk(k=10, dim=1).indices
    top_10_hits = gt_mask.gather(1, top_10_indices)

    recall_at_1 = (top_10_hits[:, 0] / num_positives).mean().item()
    recall_at_10 = (top_10_hits.sum(dim=1) / num_positives).mean().item()
    
    sorted_indices = scores.argsort(dim=1, descending=True)
    sorted_gt = gt_mask.gather(1, sorted_indices)
    first_hit_rank = torch.argmax(sorted_gt.int(), dim=1) + 1
    mrr = (1.0 / first_hit_rank).mean().item()
    
    return {'R@1': recall_at_1, 'R@10': recall_at_10, 'MRR': mrr}


class Model:
    def train_batch(self, batch_embeddings: torch.Tensor, batch_edge_index: torch.Tensor, batch_edge_attr: torch.Tensor, batch_labels: torch.Tensor, n_premises: int, step: int) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement the training step.")

    def train_epoch(self, dataset: LightweightGraphDataset, batch_size: int, accumulation_steps: int = 1) -> float:
        raise NotImplementedError("Subclasses must implement the training epoch logic.")

    def get_predictions(self, batch_embeddings: torch.Tensor, batch_edge_index: torch.Tensor, batch_edge_attr: torch.Tensor, num_batch_contexts: int, n_premises: int) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement the prediction logic.")

    @torch.no_grad()
    def eval_batch(self, batch_embeddings: torch.Tensor, batch_edge_index: torch.Tensor, batch_edge_attr: torch.Tensor, batch_labels: torch.Tensor, batch_context_file_indices: torch.Tensor, dataset: LightweightGraphDataset) -> Dict[str, float]:
        n_premises = dataset.premise_embeddings.shape[0]
        num_batch_contexts = batch_embeddings.shape[0] - n_premises
        scores = self.get_predictions(batch_embeddings, batch_edge_index, batch_edge_attr, num_batch_contexts, n_premises)

        accessible_mask = torch.zeros_like(scores, dtype=torch.bool)
        for i in range(num_batch_contexts):
            context_file_idx = batch_context_file_indices[i].item()
            in_file_mask = (dataset.premise_to_file_idx_map == context_file_idx)
            dependency_file_indices = dataset.file_dependency_edge_index[1, dataset.file_dependency_edge_index[0] == context_file_idx]
            imported_mask = torch.isin(dataset.premise_to_file_idx_map, dependency_file_indices)
            accessible_mask[i] = in_file_mask | imported_mask
        
        scores.masked_fill_(~accessible_mask, -torch.inf)
        
        gt_mask = torch.zeros_like(scores, dtype=torch.bool, device=scores.device)
        gt_mask[batch_labels[0], batch_labels[1]] = True
        
        return compute_retrieval_metrics(scores, gt_mask)

    @torch.no_grad()
    def eval(self, dataset: LightweightGraphDataset, split: str, batch_size: int) -> Dict[str, float]:
        mask = getattr(dataset, f"{split}_mask", None)
        if mask is None: raise ValueError(f"Invalid split: {split}")
        
        split_indices = mask.nonzero(as_tuple=True)[0]
        eval_generator = batchify_contexts(dataset, split_indices, batch_size)
        
        all_metrics = []
        pbar = tqdm(eval_generator, desc=f"Evaluating on {split} split")
        for batch_embeddings, batch_edge_index, batch_edge_attr, batch_labels, batch_context_file_indices in pbar:
            metrics = self.eval_batch(batch_embeddings, batch_edge_index, batch_edge_attr, batch_labels, batch_context_file_indices, dataset)
            all_metrics.append(metrics)
            pbar.set_postfix(metrics)

        if not all_metrics: return {'R@1': 0.0, 'R@10': 0.0, 'MRR': 0.0}
        
        final_metrics = {key: torch.tensor([m[key] for m in all_metrics if key in m]).mean().item() for key in all_metrics[0]}
        
        print(f"\n--- Evaluation Results for '{split}' split ---")
        print(f"  Recall@1:  {final_metrics['R@1']:.4f}")
        print(f"  Recall@10: {final_metrics['R@10']:.4f}")
        print(f"  MRR:       {final_metrics['MRR']:.4f}")
        print("------------------------------------------")

        return final_metrics


class HeadAttentionScoring(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, 
        preprocess : Literal["linear", "nn", "cosine", "none"],
        aggregation: Literal["logsumexp", "mean", "max", "gated"],
    ):
        super(HeadAttentionScoring, self).__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        assert embedding_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        if preprocess == "linear":
            self.score_W = nn.Linear(embedding_dim, embedding_dim, bias=False)
        elif preprocess =="nn":
            self.score_W = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim)
            )
        
        self.aggregation = aggregation
        self.preprocess = preprocess

        if aggregation == "gated":
            self.gate_W = nn.Linear(embedding_dim, num_heads)

    def forward(self, premise_embs: torch.Tensor, context_embs: torch.Tensor) -> torch.Tensor:
        batch_size = context_embs.size(0)
        n_premises = premise_embs.size(0)

        if self.preprocess == "nn" or self.preprocess == "linear":
            context_embs = self.score_W(context_embs)

        context_embs_headed = context_embs.view(batch_size, self.num_heads, self.embedding_dim // self.num_heads)
        premise_embs_headed = premise_embs.view(n_premises, self.num_heads, self.embedding_dim // self.num_heads)
        
        if self.preprocess == "cosine":
            context_embs_headed = F.normalize(context_embs_headed, p=2, dim=-1)
            premise_embs_headed = F.normalize(premise_embs_headed, p=2, dim=-1)

        scores_by_head = torch.einsum('bhd, phd -> bhp', context_embs_headed, premise_embs_headed)
        scores_by_head = scores_by_head.permute(0, 2, 1)
        
        if self.aggregation == "max":
            scores, _ = scores_by_head.max(dim=-1)
        elif self.aggregation == "mean":
            scores = scores_by_head.mean(dim=-1)
        elif self.aggregation == "logsumexp":
            scores = torch.logsumexp(scores_by_head, dim=-1)
        elif self.aggregation == "gated":
            score_gates = self.gate_W(context_embs)
            score_gates = F.softmax(score_gates, dim=-1)
            scores = (scores_by_head * score_gates.unsqueeze(1)).sum(dim=-1)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")

        assert scores.shape == (batch_size, n_premises)
        return scores

class TestModel(Model, nn.Module):
    def __init__(
        self,
        dataset: LightweightGraphDataset,
        hidden_dim: int,
        preprocess : Literal["linear", "nn", "cosine", "none"],
        aggregation: Literal["mean", "logsumexp", "max", "gated"],
        n_heads: int,
        lr: float = 1e-4,
        loss : Literal["bce", "mse"] = "mse",
    ):
        Model.__init__(self)
        nn.Module.__init__(self)
        
        self.embedding_dim = dataset.premise_embeddings.shape[1]
        self.hidden_dim = hidden_dim
        self.num_relations = len(dataset.edge_types_map)
        
        self.random_premise_embeds = nn.Embedding(dataset.premise_embeddings.shape[0], self.hidden_dim)
        self.random_premise_embed_for_context = nn.Embedding(dataset.premise_embeddings.shape[0], self.embedding_dim)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        self.last_metrics = {}
        self.loss = loss

        self.premise_rgcn = RGCNConv(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            num_relations=2,
        )
        self.context_rgcn = RGCNConv(
            in_channels=self.embedding_dim,
            out_channels=self.hidden_dim,
            num_relations=2,
        )

        self.scoring = HeadAttentionScoring(embedding_dim=self.hidden_dim, num_heads=n_heads, aggregation=aggregation, preprocess=preprocess)
        
        print(f"Initialized RGCNModel with {self.num_relations} relations, hidden_dim={self.hidden_dim}")

    def forward(self, batch_embeddings: torch.Tensor, batch_edge_index: torch.Tensor, batch_edge_attr: torch.Tensor, n_premises: int) -> Tuple[torch.Tensor, torch.Tensor]:
        expected_dtype = torch.float32
        
        initial_premise_embs = self.random_premise_embeds.weight.to(expected_dtype)
        n_premises = initial_premise_embs.size(0)
        
        initial_context_embs = torch.zeros_like(batch_embeddings[n_premises:]).to(expected_dtype)
        initial_premise_emb_for_context = self.random_premise_embed_for_context.weight.to(expected_dtype)
        batch_embeddings_for_context = torch.cat([initial_premise_emb_for_context, initial_context_embs], dim=0)
        
        premise_edge_index_mask = (batch_edge_index[0] < n_premises) & (batch_edge_index[1] < n_premises)
        premise_edge_index = batch_edge_index[:, premise_edge_index_mask]
        premise_edge_attr = batch_edge_attr[premise_edge_index_mask]

        refined_premise_embs = self.premise_rgcn(initial_premise_embs, premise_edge_index, premise_edge_attr)
        refined_context_embs = self.context_rgcn(batch_embeddings_for_context, batch_edge_index, batch_edge_attr)
        
        return refined_premise_embs, refined_context_embs[n_premises:]

    def get_predictions(self, batch_embeddings: torch.Tensor, batch_edge_index: torch.Tensor, batch_edge_attr: torch.Tensor, num_batch_contexts: int, n_premises: int) -> torch.Tensor:
        final_premise_embs, final_context_embs = self.forward(batch_embeddings, batch_edge_index, batch_edge_attr, n_premises)
        return self.scoring.forward(final_premise_embs, final_context_embs)

    def train_batch(self, batch_embeddings: torch.Tensor, batch_edge_index: torch.Tensor, batch_edge_attr: torch.Tensor, batch_labels: torch.Tensor, n_premises: int, step: int) -> torch.Tensor:
        num_batch_contexts = batch_embeddings.shape[0] - n_premises
        logits_tensor = self.get_predictions(batch_embeddings, batch_edge_index, batch_edge_attr, num_batch_contexts, n_premises)
        
        targets_tensor = torch.zeros_like(logits_tensor)
        pos_context_indices = batch_labels[0]
        pos_premise_indices = batch_labels[1]
        targets_tensor[pos_context_indices, pos_premise_indices] = 1.0

        self.last_metrics = compute_retrieval_metrics(logits_tensor.detach(), targets_tensor.detach())
        
        n_negative = (targets_tensor == 0).sum().item()
        n_positive = (targets_tensor == 1).sum().item()

        if n_positive == 0 or n_negative == 0: 
            return torch.tensor(0.0, device=logits_tensor.device, requires_grad=True)
        else:
            pos_weight = n_negative / n_positive
            weights = torch.ones_like(logits_tensor)
            weights[targets_tensor == 1] = pos_weight

        false_positives = (logits_tensor > 0).float() * (targets_tensor == 0).float()
        false_negatives = (logits_tensor < 0).float() * (targets_tensor == 1).float()
        avg_fp = false_positives.sum().item() / n_negative if n_negative > 0 else 0.0
        avg_fn = false_negatives.sum().item() / n_positive if n_positive > 0 else 0.0
        if step % 100 == 0:
            print(f"Step {step}: Avg False Positive Rate: {avg_fp:.4f}, Avg False Negative Rate: {avg_fn:.4f}")

        if self.loss == "bce":
            unweighted_loss = F.binary_cross_entropy_with_logits(logits_tensor, targets_tensor, reduction='none')
        elif self.loss == "mse":
            unweighted_loss = F.mse_loss(torch.sigmoid(logits_tensor), targets_tensor, reduction='none')
        else:
            raise ValueError(f"Unknown loss function: {self.loss}")
        weighted_loss = (unweighted_loss * weights).mean()

        return weighted_loss

    def overfit_on_batch(self, dataset: LightweightGraphDataset, batch_size: int, steps: int = 10000):
        self.train()
        train_indices = dataset.train_mask.nonzero(as_tuple=True)[0]
        train_generator = batchify_contexts(dataset, train_indices, batch_size)
        
        batch_embeddings, batch_edge_index, batch_edge_attr, batch_labels, _ = next(iter(train_generator))

        n_premises = dataset.premise_embeddings.shape[0]
        pbar = tqdm(range(steps), desc="Overfitting on a single batch")
        for j in pbar:
            self.optimizer.zero_grad()
            loss = self.train_batch(batch_embeddings, batch_edge_index, batch_edge_attr, batch_labels, n_premises, j)

            loss.backward()
            self.optimizer.step()
            
            memory = torch.cuda.memory_allocated(DEVICE)/1e9 if torch.cuda.is_available() else 0.0
            log_dict = {"loss": f"{loss.item():.4f}", "memory (GB)": f"{memory:.2f}"}
            log_dict.update(self.last_metrics)
            pbar.set_postfix(log_dict)

    def train_epoch(self, dataset: LightweightGraphDataset, batch_size: int, accumulation_steps: int = 1) -> float:
        self.train()
        train_indices = dataset.train_mask.nonzero(as_tuple=True)[0]
        data_generator = list(batchify_contexts(dataset, train_indices, batch_size))
        
        total_loss, num_batches_processed = 0.0, 0
        self.optimizer.zero_grad()
        
        pbar = tqdm(enumerate(data_generator), total=len(data_generator), desc="Training Epoch")
        for i, (batch_embeddings, batch_edge_index, batch_edge_attr, batch_labels, _) in pbar:
            n_premises = dataset.premise_embeddings.shape[0]
            loss = self.train_batch(batch_embeddings, batch_edge_index, batch_edge_attr, batch_labels, n_premises, i)
            
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(data_generator):
                self.optimizer.step()
                self.optimizer.zero_grad()
                
            total_loss += loss.item() * accumulation_steps
            num_batches_processed += 1
            
            log_dict = {"loss": f"{loss.item() * accumulation_steps:.4f}"}
            log_dict.update(self.last_metrics)
            pbar.set_postfix(log_dict)
            
        return total_loss / num_batches_processed if num_batches_processed > 0 else 0.0
    
HIDDEN_DIM = 2048
LEARNING_RATE = 1e-1
BATCH_SIZE = 1<<9
ACCUMULATION_STEPS = 1
OVERFIT_STEPS = 10000

testmodel = TestModel(
    dataset=dataset, 
    hidden_dim=HIDDEN_DIM, 
    lr=LEARNING_RATE,
    preprocess="none",
    aggregation="logsumexp",
    n_heads=1,
    loss="bce",
)
testmodel.to(DEVICE)

print(f"\n--- Overfitting on a single batch for {OVERFIT_STEPS} steps ---")
testmodel.overfit_on_batch(dataset, batch_size=BATCH_SIZE, steps=OVERFIT_STEPS)
print(f"--- Finished overfitting ---")
