import os
import torch  # <-- Add this import

# Monkeypatch torch.load before Lightning/DeepSpeed calls it
_orig_load = torch.load
def patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)

torch.load = patched_load


from loguru import logger
from pytorch_lightning.cli import LightningCLI

from retrieval.gnn_model import GNNRetriever
from retrieval.gnn_datamodule import GNNDataModule


class GNNCLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.link_arguments("data.graph_dependencies_config", "model.graph_dependencies_config")
        
    def before_instantiate_classes(self) -> None:
        from common import Corpus
        import random

        data_config = self.config.get("fit", {}).get("data", self.config.get("data", {}))
        model_config = self.config.get("fit", {}).get("model", self.config.get("model", {}))
        
        corpus = Corpus(data_config["corpus_path"], data_config["graph_dependencies_config"])
        
        # Collect summary info for logging
        summary_lines = [f"Total premises: {len(corpus.all_premises)}",
                         f"Edge types: {corpus.edge_types}",
                         f"Graph: nodes={corpus.premise_dep_graph.num_nodes}, edges={corpus.premise_dep_graph.edge_index.shape[1]}"]
        
        num_premises_to_show = min(5, len(corpus.all_premises))
        selected_indices = random.sample(range(len(corpus.all_premises)), num_premises_to_show)
        for i, idx in enumerate(selected_indices):
            p = corpus.all_premises[idx]
            edges_info = []
            p_idx = corpus.name2idx.get(p.full_name)
            edge_index, edge_attr = corpus.premise_dep_graph.edge_index, corpus.premise_dep_graph.edge_attr

            # Outgoing edges
            out_mask = edge_index[0] == p_idx
            out_count = 0
            for n_idx, t_id in zip(edge_index[1][out_mask], edge_attr[out_mask]):
                if out_count >= 5:
                    break
                if n_idx < len(corpus.all_premises) and int(t_id) in corpus.edge_types:
                    edges_info.append(f"  -> [{corpus.edge_types[int(t_id)]}] -> {corpus.all_premises[int(n_idx)].full_name}")
                    out_count += 1

            # Ingoing edges
            in_mask = edge_index[1] == p_idx
            in_count = 0
            for n_idx, t_id in zip(edge_index[0][in_mask], edge_attr[in_mask]):
                if in_count >= 5:
                    break
                if n_idx < len(corpus.all_premises) and int(t_id) in corpus.edge_types:
                    edges_info.append(f"  <- [{corpus.edge_types[int(t_id)]}] <- {corpus.all_premises[int(n_idx)].full_name}")
                    in_count += 1

            # Combine premise info in a single log
            premise_summary = f"Premise {i+1}/{num_premises_to_show}: {p.full_name}\nCode: {p.code[:200]}...\n" + \
                              ("\n".join(edges_info) if edges_info else "No edges")
            summary_lines.append(premise_summary)
        
        logger.info("--- Initial Corpus and Graph Sample ---\n" + "\n".join(summary_lines) + "\n--- End of Sample ---")
        
        # Add edge_types_map to model config
        model_config["edge_type_to_id"] = corpus.edge_types_map
        logger.info(f"Set model.edge_type_to_id to: {model_config['edge_type_to_id']}")
    
def main() -> None:
    logger.info(f"PID: {os.getpid()}")
    # Use the new custom CLI class
    cli = GNNCLI(GNNRetriever, GNNDataModule)
    logger.info(f"Configuration: \n {cli.config}")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()