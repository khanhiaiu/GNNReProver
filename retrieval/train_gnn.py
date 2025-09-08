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
        # Link the graph dependency config from data to model
        parser.link_arguments("data.graph_dependencies_config", "model.graph_dependencies_config")
        
        # We'll handle edge_types_map separately since it requires corpus instantiation
        
    def before_instantiate_classes(self) -> None:
        logger.info("before_instantiate_classes called")
        logger.info(f"Config structure: {self.config}")
        
        # Create the corpus to get the edge_types_map before instantiating the model
        # Access config through the proper CLI structure
        data_config = self.config.get("fit", {}).get("data", self.config.get("data", {}))
        model_config = self.config.get("fit", {}).get("model", self.config.get("model", {}))
        
        corpus_path = data_config["corpus_path"]
        graph_dependencies_config = data_config["graph_dependencies_config"]
        
        logger.info(f"Loading corpus from: {corpus_path}")
        logger.info(f"Graph dependencies config: {graph_dependencies_config}")
        
        from common import Corpus
        corpus = Corpus(corpus_path, graph_dependencies_config)
        logger.info(f"Corpus loaded successfully. Edge types map: {corpus.edge_types_map}")
        
        # --- START: ADDED LOGGING ---
        logger.info("--- Initial Corpus and Graph Sample ---")
        num_premises_to_show = 5
        for i, p in enumerate(corpus.all_premises[:num_premises_to_show]):
            logger.info(f"\n--- Premise {i+1}/{num_premises_to_show} ---")
            logger.info(f"Name: {p.full_name}")
            logger.info(f"Code: \n{p.code[:200]}...")  # Truncate for readability

            p_idx = corpus.name2idx.get(p.full_name)
            if p_idx is None:
                continue

            # Outgoing edges
            outgoing_mask = corpus.premise_dep_graph.edge_index[0] == p_idx
            if outgoing_mask.any():
                logger.info("  Outgoing Edges:")
                indices = corpus.premise_dep_graph.edge_index[1][outgoing_mask]
                types = corpus.premise_dep_graph.edge_attr[outgoing_mask]
                for neighbor_idx, type_id in zip(indices, types):
                    name = corpus.all_premises[neighbor_idx.item()].full_name
                    type_name = corpus.edge_types[type_id.item()]
                    logger.info(f"    -> [{type_name}] -> {name}")

            # Ingoing edges
            ingoing_mask = corpus.premise_dep_graph.edge_index[1] == p_idx
            if ingoing_mask.any():
                logger.info("  Ingoing Edges:")
                indices = corpus.premise_dep_graph.edge_index[0][ingoing_mask]
                types = corpus.premise_dep_graph.edge_attr[ingoing_mask]
                for neighbor_idx, type_id in zip(indices, types):
                    name = corpus.all_premises[neighbor_idx.item()].full_name
                    type_name = corpus.edge_types[type_id.item()]
                    logger.info(f"    <- [{type_name}] <- {name}")
        logger.info("--- End of Initial Corpus and Graph Sample ---\n")
        # --- END: ADDED LOGGING ---

        # Add the edge_types_map to model config
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