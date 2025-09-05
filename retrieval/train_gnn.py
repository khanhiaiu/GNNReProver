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
        parser.link_arguments("data.graph_dependencies", "model.graph_dependencies")
        # Link the context verbosity level from data to model
        parser.link_arguments("data.context_neighbor_verbosity", "model.context_neighbor_verbosity") # Changed name from context_neighbor_type
        
        # We'll handle edge_types_map separately since it requires corpus instantiation
        
    def before_instantiate_classes(self) -> None:
        logger.info("before_instantiate_classes called")
        logger.info(f"Config structure: {self.config}")
        
        # Create the corpus to get the edge_types_map before instantiating the model
        # Access config through the proper CLI structure
        data_config = self.config.get("fit", {}).get("data", self.config.get("data", {}))
        model_config = self.config.get("fit", {}).get("model", self.config.get("model", {}))
        
        corpus_path = data_config["corpus_path"]
        graph_dependencies = data_config["graph_dependencies"]
        
        logger.info(f"Loading corpus from: {corpus_path}")
        logger.info(f"Graph dependencies config: {graph_dependencies}")
        
        from common import Corpus
        corpus = Corpus(corpus_path, graph_dependencies)
        logger.info(f"Corpus loaded successfully. Edge types map: {corpus.edge_types_map}")
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