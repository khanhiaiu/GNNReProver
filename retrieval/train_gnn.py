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
        # Link the dynamically created mapping from the datamodule's corpus to the model
        parser.link_arguments("data.corpus.edge_types_map", "model.edge_type_to_id")
        # Link the graph dependency config from data to model
        parser.link_arguments("data.graph_dependencies", "model.graph_dependencies")
        # Link the context verbosity level from data to model
        parser.link_arguments("data.context_neighbor_verbosity", "model.context_neighbor_verbosity") # Changed name from context_neighbor_type

def main() -> None:
    logger.info(f"PID: {os.getpid()}")
    # Use the new custom CLI class
    cli = GNNCLI(GNNRetriever, GNNDataModule)
    logger.info(f"Configuration: \n {cli.config}")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()