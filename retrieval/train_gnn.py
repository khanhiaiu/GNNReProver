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
        parser.link_arguments("data.corpus.edge_type_to_id", "model.edge_type_to_id")

def main() -> None:
    logger.info(f"PID: {os.getpid()}")
    # Use the new custom CLI class
    cli = GNNCLI(GNNRetriever, GNNDataModule)
    logger.info(f"Configuration: \n {cli.config}")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()