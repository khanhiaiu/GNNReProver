import os
import torch  # <-- Add this import
from loguru import logger
from pytorch_lightning.cli import LightningCLI

from retrieval.gnn_model import GNNRetriever
from retrieval.gnn_datamodule import GNNDataModule


def main() -> None:
    logger.info(f"PID: {os.getpid()}")
    cli = LightningCLI(GNNRetriever, GNNDataModule)
    logger.info(f"Configuration: \n {cli.config}")


if __name__ == "__main__":
    # Set the start method to 'spawn' BEFORE any other torch/CUDA calls.
    # This is the crucial fix for the "Cannot re-initialize CUDA" error.
    # It must be in the `if __name__ == "__main__":` block.
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()