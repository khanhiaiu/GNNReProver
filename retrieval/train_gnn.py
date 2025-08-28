import os
from loguru import logger
from pytorch_lightning.cli import LightningCLI

from retrieval.gnn_model import GNNRetriever
from retrieval.gnn_datamodule import GNNDataModule


def main() -> None:
    logger.info(f"PID: {os.getpid()}")
    cli = LightningCLI(GNNRetriever, GNNDataModule)
    logger.info(f"Configuration: \n {cli.config}")


if __name__ == "__main__":
    main()