# retrieval/main.py

import os
import pickle # <-- Add pickle
from loguru import logger
from pytorch_lightning.cli import LightningCLI

from retrieval.model import PremiseRetriever
from retrieval.datamodule import RetrievalDataModule


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.link_arguments("model.model_name", "data.model_name")
        parser.link_arguments("data.max_seq_len", "model.max_seq_len")


def main() -> None:
    logger.info(f"PID: {os.getpid()}")
    cli = CLI(PremiseRetriever, RetrievalDataModule, run=False) # <-- Add run=False
    
    if cli.subcommand == "predict":
        logger.info("Starting prediction...")
        # This captures the results correctly from all workers
        list_of_batch_results = cli.trainer.predict(cli.model, datamodule=cli.datamodule)
        all_predictions = [item for batch in list_of_batch_results for item in batch]

        # Save the collected predictions
        if cli.trainer.log_dir:
            output_path = os.path.join(cli.trainer.log_dir, "predictions.pickle")
            with open(output_path, "wb") as f:
                pickle.dump(all_predictions, f)
            logger.info(f"Predictions saved to {output_path}")
        else:
            logger.warning("No log_dir found in trainer. Predictions are not saved.")
    else:
        # For 'fit', 'validate', etc., run as normal
        cli.trainer.fit(cli.model, datamodule=cli.datamodule)

    logger.info("Configuration: \n", cli.config)


if __name__ == "__main__":
    main()