#!/bin/bash

# This script automates the full GNN retriever workflow:
# 1. Trains the GNNRetriever model with a specified number of layers.
# 2. Runs dynamic, context-aware prediction using the configuration saved during training.
# 3. Evaluates the predictions and prints the final metrics.
#
# It takes four arguments: a unique name for the run, the number of GNN layers,
# the CUDA device index, and the data split to use.

# Exit immediately if a command exits with a non-zero status.
set -e

source ~/.bashrc
conda activate ReProver

python -c "from torch_geometric.loader import NeighborSampler; import torch; edge_index=torch.tensor([[0,1],[1,0]]).t(); NeighborSampler(edge_index, sizes=[1], batch_size=1); print('NeighborSampler works!')"


# --- 1. Argument Validation ---
if [ "$#" -ne 4 ]; then
    echo "ERROR: Invalid number of arguments."
    echo "Usage: $0 <unique_name_tag> <num_layers> <CUDA_INDEX> <split_type>"
    echo "  <unique_name_tag>: A descriptive name for your experiment (e.g., 'gnn_test')."
    echo "  <num_layers>: The number of GNN layers for the GNN model (e.g., 2)."
    echo "  <CUDA_INDEX>: The index of the GPU to use (e.g., 0)."
    echo "  <split_type>: The dataset split to use. Must be 'random' or 'novel_premises'."
    exit 1
fi

UNIQUE_NAME_TAG=$1
NUM_LAYERS=$2
CUDA_INDEX=$3
SPLIT_TYPE=$4

# Validate that num_layers is a positive integer
if ! [[ "$NUM_LAYERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: <num_layers> must be a positive integer. Received: '$NUM_LAYERS'."
    exit 1
fi

if [[ "$SPLIT_TYPE" != "random" && "$SPLIT_TYPE" != "novel_premises" ]]; then
    echo "ERROR: Invalid split_type '$SPLIT_TYPE'."
    echo "  <split_type> must be 'random' or 'novel_premises'."
    exit 1
fi

echo "================================================================="
echo "Starting GNN Retriever Workflow"
echo "  - Experiment Name: $UNIQUE_NAME_TAG"
echo "  - GNN Layers: $NUM_LAYERS"
echo "  - GPU Index: $CUDA_INDEX"
echo "  - Split Type: $SPLIT_TYPE"
echo "================================================================="


# --- 2. Setup Environment and Paths ---
export CUDA_VISIBLE_DEVICES=$CUDA_INDEX
export PYTHONPATH=$(pwd):$PYTHONPATH
# Suppress Python warnings to reduce log pollution
export PYTHONWARNINGS="ignore"

# Create a highly descriptive experiment name and log directory.
EXP_NAME="${UNIQUE_NAME_TAG}_${NUM_LAYERS}layers_${SPLIT_TYPE}"
LOG_DIR="lightning_logs/$(date +'%Y-%m-%d')_${EXP_NAME}"

# Define paths based on the chosen split
DATA_PATH="data/leandojo_benchmark_4/${SPLIT_TYPE}"
CORPUS_PATH="data/leandojo_benchmark_4/corpus.jsonl"
GNN_CONFIG="retrieval/confs/cli_gnn.yaml"
BASE_RETRIEVER="kaiyuy/leandojo-lean4-retriever-byt5-small"

# Define paths for artifacts that will be generated
PREDICTIONS_PATH="${LOG_DIR}/predictions_dynamic.pickle"


# --- 3. Run Training ---
echo
echo "--- STEP 1 of 3: TRAINING GNN RETRIEVER ---"
echo "Logs and checkpoints will be saved to: $LOG_DIR"
echo "Using data from: $DATA_PATH/"
echo "W&B Run Name will be: $EXP_NAME"
echo "Overriding GNN layers to: $NUM_LAYERS"
echo "------------------------------------------------"

python retrieval/train_gnn.py fit \
    --config $GNN_CONFIG \
    --model.num_layers $NUM_LAYERS \
    --data.data_path "$DATA_PATH/" \
    --trainer.logger.name "$EXP_NAME" \
    --trainer.logger.save_dir "$LOG_DIR" 

# --- 4. Find Checkpoint and Saved Config ---
echo "Training complete. Searching for artifacts..."

# Find the last saved checkpoint
GNN_CHECKPOINT_PATH=$(find "$LOG_DIR" -name "last.ckpt" | head -n 1)
if [ -z "$GNN_CHECKPOINT_PATH" ]; then
    echo "ERROR: Could not find last.ckpt within $LOG_DIR. Training may have failed."
    exit 1
fi
echo "Found checkpoint: $GNN_CHECKPOINT_PATH"

# Find the saved configuration file.
# Modern LightningCLI saves `config.yaml`. We check for that first.
# As a fallback for older versions, we also check for `hparams.yaml`.
SAVED_CONFIG_PATH=$(find "$LOG_DIR" -maxdepth 1 -name "config.yaml" | head -n 1)
if [ -z "$SAVED_CONFIG_PATH" ]; then
    echo "Could not find config.yaml in the root of the log directory, looking for hparams.yaml as a fallback..."
    SAVED_CONFIG_PATH=$(find "$LOG_DIR" -maxdepth 1 -name "hparams.yaml" | head -n 1)
fi

if [ -z "$SAVED_CONFIG_PATH" ]; then
    echo "ERROR: Could not find config.yaml or hparams.yaml in the log directory: $LOG_DIR."
    exit 1
fi
echo "Found saved config: $SAVED_CONFIG_PATH"


# --- 5. Run Dynamic Prediction ---
echo
echo "--- STEP 2 of 3: RUNNING DYNAMIC PREDICTION ---"
echo "Using saved training config from: $SAVED_CONFIG_PATH"
echo "Using data from: $DATA_PATH"
echo "Predictions will be saved to: $PREDICTIONS_PATH"
echo "-----------------------------------------------"

python retrieval/predict_dynamic.py \
    --config "$SAVED_CONFIG_PATH" \
    --retriever_ckpt_path "$BASE_RETRIEVER" \
    --gnn_ckpt_path "$GNN_CHECKPOINT_PATH" \
    --data_path "$DATA_PATH" \
    --output_path "$PREDICTIONS_PATH"

echo "Prediction complete."


# --- 6. Evaluate Predictions ---
echo
echo "--- STEP 3 of 3: EVALUATING PREDICTIONS ---"
echo "Evaluating predictions file: $PREDICTIONS_PATH"
echo "-------------------------------------------"

python retrieval/evaluate.py \
    --data-path "$DATA_PATH" \
    --preds-file "$PREDICTIONS_PATH"
