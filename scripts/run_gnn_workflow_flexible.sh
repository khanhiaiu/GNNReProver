#!/bin/bash

# This script automates the full GNN retriever workflow:
# 1. Trains the GNNRetriever model using a specified configuration file on multiple GPUs.
# 2. Runs dynamic, context-aware prediction using the configuration saved during training on a single GPU.
# 3. Evaluates the predictions and prints the final metrics.
#
# It takes five arguments: a unique name for the run, the number of GNN layers,
# the CUDA device indices, the data split to use, and the path to the config file.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- 1. Argument Validation ---
if [ "$#" -ne 5 ]; then
    echo "ERROR: Invalid number of arguments."
    echo "Usage: $0 <unique_name_tag> <num_layers> <CUDA_DEVICES> <split_type> <config_path>"
    echo "  <unique_name_tag>: A descriptive name for your experiment (e.g., 'gated_gnn_test')."
    echo "  <num_layers>: The number of GNN layers for the model (e.g., 2)."
    echo "  <CUDA_DEVICES>: Comma-separated list of GPU indices for training (e.g., '0,1,2')."
    echo "  <split_type>: The dataset split to use. Must be 'random' or 'novel_premises'."
    echo "  <config_path>: The path to the GNN YAML config file (e.g., 'retrieval/confs/cli_gated_gnn.yaml')."
    exit 1
fi

UNIQUE_NAME_TAG=$1
NUM_LAYERS=$2
CUDA_DEVICES=$3
SPLIT_TYPE=$4
CONFIG_PATH=$5

# Validate that num_layers is a positive integer
if ! [[ "$NUM_LAYERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: <num_layers> must be a positive integer. Received: '$NUM_LAYERS'."
    exit 1
fi

# Validate split_type
if [[ "$SPLIT_TYPE" != "random" && "$SPLIT_TYPE" != "novel_premises" ]]; then
    echo "ERROR: Invalid split_type '$SPLIT_TYPE'."
    echo "  <split_type> must be 'random' or 'novel_premises'."
    exit 1
fi

# Validate that the config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: Config file not found at '$CONFIG_PATH'."
    exit 1
fi

echo "================================================================="
echo "Starting Flexible GNN Retriever Workflow"
echo "  - Experiment Name: $UNIQUE_NAME_TAG"
echo "  - GNN Layers: $NUM_LAYERS"
echo "  - GPU Devices for Training: $CUDA_DEVICES"
echo "  - Split Type: $SPLIT_TYPE"
echo "  - Config File: $CONFIG_PATH"
echo "================================================================="


# --- 2. Setup Environment and Paths ---
# Activate your conda environment if it's not already active
# source ~/.bashrc
# conda activate ReProver

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
export PYTHONPATH=$(pwd):$PYTHONPATH
# Suppress Python warnings to reduce log pollution
export PYTHONWARNINGS="ignore"

# Calculate the number of GPUs for the trainer by counting comma-separated values
NUM_GPUS=$(echo "$CUDA_DEVICES" | awk -F, '{print NF}')

# Create a highly descriptive experiment name and log directory.
EXP_NAME="${UNIQUE_NAME_TAG}_${NUM_LAYERS}layers_${SPLIT_TYPE}"
LOG_DIR="lightning_logs/${EXP_NAME}"

# Define paths based on the chosen split
DATA_PATH="data/leandojo_benchmark_4/${SPLIT_TYPE}"
BASE_RETRIEVER="kaiyuy/leandojo-lean4-retriever-byt5-small"

# Define paths for artifacts that will be generated
PREDICTIONS_PATH="${LOG_DIR}/predictions_dynamic.pickle"


# --- 3. Run Training ---
echo
echo "--- STEP 1 of 3: TRAINING GNN RETRIEVER on ${NUM_GPUS} GPU(s) ---"
echo "Logs and checkpoints will be saved to: $LOG_DIR"
echo "Using data from: $DATA_PATH/"
echo "W&B Run Name will be: $EXP_NAME"
echo "Using config from: $CONFIG_PATH"
echo "Overriding GNN layers to: $NUM_LAYERS"
echo "------------------------------------------------"

python retrieval/train_gnn.py fit \
    --config $CONFIG_PATH \
    --model.num_layers $NUM_LAYERS \
    --data.data_path "$DATA_PATH/" \
    --trainer.logger.name "$EXP_NAME" \
    --trainer.logger.save_dir "$LOG_DIR" \
    --trainer.devices $NUM_GPUS

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
SAVED_CONFIG_PATH=$(find "$LOG_DIR" -maxdepth 2 -name "config.yaml" | head -n 1)
if [ -z "$SAVED_CONFIG_PATH" ]; then
    echo "Could not find config.yaml, looking for hparams.yaml as a fallback..."
    SAVED_CONFIG_PATH=$(find "$LOG_DIR" -maxdepth 2 -name "hparams.yaml" | head -n 1)
fi

if [ -z "$SAVED_CONFIG_PATH" ]; then
    echo "ERROR: Could not find config.yaml or hparams.yaml in the log directory: $LOG_DIR."
    exit 1
fi
echo "Found saved config: $SAVED_CONFIG_PATH"

# --- 5. Run Dynamic Prediction (on a single GPU) ---
# For prediction and evaluation, we switch to a single GPU to simplify the process.
FIRST_GPU=$(echo "$CUDA_DEVICES" | cut -d',' -f1)
export CUDA_VISIBLE_DEVICES=$FIRST_GPU
echo
echo "--- STEP 2 of 3: RUNNING DYNAMIC PREDICTION on GPU ${FIRST_GPU} ---"
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

echo "Workflow finished successfully for experiment: $EXP_NAME"