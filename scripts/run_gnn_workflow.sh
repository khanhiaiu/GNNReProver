#!/bin/bash

# This script automates the full GNN retriever workflow:
# 1. Trains the GNNRetriever model with a specified number of layers.
# 2. Creates a GNN-enhanced index of all premises.
# 3. Runs dynamic, context-aware prediction.
# 4. Evaluates the predictions and prints the final metrics.
#
# It takes four arguments: a unique name for the run, the number of GNN layers,
# the CUDA device index, and the data split to use.

# Exit immediately if a command exits with a non-zero status.
set -e

source ~/.bashrc
conda activate ReProver

# --- 1. Argument Validation ---
if [ "$#" -ne 4 ]; then
    echo "ERROR: Invalid number of arguments."
    echo "Usage: $0 <unique_name_tag> <num_layers> <CUDA_INDEX> <split_type>"
    echo "  <unique_name_tag>: A descriptive name for your experiment (e.g., 'gnn_test')."
    echo "  <num_layers>: The number of GCNConv layers for the GNN model (e.g., 2)."
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

# Create a highly descriptive experiment name and log directory.
EXP_NAME="${UNIQUE_NAME_TAG}_${NUM_LAYERS}layers_${SPLIT_TYPE}"
LOG_DIR="lightning_logs/2025-08-30_${EXP_NAME}" # TODO

# Define paths based on the chosen split
DATA_PATH="data/leandojo_benchmark_4/${SPLIT_TYPE}"
CORPUS_PATH="data/leandojo_benchmark_4/corpus.jsonl"
GNN_CONFIG="retrieval/confs/cli_gnn.yaml"
BASE_RETRIEVER="kaiyuy/leandojo-lean4-retriever-byt5-small"

# Define paths for artifacts that will be generated
GNN_INDEX_PATH="${LOG_DIR}/indexed_corpus_gnn.pickle"
PREDICTIONS_PATH="${LOG_DIR}/predictions_dynamic.pickle"


# --- 3. Run Training ---
echo
echo "--- STEP 1 of 4: TRAINING GNN RETRIEVER ---"
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

# --- DEBUGGING AND DYNAMICALLY FINDING THE CHECKPOINT PATH ---
echo "Training complete. Displaying directory structure for debugging:"
ls -R "$LOG_DIR"
echo "--------------------------------------------------------"
echo "Searching for the generated checkpoint..."

# This robust find command searches all subdirectories for the checkpoint.
GNN_CHECKPOINT_PATH=$(find "$LOG_DIR" -name "last.ckpt")

if [ -z "$GNN_CHECKPOINT_PATH" ]; then
    echo "ERROR: Could not find last.ckpt within $LOG_DIR. Training may have failed or the checkpoint was not saved."
    exit 1
fi
# If find returns multiple matches (it shouldn't), take the first one.
GNN_CHECKPOINT_PATH=$(echo "$GNN_CHECKPOINT_PATH" | head -n 1)

echo "Found checkpoint: $GNN_CHECKPOINT_PATH"


# --- 4. Create GNN-Enhanced Index ---
echo
echo "--- STEP 2 of 4: CREATING GNN-ENHANCED INDEXED CORPUS ---"
echo "Using GNN checkpoint: $GNN_CHECKPOINT_PATH"
echo "Output will be saved to: $GNN_INDEX_PATH"
echo "--------------------------------------------------------"

python retrieval/index.py \
    --ckpt_path $BASE_RETRIEVER \
    --gnn_ckpt_path "$GNN_CHECKPOINT_PATH" \
    --corpus-path "$CORPUS_PATH" \
    --output-path "$GNN_INDEX_PATH"

echo "Indexing complete."


# --- 5. Run Dynamic Prediction ---
echo
echo "--- STEP 3 of 4: RUNNING DYNAMIC PREDICTION ---"
echo "Using data from: $DATA_PATH"
echo "Predictions will be saved to: $PREDICTIONS_PATH"
echo "-----------------------------------------------"

# CORRECTED THE ARGUMENT NAME FROM --retri_ckpt_path to --retriever_ckpt_path
python retrieval/predict_dynamic.py \
    --retriever_ckpt_path $BASE_RETRIEVER \
    --gnn_ckpt_path "$GNN_CHECKPOINT_PATH" \
    --indexed_corpus_path "$GNN_INDEX_PATH" \
    --data_path "$DATA_PATH" \
    --output_path "$PREDICTIONS_PATH"

echo "Prediction complete."


# --- 6. Evaluate Predictions ---
echo
echo "--- STEP 4 of 4: EVALUATING PREDICTIONS ---"
echo "Evaluating predictions file: $PREDICTIONS_PATH"
echo "-------------------------------------------"

python retrieval/evaluate.py \
    --data-path "$DATA_PATH" \
    --preds-file "$PREDICTIONS_PATH"

echo
echo "================================================================="
echo "Workflow Finished Successfully!"
echo "Final results are printed above."
echo "================================================================="