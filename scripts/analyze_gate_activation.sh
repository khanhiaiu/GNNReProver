source ~/.bashrc

conda activate ReProver

# Set your PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH

# (Set variables as before)
RUN_NAME="gated_gnn_2gpu_random_2layers_random"
LOG_DIR="lightning_logs/$RUN_NAME"
SPLIT_TYPE="random"
BASE_RETRIEVER="kaiyuy/leandojo-lean4-retriever-byt5-small"
GNN_CHECKPOINT_PATH=$(find "$LOG_DIR" -name "last.ckpt" | head -n 1)
SAVED_CONFIG_PATH=$(find "$LOG_DIR" -name "config.yaml" -o -name "hparams.yaml" | head -n 1)
# set cuda available devices to 1, 2
export CUDA_VISIBLE_DEVICES=1,2

python retrieval/analyze_gate_activations.py \
    --config "$SAVED_CONFIG_PATH" \
    --retriever_ckpt_path "$BASE_RETRIEVER" \
    --gnn_ckpt_path "$GNN_CHECKPOINT_PATH" \
    --data_path "data/leandojo_benchmark_4/${SPLIT_TYPE}" \
    --splits val test