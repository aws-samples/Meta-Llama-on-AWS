#!/bin/bash
set -euo pipefail

# Configuration
readonly SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
readonly MODEL_ID="meta-llama/Llama-4-Scout-17B-16E"
readonly CONFIG="config.yaml"
readonly MAX_SEQ_LEN=2048
readonly BATCH_SIZE=2
readonly DOWNLOAD_DIR="$SCRIPT_DIR/Llama-4-Scout-17B-16E"
readonly ADAPTER_DIR="$SCRIPT_DIR/l4_lora_output"
readonly FINAL_OUTPUT_DIR="$SCRIPT_DIR/final_merged_weights"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2
}

validate_requirements() {
    [[ -f "$CONFIG" ]] || { log "ERROR: Config file $CONFIG not found"; exit 1; }
    [[ -d "$DOWNLOAD_DIR" ]] || { log "ERROR: Download directory $DOWNLOAD_DIR not found"; exit 1; }
    command -v python3 >/dev/null || { log "ERROR: python3 not found"; exit 1; }
    command -v tune >/dev/null || { log "ERROR: tune command not found"; exit 1; }
}

setup_directories() {
    log "Creating output directories..."
    mkdir -p "$ADAPTER_DIR" "$FINAL_OUTPUT_DIR"
}

run_tuning() {
    log "Starting LoRA fine-tuning..."
    cd "$SCRIPT_DIR"
    PYTHONPATH="$PWD:${PYTHONPATH:-}" tune run --nproc_per_node 8 lora_finetune_distributed \
        --config "$CONFIG" \
        batch_size="$BATCH_SIZE" \
        tokenizer.max_seq_len="$MAX_SEQ_LEN" \
        fsdp_cpu_offload=True \
        output_dir="$ADAPTER_DIR"
}

merge_weights() {
    log "Merging weights..."
    python3 merge_weights.py "$ADAPTER_DIR" "$DOWNLOAD_DIR" "$FINAL_OUTPUT_DIR"
}

copy_tokenizer() {
    log "Copying tokenizer files..."
    cp "$DOWNLOAD_DIR"/tokenizer* "$FINAL_OUTPUT_DIR/"
}

main() {
    log "Starting Llama-4 fine-tuning pipeline..."
    validate_requirements
    setup_directories
    run_tuning
    merge_weights
    copy_tokenizer
    log "Pipeline completed successfully!"
}

main "$@"
