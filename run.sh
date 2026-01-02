#!/bin/bash

set -e


cache_dir="/tmp"  # change this to your desired cache directory
export HF_HOME="$cache_dir"
export TORCH_HOME="$cache_dir"
export VLLM_CACHE_ROOT="$cache_dir"
export VLLM_CONFIG_ROOT="$cache_dir"
export FLASHINFER_CACHE_DIR="$cache_dir"
export FLASHINFER_CUBIN_DIR="$cache_dir"
export TRITON_CACHE_DIR="$cache_dir"
export TORCH_EXTENSIONS_DIR="$cache_dir"
export TMPDIR="$cache_dir"
# Set Hugging Face token
export HUGGING_FACE_HUB_TOKEN="<your_token_here>"
# CUDA memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1 


execute_step() {
    local step_name="$1"
    shift
    local full_cmd="$*"

    # Clean the command for printing
    local clean_cmd=$(echo "$full_cmd" | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g')

    echo -e "\n--- Step: $step_name ---"
    echo -e "Command:\n> ${clean_cmd}"

    eval "$full_cmd"
}


# Llama3-8B: 1. fine-tune; 2. test ASR
# Train:
CURRENT_FOLDER=$(pwd)
OUT_DIR="out"

execute_step "Start Llama3-8B Training" \
    llamafactory-cli train \
        --stage dpo \
        --do_train True \
        --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
        --preprocessing_num_workers 16 \
        --finetuning_type lora \
        --template mistral \
        --flash_attn fa2 \
        --dataset_dir $CURRENT_FOLDER/data/lmf_data/ \
        --dataset my-dpo-data-alpaca \
        --cutoff_len 4096 \
        --learning_rate 0.00016 \
        --num_train_epochs 3.0 \
        --max_samples 200 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 16 \
        --lr_scheduler_type cosine \
        --max_grad_norm 1.0 \
        --logging_steps 1 \
        --save_steps 5000 \
        --warmup_steps 0 \
        --packing False \
        --enable_thinking False \
        --report_to none \
        --output_dir saves/Meta-Llama-3-8B-Instruct/lora/$OUT_DIR \
        --bf16 True \
        --plot_loss True \
        --trust_remote_code True \
        --ddp_timeout 180000000 \
        --include_num_input_tokens_seen True \
        --optim adamw_torch \
        --lora_rank 64 \
        --lora_alpha 64 \
        --lora_dropout 0.1 \
        --lora_target all \
        --pref_beta 0.1 \
        --pref_ftx 0 \
        --pref_loss sigmoid


# Test:
execute_step "test model for asr" \
    python test.py \
        -m saves/Meta-Llama-3-8B-Instruct/lora/$OUT_DIR \
        --batch_size 96


echo -e "\n\n=== JOB COMPLETED ==="
echo -e "Evaluation logs and results are saved to saves/Meta-Llama-3-8B-Instruct/lora/$OUT_DIR-log"