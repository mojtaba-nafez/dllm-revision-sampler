#!/usr/bin/env bash
# ===== Mandatory for proper import and evaluation =====
export PYTHONPATH=.:$PYTHONPATH             
export HF_ALLOW_CODE_EVAL=1                 # Allow code evaluation
export HF_DATASETS_TRUST_REMOTE_CODE=True   # For datasets that use remote code

# ===== Optional but recommended for stability and debugging =====
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1    # Enable async error handling for multi-GPU communication to avoid deadlocks
export NCCL_DEBUG=warn                      # Show NCCL warnings for better diagnosis without flooding logs
export TORCH_DISTRIBUTED_DEBUG=DETAIL       # Provide detailed logging for PyTorch distributed debugging

# ===== Input Arguments =====
model_name_or_path="GSAI-ML/LLaDA-8B-Instruct"
sampling_strategy="dynamic-revise"
instruct=True
num_gpu=1
limit=400
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name_or_path)
      model_name_or_path="$2"; shift 2 ;;
    --sampling_strategy)
      sampling_strategy="$2"; shift 2 ;;
    --instruct)
      instruct="$2"; shift 2 ;;
    --num_gpu)
      num_gpu="$2"; shift 2 ;;
    --limit)
      limit="$2"; shift 2 ;;
    *) 
      echo "Error: Unknown argument: $1"; exit 1 ;;
  esac
done

# ===== Conditional Configurations =====
if [ "$instruct" = "True" ]; then
    echo ">>> Running in INSTRUCT mode"
    common_args="--model llada --apply_chat_template"
else
    echo ">>> Running in BASE mode"
    common_args="--model llada"
fi

echo ">>> Task: bbh.sh"

# =======================
# LLaDA-1.0 Tasks
# =======================

if [ "$instruct" = "True" ]; then
    # Instruct Tasks
    accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
        --tasks bbh --num_fewshot 3 ${common_args} \
        --model_args "pretrained=${model_name_or_path},max_new_tokens=256,steps=256,block_size=256,cfg_scale=0.0,sampling_strategy=${sampling_strategy}" --limit "${limit}" \
        --log_samples --output_path ./logs/${model_name_or_path}_bbh_samples.json


else
    # Base Tasks   
    accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
        --tasks bbh --num_fewshot 3 ${common_args} \
        --model_args "pretrained=${model_name_or_path},max_new_tokens=256,steps=256,block_size=256,cfg_scale=0.0,sampling_strategy=${sampling_strategy}" --limit "${limit}" \
        --log_samples --output_path ./logs/${model_name_or_path}_bbh_samples.json
fi
