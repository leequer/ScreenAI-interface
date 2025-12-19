# 1. Export paths
cd src/open-r1-multimodal
export DEBUG_MODE="true"
export PROJ_ROOT="$HOME_ROOT/code_mllm"
RUN_NAME="Qwen2.5-VL-7B-GRPO-websight"
export LOG_PATH="./debug_log_$RUN_NAME.txt"
export PLACEHOLDER_PATH="$PROJ_ROOT/VLM-R1/placeholder.jpg"
export CSS_PATH="$PROJ_ROOT/VLM-R1/tailwind.min.css"
image_folder="$PROJ_ROOT/LLaMA-Factory/data"
data_file_paths="$PROJ_ROOT/LLaMA-Factory/data/CodeMLLM/websight/train_rl.json"

# 2. Experiment parameters
model_name="Qwen/Qwen2.5-VL-7B-Instruct"
gpu_num="8"
bs_per_device=1 
num_generations=8 # assert (bs_per_device x gpu_num) % num_generations == 0
resume="True"





#!/usr/bin/env bash
set -euo pipefail

# 1. Ensure PROJ_ROOT is set
if [[ -z "${PROJ_ROOT:-}" ]]; then
  echo "ERROR: PROJ_ROOT is not defined." >&2
  exit 1
fi



# 3. Declare expected type for each
declare -A expected=(
  [PLACEHOLDER_PATH]=file
  [CSS_PATH]=file
  [image_folder]=dir
  [data_file_paths]=file
)

# 4. Test existence
all_good=true
for var in "${!expected[@]}"; do
  path="${!var}"
  type="${expected[$var]}"
  case "$type" in
    file)
      if [[ ! -f "$path" ]]; then
        echo "✗ File missing: $var → $path" >&2
        all_good=false
      else
        echo "✔ File exists:  $var → $path"
      fi
      ;;
    dir)
      if [[ ! -d "$path" ]]; then
        echo "✗ Directory missing: $var → $path" >&2
        all_good=false
      else
        echo "✔ Directory exists: $var → $path"
      fi
      ;;
    *)
      echo "WARNING: Unknown type for $var: $type" >&2
      ;;
  esac
done

# 5. Exit non-zero if any missing
if ! $all_good; then
  echo "One or more paths were missing." >&2
  exit 2
fi

echo "All paths verified successfully."


torchrun --nproc_per_node=$gpu_num \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    src/open_r1/grpo_jsonl.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir $PROJ_ROOT/VLM-R1/output/$RUN_NAME \
    --model_name_or_path $model_name \
    --dataset_name none \
    --image_folders $image_folder\
    --data_file_paths $data_file_paths \
    --freeze_vision_modules true \
    --max_pixels 1843200 \
    --max_prompt_length 4096 \
    --max_completion_length 2048 \
    --num_generations $num_generations \
    --per_device_train_batch_size $bs_per_device \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 2 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true \
    --resume_from_checkpoint $resume \
