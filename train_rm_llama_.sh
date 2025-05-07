#!/usr/bin/env bash
set -x


script_content=$(cat <<'EOF'
python3 train_rm_.py \
    --save_path ${CACHE_DIR}/checkpoint/llama3-1.7b-rm \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --train_batch_size 256 \
    --micro_train_batch_size 1 \
    --pretrain NotTheStallion/Llama-3-8b-sft-mixture-1.71B-layer-reduced \
    --bf16 \
    --max_epochs 10 \
    --max_len 8192 \
    --learning_rate 9e-6 \
    --dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \
    --apply_chat_template \
    --chosen_key chosen \
    --rejected_key rejected \
    --load_checkpoint \
    --gradient_checkpointing \
    --max_samples 100 \
    --use_wandb d140b44c59ccab7e804b577846b34e68bd794886\
    --wandb_project llama3-1.7b-rm \
    --wandb_org the_ranch
EOF
)


# module load compiler/cuda
# module load language/python/3.10
export CACHE_DIR="/beegfs/mkherraz/"
export CUDA_VISIBLE_DEVICES=0
$script_content
