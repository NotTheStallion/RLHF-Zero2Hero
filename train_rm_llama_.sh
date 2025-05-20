#!/usr/bin/env bash
set -x



# module load compiler/cuda
# module load language/python/3.10
export CACHE_DIR="/beegfs/mkherraz/"
export CUDA_VISIBLE_DEVICES=0


python3 train_rm_.py \
    --save_path ${CACHE_DIR}/checkpoint/llama3-3.2-1b-rm \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --train_batch_size 256 \
    --micro_train_batch_size 1 \
    --pretrain meta-llama/Llama-3.2-1B \
    --bf16 \
    --max_epochs 10 \
    --max_len 8192 \
    --learning_rate 9e-6 \
    --dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \
    --chosen_key chosen \
    --rejected_key rejected \
    --load_checkpoint \
    --gradient_checkpointing \
    --use_wandb d140b44c59ccab7e804b577846b34e68bd794886\
    --wandb_project llama3-3.2-1b-rm \
    --wandb_org the_ranch \
    --max_samples 2000 \
    # --apply_chat_template \
