set -x

python3 train_rm_.py \
    --save_path ./checkpoint/llama3-8b-rm \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --train_batch_size 256 \
    --micro_train_batch_size 1 \
    --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
    --bf16 \
    --max_epochs 1 \
    --max_len 8192 \
    --learning_rate 9e-6 \
    --dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \
    --apply_chat_template \
    --chosen_key chosen \
    --rejected_key rejected \
    --load_checkpoint \
    --gradient_checkpointing \
    --max_samples 100
