#!/usr/bin/env bash
set -x 

# python -m openrlhf.cli.serve_rm \
#     --reward_pretrain OpenRLHF/Llama-3-8b-rm-700k \
#     --port 5000 \
#     --bf16 \
#     --flash_attn \
#     --normalize_reward \
#     --max_len 8192 \
#     --batch_size 16

# module load compiler/cuda
module load compiler/gcc/11.2.0
# module load language/python/3.10

export CUDA_VISIBLE_DEVICES=0
export CACHE_DIR="/beegfs/mkherraz/"

python3 train_ppo_ray.py \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --colocate_actor_ref \
   --pretrain meta-llama/Llama-3.2-1B \
   --reward_pretrain NotTheStallion/llama3-1.7B-RewardModel \
   --save_path /openrlhf/examples/checkpoint/llama3-8b-rlhf \
   --micro_train_batch_size 1 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 3 \
   --max_samples 3 \
   --rollout_batch_size 1024 \
   --max_epochs 10 \
   --prompt_max_len 30000 \
   --generate_max_len 3000 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --normalize_reward \
   --adam_offload \
   --gradient_checkpointing \
   --use_wandb d140b44c59ccab7e804b577846b34e68bd794886 \
   # --load_in_4bit
   # --apply_chat_template \

