#!/usr/bin/env bash
set -x

script_content=$(cat <<'EOF'
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
EOF
)

if [[ $1 == "sbatch" ]]; then
    # Job name
    #SBATCH -J TEST_Slurm
    # Asking for one node
    #SBATCH -N 1
    #SBATCH -n 4
    # Standard output
    #SBATCH -o slurm.sh%j.out
    # Standard error
    #SBATCH -e slurm.sh%j.err

    echo "=====my job information ===="
    echo "Node List: " $SLURM_NODELIST
    echo "my jobID: " $SLURM_JOB_ID
    echo "Partition: " $SLURM_JOB_PARTITION
    echo "submit directory:" $SLURM_SUBMIT_DIR
    echo "submit host:" $SLURM_SUBMIT_HOST
    echo "In the directory:" $PWD
    echo "As the user:" $USER

    # module purge
    # module load compiler/gcc
    module load compiler/cuda
    module load language/python/3.10
    pip install -r requirements.txt
    bash -c "srun -n4 -C sirocco&v100 ${script_content}"
else
    bash -c "${script_content}"
fi
