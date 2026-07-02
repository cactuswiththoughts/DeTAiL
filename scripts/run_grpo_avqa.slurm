#!/bin/bash
#SBATCH -J grpo_avqa
#SBATCH -o logs/%j_grpo_avqa.out
#SBATCH -e logs/%j_grpo_avqa.err
##SBATCH --qos=regular
##SBATCH --gres=gpu:2
##SBATCH --gres=gpu:1
##SBATCH --nodes=1
##SBATCH --partition=a6
##SBATCH --partition=a5
##SBATCH --requeue

export HOME=/data/sls/scratch/limingw
export CONDA_ROOT=/data/sls/scratch/limingw/miniconda3
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate ms_swift
ulimit -s unlimited

## Creating SLURM nodes list
export NODELIST=nodelist.$
srun -l bash -c 'hostname' |  sort -k 2 -u | awk -vORS=, '{print $2":4"}' | sed 's/,$//' > $NODELIST

## Number of total processes
echo " "
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " GPUs per node:= " $SLURM_JOB_GPUS
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NODE

####    Use MPI for communication with Horovod - this can be hard-coded during installation as well.
export HOROVOD_GPU_ALLREDUCE=MPI
export HOROVOD_GPU_ALLGATHER=MPI
export HOROVOD_GPU_BROADCAST=MPI
export NCCL_DEBUG=DEBUG

echo " Running on multiple nodes/GPU devices"
echo ""
echo " Run started at:- "
date

nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0))"
cd ..

stage=21
stop_stage=21
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    for x in train val; do
        python utils/prepare_avqa_ms_swift_format.py \
            --input $PWD/../data/AVQA/${x}_qa.data \
            --output $PWD/../data/AVQA/${x}_grpo.jsonl
    done
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    python utils/prepare_mmau_ms_swift_format.py \
        --input $PWD/OP-TTRAV/verl/data/MMAU/test_mini_test.json \
        --output $PWD/../data/AVQA/test_mini_test_mmau_grpo.jsonl
fi

if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
cd ms-swift
model_name=Qwen2.5-Omni-7B
num_gen=4
n_epoch=1

swift rlhf \
    --rlhf_type grpo \
    --model Qwen/$model_name \
    --reward_funcs external_r1v_acc format \
    --reward_weights 1 0.5 \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset "$PWD/../../data/AVQA/train_grpo.jsonl" \
    --load_from_cache_file true \
    --external_plugins "examples/train/grpo/plugin/plugin.py" \
    --max_completion_length 2048 \
    --num_train_epochs $n_epoch \
    --per_device_eval_batch_size 2 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir output/$model_name/avqa \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations $num_gen \
    --generation_batch_size $num_gen \
    --temperature 1. \
    --top_p 0.99 \
    --top_k 50 \
    --system 'examples/train/grpo/prompt.txt' \
    --log_completions true
cd ..
fi

if [ $stage -le 20 ] && [ $stop_stage -ge 20 ]; then
    cd ms-swift
    
    output_dir=output/Qwen2.5-Omni-7B/avqa
    last_ckpt_dir=$(ls -vd $output_dir/v*-*-* | tail -n 1)
    last_ckpt_dir=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
    echo $last_ckpt_dir

    swift infer \
    --adapters $last_ckpt_dir \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048 \
    --val_dataset "$PWD/../../data/AVQA/test_mini_test_mmau_grpo.jsonl" \
    --metric acc
    cd ..
fi

if [ $stage -le 21 ] && [ $stop_stage -ge 21 ]; then
    output_dir=ms-swift/output/Qwen2.5-Omni-7B/avqa
    last_ckpt_dir=$(ls -vd $output_dir/v*-*-* | tail -n 1)
    last_ckpt_dir=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
    echo $last_ckpt_dir
    response_path=$(ls -vd $last_ckpt_dir/infer_result/*.jsonl | tail -n 1)

    python utils/eval_mmau.py --response_path "$response_path" --solution_path "$PWD/../data/AVQA/test_mini_test_mmau_grpo.jsonl"
fi
