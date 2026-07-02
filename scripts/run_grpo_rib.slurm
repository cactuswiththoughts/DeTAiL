#!/bin/bash
#SBATCH -J rib_leads
#SBATCH -o logs/%j_rib_leads.out
#SBATCH -e logs/%j_rib_leads.err
#SBATCH --qos=regular
##SBATCH --gres=gpu:4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --partition=a5
#SBATCH --requeue

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

stage=42
stop_stage=42
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    for i in {0..9}; do
        for x in train test; do
            python utils/prepare_leads_ms_swift_format_3class.py \
                --tsv $PWD/../data/leads/cv_folds/fold_0$i/${x}.tsv \
                --text $PWD/../data/leads/cv_folds/fold_0$i/${x}_whisper.wrd \
                --lbl $PWD/../data/leads/cv_folds/fold_0$i/${x}.lbl \
                --output $PWD/../data/leads/cv_folds/fold_0$i/${x}_3class_grpo.jsonl \
                --root_dir $PWD/../data/leads/wavs
        done
    done
fi

if [ $stage -le 30 ] && [ $stop_stage -ge 30 ]
then
n_gen=4
n_epoch=1
model_name=Qwen/Qwen2.5-Omni-7B

cd ms-swift
#for i in {0..9}
for i in {5..9}
do
swift rlhf \
    --rlhf_type grpo \
    --model $model_name \
    --external_plugins examples/train/rib/rib_adress_plugin.py \
    --reward_funcs rib_3class_acc rib_3class_format rib_prior \
    --reward_weights 1 0.5 0.5 \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset "$PWD/../data/leads/cv_folds/fold_0$i/train_3class_grpo.jsonl" \
    --load_from_cache_file true \
    --max_completion_length 2048 \
    --num_train_epochs $n_epoch \
    --per_device_eval_batch_size 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir output/$model_name/leads_rib/3class/fold_0$i/ \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations $n_gen \
    --generation_batch_size 8 \
    --temperature 1. \
    --top_p 0.99 \
    --top_k 50 \
    --system 'examples/train/grpo/prompt.txt' \
    --log_completions true
done
cd ..
fi

if [ $stage -le 42 ] && [ $stop_stage -ge 42 ]; then
    for i in {0..9}
    do
    echo fold $i, 3 class
    output_dir=ms-swift/output/Qwen/Qwen2.5-Omni-7B/leads_rib/3class/fold_0$i
    last_ckpt_dir=$(ls -vd $output_dir/v*-*-* | tail -n 1)
    last_ckpt_dir=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
    echo $last_ckpt_dir

    python utils/eval_leads_grpo_llm_prob_ci_only.py \
        --response_path $last_ckpt_dir/../pred_3class.jsonl \
        --lbl_path $PWD/../data/leads/cv_folds/fold_0$i/test.lbl \
        --out_dir $last_ckpt_dir/../
    done
    conda deactivate
fi
