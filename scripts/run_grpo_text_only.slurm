#!/bin/bash
#SBATCH -J grpo_leads_text_only
#SBATCH -o logs/%j_grpo_leads_text_only.out
#SBATCH -e logs/%j_grpo_leads_text_only.err
#SBATCH --qos=regular
#SBATCH --partition=a5
##SBATCH --gres=gpu:4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
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

stage=32
stop_stage=32
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    for i in {0..9}; do
        for x in train test; do
            python utils/prepare_leads_ms_swift_format_3class_text_only.py \
                --tsv /orcd/pool/006/limingw/workplace/DementiaTTRL/data/leads/cv_folds/fold_0$i/${x}.tsv \
                --text /orcd/pool/006/limingw/workplace/DementiaTTRL/data/leads/cv_folds/fold_0$i/${x}_whisper.wrd \
                --lbl /orcd/pool/006/limingw/workplace/DementiaTTRL/data/leads/cv_folds/fold_0$i/${x}.lbl \
                --output /orcd/pool/006/limingw/workplace/DementiaTTRL/data/leads/cv_folds/fold_0$i/${x}_3class_grpo_text_only.jsonl \
                --root_dir /orcd/pool/006/limingw/workplace/DementiaTTRL/data/leads/wavs
        done
    done
fi

if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]
then
for i in {0..9}
#for i in {6..9}
do
echo fold $i, 3 class
output_dir=output/$model_name/leads/fold_0$i/3class/text_only
#last_ckpt_dir=$(ls -vd $output_dir/v*-*-* | tail -n 1)
#last_ckpt_dir=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
#echo $last_ckpt_dir
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
    --dataset "$PWD/../../data/leads/cv_folds/fold_0$i/train_3class_grpo_text_only.jsonl" \
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
    --output_dir $output_dir \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations $num_gen \
    --generation_batch_size $num_gen \
    --temperature 1. \
    --top_p 0.99 \
    --top_k 50 \
    --system 'examples/train/grpo/prompt.txt' \
    --log_completions true #\
#    --resume_from_checkpoint $last_ckpt_dir
#    --per_device_train_batch_size 2 \
#    --num_generations 8 \
#    --deepspeed zero2 \
done
cd ..
conda deactivate
fi

if [ $stage -le 20 ] && [ $stop_stage -ge 20 ]
then
    cd ms-swift
    for i in {0..2}
    do
    echo fold $i
    output_dir=output/Qwen2.5-7B-Instruct/leads/fold_0$i/3class/text_only
    last_ckpt_dir=$(ls -vd $output_dir/v*-*-* | tail -n 1)
    last_ckpt_dir=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
    echo $last_ckpt_dir

    swift infer \
    --adapters $last_ckpt_dir \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048 \
    --val_dataset "$PWD/../../data/leads/cv_folds/fold_0$i/test_3class_grpo.jsonl"
    done
    cd ..
fi

if [ $stage -le 30 ] && [ $stop_stage -ge 30 ]
then
    for i in {0..2}
    do
    echo fold $i, 3 class
    output_dir=ms-swift/output/Qwen2.5-7B-Instruct/leads/fold_0$i/3class/text_only
    last_ckpt_dir=$(ls -vd $output_dir/v*-*-* | tail -n 1)
    last_ckpt_dir=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
    infer_result=$(ls -vd $last_ckpt_dir/infer_result/*.jsonl | tail -n 1)
    echo $last_ckpt_dir

    python utils/eval_leads_grpo_llm_prob.py \
        --lora_path $last_ckpt_dir \
        --response_path $infer_result \
        --lbl_path $PWD/../data/leads/cv_folds/fold_0$i/test.lbl \
    	--n_class 3 \
        --out_dir $last_ckpt_dir/../
    done
    conda deactivate
fi

if [ $stage -le 31 ] && [ $stop_stage -ge 31 ]
then
    for i in {0..2}
    do
    echo fold $i, 3 class
    output_dir=ms-swift/output/Qwen2.5-7B-Instruct/leads/fold_0$i/3class/text_only
    last_ckpt_dir=$(ls -vd $output_dir/v*-*-* | tail -n 1)
    last_ckpt_dir=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
    infer_result=$(ls -vd $last_ckpt_dir/infer_result/*.jsonl | tail -n 1)
    echo $last_ckpt_dir

    python utils/eval_leads_grpo_llm_prob_merge_class.py \
        --response_path $infer_result \
        --lbl_path $PWD/../data/leads/cv_folds/fold_0$i/test.lbl \
        --out_dir $last_ckpt_dir/../
    done
fi

if [ $stage -le 32 ] && [ $stop_stage -ge 32 ]
then
    for i in {0..2}
    do
    echo fold $i, 3 class
    output_dir=ms-swift/output/Qwen2.5-7B-Instruct/leads/fold_0$i/3class/text_only
    last_ckpt_dir=$(ls -vd $output_dir/v*-*-* | tail -n 1)
    last_ckpt_dir=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
    infer_result=$(ls -vd $last_ckpt_dir/infer_result/*.jsonl | tail -n 1)
    echo $last_ckpt_dir

    python utils/eval_leads_grpo_llm_prob_ci_only.py \
        --response_path $infer_result \
        --lbl_path $PWD/../data/leads/cv_folds/fold_0$i/test.lbl \
        --out_dir $last_ckpt_dir/../
    done
fi
