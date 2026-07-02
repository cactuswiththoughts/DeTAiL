#!/bin/bash
#SBATCH -J sft_leads
#SBATCH -o logs/%j_sft_leads.out
#SBATCH -e logs/%j_sft_leads.err
#SBATCH --qos=regular
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
##SBATCH --partition=a6
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

# Line-aligned with prepare_leads_ms_swift_format_zs_3class.py (train.tsv / train.lbl order).
TRAIN_ZS_JSONL="${FOLD_DIR}/train_3class_zs.jsonl"
TRAIN_LBL="${FOLD_DIR}/train.lbl"
TRAIN_SFT_JSONL="${FOLD_DIR}/train_3class_zs_sft.jsonl"

stage=32
stop_stage=32
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]
then
for i in {0..9}
do
FOLD_DIR="/data/sls/scratch/limingw/workplace/speaker_anonymization/data/leads/cv_folds/fold_0$i"
echo $FOLD_DIR
python utils/prepare_leads_ms_swift_format_sft.py --fold_dir $FOLD_DIR
done
fi

if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]
then
# Append assistant targets for ms_swift SFT (zs JSONL is prompt-only).
cd ms-swift

for i in {0..9}
do
FOLD_DIR="/data/sls/scratch/limingw/workplace/speaker_anonymization/data/leads/cv_folds/fold_0$i"
TRAIN_SFT_JSONL=$FOLD_DIR/train_3class_sft.jsonl
swift sft \
  --model Qwen/Qwen2.5-Omni-7B \
  --tuner_type lora \
  --dataset "${TRAIN_SFT_JSONL}" \
  --load_from_cache_file true \
  --split_dataset_ratio 0.01 \
  --torch_dtype bfloat16 \
  --freeze_vit true \
  --freeze_aligner true \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --gradient_accumulation_steps 1 \
  --eval_steps 50 \
  --save_steps 50 \
  --save_total_limit 2 \
  --logging_steps 5 \
  --max_length 2048 \
  --output_dir output/Qwen2.5-Omni-7B/leads/fold_0$i/sft_3class \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 4 \
  --dataset_num_proc 4
done
cd ..
fi

if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]
then
# Append assistant targets for ms_swift SFT (zs JSONL is prompt-only).
cd ms-swift

for i in {0..9}
do
FOLD_DIR="/data/sls/scratch/limingw/workplace/speaker_anonymization/data/leads/cv_folds/fold_0$i"
TRAIN_SFT_JSONL=$FOLD_DIR/train_3class_sft.jsonl
swift sft \
  --model Qwen/Qwen2.5-Omni-7B \
  --tuner_type lora \
  --dataset "${TRAIN_SFT_JSONL}" \
  --load_from_cache_file true \
  --split_dataset_ratio 0.01 \
  --torch_dtype bfloat16 \
  --freeze_vit true \
  --freeze_aligner true \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --gradient_accumulation_steps 1 \
  --eval_steps 50 \
  --save_steps 50 \
  --save_total_limit 2 \
  --logging_steps 5 \
  --max_length 2048 \
  --output_dir output/Qwen2.5-Omni-7B/leads/fold_0$i/sft_3class \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 4 \
  --dataset_num_proc 4
done
cd ..
fi

if [ $stage -le 20 ] && [ $stop_stage -ge 20 ]
then
    cd ms-swift
    for i in {0..9}
    do
    output_dir=output/Qwen2.5-Omni-7B/leads/fold_0$i/sft_3class
    last_ckpt_dir=$(ls -vd $output_dir/v*-*-* | tail -n 1)
    last_ckpt_dir=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
    echo $last_ckpt_dir

    swift infer \
    --adapters $last_ckpt_dir \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048 \
    --val_dataset "$PWD/../../data/leads/cv_folds/fold_0$i/test_3class_sft.jsonl" \
    --result_path $last_ckpt_dir/../pred_3class.jsonl 
    done
    cd ..
fi

if [ $stage -le 30 ] && [ $stop_stage -ge 30 ]; then
    for i in {0..9}
    do
    for n in 3
    do
    echo fold $i, $n class, LLM probs
    output_dir=ms-swift/output/Qwen2.5-Omni-7B/leads/fold_0$i/sft_3class
    last_ckpt_dir=$(ls -vd $output_dir/v*-*-* | tail -n 1)
    last_ckpt_dir=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
    python utils/eval_leads_sft_llm_prob.py \
    --response_path $last_ckpt_dir/../pred_${n}class.jsonl \
    --lbl_path $PWD/../data/leads/cv_folds/fold_0$i/test.lbl \
    --lora_path $last_ckpt_dir \
	--n_class $n \
    --out_dir $last_ckpt_dir/..
    done
    done
fi

if [ $stage -le 31 ] && [ $stop_stage -ge 31 ]; then
    for i in {0..9}
    do
    for n in 3
    do
    echo fold $i, $n class, LLM probs
    output_dir=ms-swift/output/Qwen2.5-Omni-7B/leads/fold_0$i/sft_3class
    last_ckpt_dir=$(ls -vd $output_dir/v*-*-* | tail -n 1)
    last_ckpt_dir=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
    python utils/eval_leads_grpo_llm_prob_merge_class.py \
    --response_path $last_ckpt_dir/../pred_${n}class.jsonl \
    --lbl_path $PWD/../data/leads/cv_folds/fold_0$i/test.lbl \
    --out_dir $last_ckpt_dir/..
    done
    done
fi

if [ $stage -le 32 ] && [ $stop_stage -ge 32 ]; then
    for i in {0..9}
    do
    echo fold $i, 3 class
    output_dir=ms-swift/output/Qwen2.5-Omni-7B/leads/fold_0$i/sft_3class
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
