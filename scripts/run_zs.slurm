#!/bin/bash
#SBATCH -J zs_leads
#SBATCH -o logs/%j_zs_leads.out
#SBATCH -e logs/%j_zs_leads.err
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
stage=30
stop_stage=32
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    for i in {0..9}; do
        for x in train test; do
            python utils/prepare_leads_ms_swift_format_zs.py \
                --tsv $PWD/../data/leads/cv_folds/fold_0$i/${x}.tsv \
                --text $PWD/../data/leads/cv_folds/fold_0$i/${x}_whisper.wrd \
                --lbl $PWD/../data/leads/cv_folds/fold_0$i/${x}.lbl \
                --output $PWD/../data/leads/cv_folds/fold_0$i/${x}_2class_zs.jsonl \
                --root_dir $PWD/../data/leads/wavs

            python utils/prepare_leads_ms_swift_format_zs_3class.py \
                --tsv $PWD/../data/leads/cv_folds/fold_0$i/${x}.tsv \
                --text $PWD/../data/leads/cv_folds/fold_0$i/${x}_whisper.wrd \
                --lbl $PWD/../data/leads/cv_folds/fold_0$i/${x}.lbl \
                --output $PWD/../data/leads/cv_folds/fold_0$i/${x}_3class_zs.jsonl \
                --root_dir $PWD/../data/leads/wavs
        done
    done
fi

if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]
then
cd ms-swift

for i in {0..9}
do
#for n in 2 3
for n in 2
do
echo fold $i, $n class
swift infer \
  --model Qwen/Qwen2.5-Omni-7B \
  --val_dataset $PWD/../../data/leads/cv_folds/fold_0$i/test_${n}class_zs.jsonl \
  --result_path $PWD/../../data/leads/cv_folds/fold_0$i/test_${n}class_zs_result.jsonl \
  --temperature 0 \
  --max_new_tokens 8 \
  --stream false \
  --remove_unused_columns false \
  --logprobs true \
  --top_logprobs 2
done
done
cd ..
fi

if [ $stage -le 20 ] && [ $stop_stage -ge 20 ]
then
for n in 2 3
do
for i in {0..9}
do
echo fold $i, $n class
python utils/eval_leads_zs.py \
  --response_path $PWD/../data/leads/cv_folds/fold_0$i/test_${n}class_zs_result.jsonl \
  --lbl_path $PWD/../data/leads/cv_folds/fold_0$i/test.lbl \
  --n_class $n
done
done
fi

if [ $stage -le 21 ] && [ $stop_stage -ge 21 ]
then
for n in 2
do
for i in {0..9}
do
python utils/eval_leads_zs_merge_class.py \
  --response_path $PWD/../data/leads/cv_folds/fold_0$i/test_3class_zs_result.jsonl \
  --lbl_path $PWD/../data/leads/cv_folds/fold_0$i/test.lbl \
  --n_class $n
done
done
fi

if [ $stage -le 30 ] && [ $stop_stage -ge 30 ]; then
    for i in {0..9}
    do
    if [ ! -d $PWD/../data/leads/cv_folds/fold_0$i/zs_3class ]; then
        mkdir -p $PWD/../data/leads/cv_folds/fold_0$i/zs_3class
        echo $PWD/../data/leads/cv_folds/fold_0$i/zs_3class
    fi
    for n in 3
    do
    echo fold $i, $n class, LLM probs
    python utils/eval_leads_grpo_llm_prob.py \
    --response_path $PWD/../data/leads/cv_folds/fold_0$i/test_3class_zs_result.jsonl \
    --lbl_path $PWD/../data/leads/cv_folds/fold_0$i/test.lbl \
    --out_dir $PWD/../data/leads/cv_folds/fold_0$i/zs_3class \
    --no_use_generated_think
    done
    done
fi

if [ $stage -le 31 ] && [ $stop_stage -ge 31 ]; then
    for i in {0..9}
    do
    for n in 3
    do
    echo fold $i, $n class, LLM probs
    python utils/eval_leads_grpo_llm_prob_merge_class.py \
    --response_path $PWD/../data/leads/cv_folds/fold_0$i/test_3class_zs_result.jsonl \
    --lbl_path $PWD/../data/leads/cv_folds/fold_0$i/test.lbl \
    --out_dir $PWD/../data/leads/cv_folds/fold_0$i/zs_3class
    done
    done
fi

if [ $stage -le 32 ] && [ $stop_stage -ge 32 ]; then
    for i in {0..9}
    do
    for n in 3
    do
    echo fold $i, $n class, LLM probs
    python utils/eval_leads_grpo_llm_prob_ci_only.py \
    --response_path $PWD/../data/leads/cv_folds/fold_0$i/test_3class_zs_result.jsonl \
    --lbl_path $PWD/../data/leads/cv_folds/fold_0$i/test.lbl \
    --out_dir $PWD/../data/leads/cv_folds/fold_0$i/zs_3class
    done
    done
fi
