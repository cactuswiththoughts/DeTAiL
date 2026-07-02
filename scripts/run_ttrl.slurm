#!/bin/bash
#SBATCH -J ttrl
#SBATCH -o logs/%j_ttrl.out
#SBATCH -e logs/%j_ttrl.err
#SBATCH --qos=regular
##SBATCH --gres=gpu:4
#SBATCH --gres=gpu:1
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

stage=24
stop_stage=24
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    for x in train test; do
        python utils/prepare_ms_swift_format.py \
            --tsv $PWD/../data/adress/${x}.tsv \
            --text $PWD/../data/adress/${x}_raw.wrd \
            --lbl $PWD/../data/adress/${x}.label \
            --output $PWD/../data/adress/${x}_2class_grpo.jsonl \
            --root_dir $PWD/../data/adress/wavs
    done
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    for x in train val; do
#        python utils/extract_3class_labels.py \
#            --tsv_path $PWD/../data/fhs_gold92/${x}.tsv \
#            --all_lbl_path $PWD/../data/fhs_gold92/all.label \
#            --all_tsv_path $PWD/../data/fhs_gold92/all.tsv \
#            --output_path $PWD/../data/fhs_gold92/${x}_3class.label

        python utils/prepare_fhs_gold92_ms_swift_format.py \
            --tsv $PWD/../data/fhs_gold92/$x.tsv \
            --text $PWD/../data/fhs_gold92/${x}.wrd \
            --lbl $PWD/../data/fhs_gold92/${x}_3class.label \
            --output $PWD/../data/fhs_gold92/${x}_3class_grpo.jsonl \
            --root_dir $PWD/../data/fhs_gold92/wavs
    done
fi

if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]
then
cd ms-swift
model_name=Qwen2.5-Omni-7B
output_dir=output/$model_name/adress/2class
num_gen=4
n_epoch=2

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
    --dataset "$PWD/../../data/adress/train_2class_grpo.jsonl" \
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
    --log_completions true
cd ..
fi

if [ $stage -le 11 ] && [ $stop_stage -ge 11 ]
then
cd ms-swift
model_name=Qwen2.5-Omni-7B
output_dir=output/$model_name/fhs_gold92/3class
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
    --dataset "$PWD/../../data/fhs_gold92/train_3class_grpo.jsonl" \
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
    --log_completions true
cd ..
fi

if [ $stage -le 12 ] && [ $stop_stage -ge 12 ]
then
    echo "stage 20: Running zero-shot inference on LEADS dataset..."
    cd ms-swift
    for i in {0..9}
    do
    output_dir=output/Qwen2.5-Omni-7B/adress/2class
    last_ckpt_dir=$(ls -vd $output_dir/v*-*-* | tail -n 1)
    last_ckpt_dir=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
    echo $last_ckpt_dir

    swift infer \
    --adapters $last_ckpt_dir \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048 \
    --val_dataset "$PWD/../../data/leads/cv_folds/fold_0$i/test_2class_grpo.jsonl" \
    --result_path $output_dir/leads/2class/fold_0$i/pred.jsonl 
    done
    cd ..
fi

if [ $stage -le 13 ] && [ $stop_stage -ge 13 ]
then
    echo "stage 13: Running zero-shot inference (3-class) on LEADS dataset..."
    cd ms-swift
    for i in {0..9}
    do
    output_dir=output/Qwen2.5-Omni-7B/fhs_gold92/3class
    last_ckpt_dir=$(ls -vd $output_dir/v*-*-* | tail -n 1)
    last_ckpt_dir=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
    echo $last_ckpt_dir

    swift infer \
    --adapters $last_ckpt_dir \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048 \
    --val_dataset "$PWD/../../data/leads/cv_folds/fold_0$i/test_3class_grpo.jsonl" \
    --result_path $output_dir/leads/3class/fold_0$i/pred.jsonl 
    done
    cd ..
fi

if [ $stage -le 14 ] && [ $stop_stage -ge 14 ]; then
    for i in {0..9}
    do
    echo fold $i, 3 class
    output_dir=ms-swift/output/Qwen2.5-Omni-7B/fhs_gold92/3class/
    last_ckpt_dir=$(ls -vd $output_dir/v*-*-* | tail -n 1)
    last_ckpt_dir=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
    echo $last_ckpt_dir

    python utils/eval_leads_grpo_llm_prob.py \
        --lora_path $last_ckpt_dir \
        --response_path $last_ckpt_dir/../../leads/3class/fold_0$i/pred.jsonl \
        --lbl_path $PWD/../data/leads/cv_folds/fold_0$i/test.lbl \
    	--n_class 3 \
        --out_dir $last_ckpt_dir/../
    done
    conda deactivate
fi

echo "Evaluate zero-shot performance"
if [ $stage -le 22 ] && [ $stop_stage -ge 22 ]; then
    conda activate knnvc_whisper
    
    output_dir=ms-swift/output/Qwen2.5-Omni-7B/adress/2class
    last_ckpt_dir=$(ls -vd $output_dir/v*-*-* | tail -n 1)
    last_ckpt_dir=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
    echo $last_ckpt_dir

    for i in {0..9}
    do
    python utils/eval_leads_zs_grpo.py \
        --response_path $output_dir/leads/2class/fold_0$i/pred.jsonl \
        --lbl_path $PWD/../data/leads/cv_folds/fold_0$i/test.lbl \
        --n_class 2 \
        --out_dir $last_ckpt_dir
    done
    conda deactivate
fi

if [ $stage -le 23 ] && [ $stop_stage -ge 23 ]; then
    conda activate knnvc_whisper
    
    output_dir=ms-swift/output/Qwen2.5-Omni-7B/fhs_gold92/3class
    last_ckpt_dir=$(ls -vd $output_dir/v*-*-* | tail -n 1)
    last_ckpt_dir=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
    echo $last_ckpt_dir

    for i in {0..9}
    do
    python utils/eval_leads_zs_grpo.py \
        --response_path $output_dir/leads/3class/fold_0$i/pred.jsonl \
        --lbl_path $PWD/../data/leads/cv_folds/fold_0$i/test.lbl \
        --n_class 3 \
        --out_dir $last_ckpt_dir
    done
    conda deactivate
fi

if [ $stage -le 24 ] && [ $stop_stage -ge 24 ]; then
    for i in {0..9}
    do
    echo fold $i, 2 class
    output_dir=ms-swift/output/Qwen2.5-Omni-7B/adress/2class
    last_ckpt_dir=$(ls -vd $output_dir/v*-*-* | tail -n 1)
    last_ckpt_dir=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
    python utils/eval_leads_grpo_llm_prob.py \
        --lora_path $last_ckpt_dir \
        --response_path $last_ckpt_dir/../../leads/2class/fold_0$i/pred.jsonl \
        --lbl_path $PWD/../data/leads/cv_folds/fold_0$i/test.lbl \
    	--n_class 2 \
        --out_dir $last_ckpt_dir
    done
    conda deactivate
fi

echo "2-class TTRL training"
if [ $stage -le 30 ] && [ $stop_stage -ge 30 ]; then
cd ms-swift
model_name=Qwen2.5-Omni-7B
output_dir=output/$model_name/adress/2class
last_ckpt_dir=$(ls -vd $output_dir/v*-*-* | tail -n 1)
last_ckpt_dir=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
echo $last_ckpt_dir

num_gen=4
n_epoch=2
#for i in {0..9}
for i in {3..9}
do
echo fold $i
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/$model_name \
    --adapters $last_ckpt_dir \
    --reward_funcs ttrl_classification \
    --reward_weights 1 \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset "$PWD/../../data/leads/cv_folds/fold_0$i/train_2class_grpo.jsonl" \
    --load_from_cache_file true \
    --external_plugins "examples/train/grpo/plugin/plugin_ttrl.py" \
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
    --output_dir $output_dir/../leads_ttrl/cv_folds/fold_0$i/2class \
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
done
cd ..
fi

echo "2-class evaluation on ADReSS"
if [ $stage -le 21 ] && [ $stop_stage -ge 31 ]; then
    cd ms-swift
    swift infer \
    --adapters $last_ckpt_dir \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048 \
    --val_dataset "$PWD/../../data/adress/test_2class_grpo.jsonl" \
    --result_path $last_ckpt_dir/../pred.jsonl 
    cd ..
fi

echo "2-class TTRL evaluation"
if [ $stage -le 31 ] && [ $stop_stage -ge 31 ]; then
    cd ms-swift
    for i in {0..9}
    do
    output_dir=output/Qwen2.5-Omni-7B/adress/leads_ttrl/cv_folds/fold_0$i/2class
    last_ckpt_dir=$(ls -vd $output_dir/v*-*-* | tail -n 1)
    last_ckpt_dir=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
    echo $last_ckpt_dir

    swift infer \
    --adapters $last_ckpt_dir \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048 \
    --val_dataset "$PWD/../../data/leads/cv_folds/fold_0$i/test_2class_grpo.jsonl" \
    --result_path $last_ckpt_dir/../pred.jsonl
    done
    cd ..
fi

if [ $stage -le 32 ] && [ $stop_stage -ge 32 ]; then
    conda activate knnvc_whisper 
    for i in {0..9}
    do
    echo fold $i, 2 class
    output_dir=ms-swift/output/Qwen2.5-Omni-7B/adress/leads_ttrl/cv_folds/fold_0$i/2class
    last_ckpt_dir=$(ls -vd $output_dir/v*-*-* | tail -n 1)
    python utils/eval_leads_zs_grpo.py \
        --response_path $last_ckpt_dir/pred.jsonl \
        --lbl_path $PWD/../data/leads/cv_folds/fold_0$i/test.lbl \
    	--n_class 2 \
        --out_dir $last_ckpt_dir
    done
    conda deactivate
fi

if [ $stage -le 33 ] && [ $stop_stage -ge 33 ]; then
    for i in {0..9}
    do
    echo fold $i, 2 class
    output_dir=ms-swift/output/Qwen2.5-Omni-7B/adress/leads_ttrl/cv_folds/fold_0$i/2class
    last_ckpt_dir=$(ls -vd $output_dir/v*-*-* | tail -n 1)
    python utils/eval_leads_zs_grpo.py \
        --response_path $last_ckpt_dir/pred.jsonl \
        --lbl_path $PWD/../data/leads/cv_folds/fold_0$i/test.lbl \
    	--n_class 2 \
        --out_dir $last_ckpt_dir
    done
    conda deactivate
fi

if [ $stage -le 34 ] && [ $stop_stage -ge 34 ]; then
    for i in {0..9}
    do
    echo fold $i, 2 class
    output_dir=ms-swift/output/Qwen2.5-Omni-7B/adress/leads_ttrl/cv_folds/fold_0$i/2class
    last_ckpt_dir=$(ls -vd $output_dir/v*-*-* | tail -n 1)
    last_ckpt_dir=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
    python utils/eval_leads_grpo_llm_prob.py \
        --lora_path $last_ckpt_dir \
        --response_path $last_ckpt_dir/../pred.jsonl \
        --lbl_path $PWD/../data/leads/cv_folds/fold_0$i/test.lbl \
    	--n_class 2 \
        --out_dir $last_ckpt_dir
    done
    conda deactivate
fi
