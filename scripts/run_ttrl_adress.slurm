#!/bin/bash
#SBATCH -J adress_ttrl
#SBATCH -o logs/%j_adress_ttrl.out
#SBATCH -e logs/%j_adress_ttrl.err
#SBATCH --qos=regular
##SBATCH --gres=gpu:4
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

stage=40
stop_stage=41
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

echo "Zero-shot Inference on ADReSS"
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    cd ms-swift

    swift infer \
    --model Qwen/Qwen2.5-Omni-7B \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048 \
    --val_dataset "$PWD/../../data/adress/test_2class_grpo.jsonl" \
    --result_path output/Qwen2.5-Omni-7B/adress/2class_zs/pred.jsonl 
    cd ..
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
 #   python utils/eval_leads_zs.py \
 #       --response_path ms-swift/output/Qwen2.5-Omni-7B/adress/2class_zs/pred.jsonl \
 #       --lbl_path $PWD/../data/adress/test.label \
 #    	 --n_class 2 \
 #       --out_dir output/Qwen2.5-Omni-7B/adress/2class_zs

    python utils/eval_leads_grpo_llm_prob.py \
        --response_path ms-swift/output/Qwen2.5-Omni-7B/adress/2class_zs/pred.jsonl \
        --lbl_path $PWD/../data/adress/test.label \
    	--n_class 2 \
        --out_dir ms-swift/output/Qwen2.5-Omni-7B/adress/2class_zs
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

echo "Inference on ADReSS"
if [ $stage -le 11 ] && [ $stop_stage -ge 11 ]; then
    cd ms-swift
    output_dir=output/Qwen2.5-Omni-7B/adress/2class
    last_ckpt_dir=$(ls -vd $output_dir/v*-*-* | tail -n 1)
    last_ckpt_dir=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
    echo $last_ckpt_dir

    swift infer \
    --adapters $last_ckpt_dir \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048 \
    --val_dataset "$PWD/../../data/adress/test_2class_grpo.jsonl" \
    --result_path $last_ckpt_dir/../pred.jsonl 
    cd ..
fi

if [ $stage -le 12 ] && [ $stop_stage -ge 12 ]; then
    output_dir=ms-swift/output/Qwen2.5-Omni-7B/adress/2class
    last_ckpt_dir=$(ls -vd $output_dir/v*-*-* | tail -n 1)
    last_ckpt_dir=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
    echo $last_ckpt_dir

    python utils/eval_leads_grpo_llm_prob.py \
        --lora_path $last_ckpt_dir \
        --response_path $last_ckpt_dir/../pred.jsonl \
        --lbl_path $PWD/../data/adress/test.label \
    	--n_class 2 \
        --out_dir $last_ckpt_dir
fi

echo "Zero-shot inference on LEADS"
if [ $stage -le 21 ] && [ $stop_stage -ge 21 ]
then
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

echo "Evaluate zero-shot performance"
if [ $stage -le 21 ] && [ $stop_stage -ge 21 ]; then
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
fi

echo "2-class GRPO zero-shot transfer evaluation"
if [ $stage -le 40 ] && [ $stop_stage -ge 40 ]; then
    cd ms-swift
    output_dir=output/Qwen2.5-Omni-7B/adress/2class   
    for i in {0..9}
    do
    last_ckpt_dir=$(ls -vd $output_dir/v*-*-* | tail -n 1)
    last_ckpt_dir=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
    echo $last_ckpt_dir

    swift infer \
    --adapters $last_ckpt_dir \
    --stream true \
    --temperature 0 \
    --max_new_tokens 4096 \
    --val_dataset "$PWD/../../data/leads/cv_folds/fold_0$i/test_2class_grpo.jsonl" \
    --result_path $last_ckpt_dir/../leads/fold_0$i/pred.jsonl
    done
    cd ..
fi

if [ $stage -le 41 ] && [ $stop_stage -ge 41 ]; then
    output_dir=ms-swift/output/Qwen2.5-Omni-7B/adress/2class
    for i in {0..9}
    do
    echo fold $i, 2 class
    last_ckpt_dir=$(ls -vd $output_dir/v*-*-* | tail -n 1)
    last_ckpt_dir=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
    python utils/eval_leads_grpo_llm_prob.py \
        --lora_path $last_ckpt_dir \
        --response_path $last_ckpt_dir/../leads/fold_0$i/pred.jsonl \
        --lbl_path $PWD/../data/leads/cv_folds/fold_0$i/test.lbl \
    	--n_class 2 \
        --out_dir $last_ckpt_dir
    done
fi
