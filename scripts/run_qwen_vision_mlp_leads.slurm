#!/bin/bash
#SBATCH -J qwen_mlp_leads
#SBATCH -o logs/%j_qwen_mlp_leads.out
#SBATCH -e logs/%j_qwen_mlp_leads.err
##SBATCH --qos=priority
#SBATCH --qos=regular
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --partition=a6
#SBATCH --exclude=sls-a5-3

export HOME=/data/sls/scratch/limingw
export CONDA_ROOT=/data/sls/scratch/limingw/miniconda3
source $CONDA_ROOT/etc/profile.d/conda.sh 
conda activate ms_swift
wav_dir=/data/sls/scratch/limingw/workplace/speaker_anonymization/data/leads/wavs
cd ..

stage=80
stop_stage=90
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    for i in {0..9}; do
        python utils/prepare_leads_ms_swift_format_vision_sft.py \
            --root_dir $wav_dir/../cv_folds/fold_0$i/mels \
            --fold_dir $wav_dir/../cv_folds/fold_0$i
    done
fi

if [ $stage -le 40 ] && [ $stop_stage -ge 40 ]; then
    cd llm
    class_set="0 1"
    class_tag=$(echo "$class_set" | tr ' ' '_')
    for i in {0..9}; do
        echo Fold-$i, $class_set
        lora_path=${LORA_PATH:-""}
        lora_args=()
        lora_tag=base
        if [ -n "$lora_path" ]; then
            lora_args=(--lora-path "$lora_path")
            lora_tag=$(printf '%s' "$(basename "$lora_path")" | tr -c '[:alnum:]_-' '_')
        fi

        # Qwen training
        python traintest_qwen35.py \
        --model-name Qwen/Qwen3.5-9B \
        --modality image \
        --train-jsonl $wav_dir/../cv_folds/fold_0$i/train_3class_vision_sft.jsonl \
        --test-jsonl $wav_dir/../cv_folds/fold_0$i/test_3class_vision_sft.jsonl \
        --cache-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen35_embeddings_${lora_tag} \
        --overwrite-cache \
        --exp-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen35_${lora_tag}_class${class_tag} \
        --class_set $class_set \
        "${lora_args[@]}"
    done
    cd ..
fi

if [ $stage -le 50 ] && [ $stop_stage -ge 50 ]; then
    cd llm
    class_set="1 2"
    class_tag=$(echo "$class_set" | tr ' ' '_')
    for i in {0..9}; do
        echo Fold-$i, $class_set
        lora_path=${LORA_PATH:-""}
        lora_args=()
        lora_tag=base
        if [ -n "$lora_path" ]; then
            lora_args=(--lora-path "$lora_path")
            lora_tag=$(printf '%s' "$(basename "$lora_path")" | tr -c '[:alnum:]_-' '_')
        fi

        # Qwen training
        python traintest_qwen35.py \
        --model-name Qwen/Qwen3.5-9B \
        --train-jsonl $wav_dir/../cv_folds/fold_0$i/train_3class_vision_sft.jsonl \
        --test-jsonl $wav_dir/../cv_folds/fold_0$i/test_3class_vision_sft.jsonl \
        --cache-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen35_embeddings_${lora_tag} \
        --exp-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen35_${lora_tag}_class${class_tag} \
        --class_set $class_set \
        "${lora_args[@]}"
    done
    cd ..
fi

if [ $stage -le 60 ] && [ $stop_stage -ge 60 ]; then
    cd llm
    class_set="0 1 2"
    class_tag=$(echo "$class_set" | tr ' ' '_')
    for i in {0..9}; do
        echo Fold-$i, $class_set
        lora_path=${LORA_PATH:-""}
        lora_args=()
        lora_tag=base
        if [ -n "$lora_path" ]; then
            lora_args=(--lora-path "$lora_path")
            lora_tag=$(printf '%s' "$(basename "$lora_path")" | tr -c '[:alnum:]_-' '_')
        fi

        # Qwen training
        python traintest_qwen35.py \
        --model-name Qwen/Qwen3.5-9B \
        --train-jsonl $wav_dir/../cv_folds/fold_0$i/train_3class_vision_sft.jsonl \
        --test-jsonl $wav_dir/../cv_folds/fold_0$i/test_3class_vision_sft.jsonl \
        --cache-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen35_embeddings_${lora_tag} \
        --exp-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen35_${lora_tag}_class${class_tag} \
        --class_set $class_set \
        "${lora_args[@]}"
    done
    cd ..
fi

if [ $stage -le 70 ] && [ $stop_stage -ge 70 ]; then
    cd llm
    class_set="0 1"
    class_tag=$(echo "$class_set" | tr ' ' '_')
    for i in {0..9}; do
        echo Fold-$i, $class_set
        lora_path=${LORA_PATH:-""}
        lora_args=()
        lora_tag=base
        if [ -n "$lora_path" ]; then
            lora_args=(--lora-path "$lora_path")
            lora_tag=$(printf '%s' "$(basename "$lora_path")" | tr -c '[:alnum:]_-' '_')
        fi

        # Qwen training
        python traintest_qwen35.py \
        --model-name Qwen/Qwen3-Omni-30B-A3B-Thinking \
        --modality audio \
        --train-jsonl $wav_dir/../cv_folds/fold_0$i/train_3class_sft.jsonl \
        --test-jsonl $wav_dir/../cv_folds/fold_0$i/test_3class_sft.jsonl \
        --cache-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen3_embeddings_${lora_tag} \
        --exp-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen3_${lora_tag}_class${class_tag} \
        --class_set $class_set \
        "${lora_args[@]}"
    done
    cd ..
fi

if [ $stage -le 80 ] && [ $stop_stage -ge 80 ]; then
    cd llm
    class_set="1 2"
    class_tag=$(echo "$class_set" | tr ' ' '_')
    for i in {0..9}; do
        echo Fold-$i, $class_set
        lora_path=${LORA_PATH:-""}
        lora_args=()
        lora_tag=base
        if [ -n "$lora_path" ]; then
            lora_args=(--lora-path "$lora_path")
            lora_tag=$(printf '%s' "$(basename "$lora_path")" | tr -c '[:alnum:]_-' '_')
        fi

        # Qwen training
        python traintest_qwen35.py \
        --model-name Qwen/Qwen3-Omni-30B-A3B-Thinking \
        --modality audio \
        --train-jsonl $wav_dir/../cv_folds/fold_0$i/train_3class_sft.jsonl \
        --test-jsonl $wav_dir/../cv_folds/fold_0$i/test_3class_sft.jsonl \
        --cache-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen3_embeddings_${lora_tag} \
        --exp-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen3_${lora_tag}_class${class_tag} \
        --class_set $class_set \
        "${lora_args[@]}"
    done
    cd ..
fi

if [ $stage -le 90 ] && [ $stop_stage -ge 90 ]; then
    cd llm
    class_set="0 1 2"
    class_tag=$(echo "$class_set" | tr ' ' '_')
    for i in {0..9}; do
        echo Fold-$i, $class_set
        lora_path=${LORA_PATH:-""}
        lora_args=()
        lora_tag=base
        if [ -n "$lora_path" ]; then
            lora_args=(--lora-path "$lora_path")
            lora_tag=$(printf '%s' "$(basename "$lora_path")" | tr -c '[:alnum:]_-' '_')
        fi

        # Qwen training
        python traintest_qwen35.py \
        --model-name Qwen/Qwen3-Omni-30B-A3B-Thinking \
        --modality audio \
        --train-jsonl $wav_dir/../cv_folds/fold_0$i/train_3class_sft.jsonl \
        --test-jsonl $wav_dir/../cv_folds/fold_0$i/test_3class_sft.jsonl \
        --cache-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen3_embeddings_${lora_tag} \
        --exp-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen3_${lora_tag}_class${class_tag} \
        --class_set $class_set \
        "${lora_args[@]}"
    done
    cd ..
fi
