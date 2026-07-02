#!/bin/bash
#SBATCH -J qwen_mlp_adress
#SBATCH -o logs/%j_qwen_mlp_adress.out
#SBATCH -e logs/%j_qwen_mlp_adress.err
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
wav_dir=/data/sls/scratch/limingw/workplace/speaker_anonymization/data/adress/wavs
cd ..

stage=10
stop_stage=30
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    for x in train test; do
    python utils/prepare_adress_ms_swift_format_fbank.py \
        --root_dir $wav_dir \
        --tsv $wav_dir/../$x.tsv \
        --text $wav_dir/../${x}.wrd \
        --lbl $wav_dir/../${x}.label \
        --fbank_dir $wav_dir/../mels \
        --output $wav_dir/../${x}_2class_grpo_mel.jsonl
    done

#    python utils/prepare_adress_ms_swift_format_vision_sft.py \
#        --root_dir $wav_dir/../mels \
#        --fold_dir $wav_dir/..
fi

if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
    cd llm
    class_set="0 1"
    class_tag=$(echo "$class_set" | tr ' ' '_')

    echo $class_set
    LORA_PATH=
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
    --train-jsonl $wav_dir/../train_2class_vision_sft.jsonl \
    --test-jsonl $wav_dir/../test_2class_vision_sft.jsonl \
    --cache-dir $wav_dir/../mlp_qwen35_embeddings_${lora_tag} \
    --overwrite-cache \
    --exp-dir $wav_dir/../mlp_qwen35_${lora_tag}_class${class_tag} \
    --class_set $class_set \
    "${lora_args[@]}"

    cd ..
fi

if [ $stage -le 20 ] && [ $stop_stage -ge 20 ]; then
    cd llm
    class_set="0 1"
    class_tag=$(echo "$class_set" | tr ' ' '_')
    
    echo $class_set
    LORA_PATH=   
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
    --train-jsonl $wav_dir/../train_2class_sft.jsonl \
    --test-jsonl $wav_dir/../test_2class_sft.jsonl \
    --cache-dir $wav_dir/../mlp_qwen3_embeddings_${lora_tag} \
    --exp-dir $wav_dir/../mlp_qwen3_${lora_tag}_class${class_tag} \
    --class_set $class_set \
    "${lora_args[@]}"

    cd ..
fi
