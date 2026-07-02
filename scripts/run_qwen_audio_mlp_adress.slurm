#!/bin/bash
#SBATCH -J qwen_audio_mlp_adress
#SBATCH -o logs/%j_qwen_audio_mlp_adress.out
#SBATCH -e logs/%j_qwen_audio_mlp_adress.err
##SBATCH --qos=priority
#SBATCH --qos=regular
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --partition=a5
#SBATCH --exclude=sls-a5-3

export HOME=/data/sls/scratch/limingw
export CONDA_ROOT=/data/sls/scratch/limingw/miniconda3
source $CONDA_ROOT/etc/profile.d/conda.sh 
conda activate ms_swift

cd ..
data_dir=/data/sls/scratch/limingw/workplace/speaker_anonymization/data/adress
wav_dir=$data_dir/wavs

stage=30
stop_stage=40
if [ $stage -le 30 ] && [ $stop_stage -ge 30 ]; then
    for x in train test
    do
    python utils/prepare_adress_ms_swift_format_zs.py --tsv $data_dir/$x.tsv --text $data_dir/${x}_raw.wrd --lbl $data_dir/$x.label --output $data_dir/${x}_2class_zs_audio_only.jsonl --root_dir $wav_dir --modality A
    done
    python utils/prepare_adress_ms_swift_format_sft.py --fold_dir $data_dir --zs_path {}_2class_zs_audio_only.jsonl --out_path {}_2class_sft_audio_only.jsonl
fi

if [ $stage -le 40 ] && [ $stop_stage -ge 40 ]; then
    class_set=${CLASS_SET:-"0 1"}
    class_tag=$(echo "$class_set" | tr ' ' '_')
    cd llm
    LORA_PATH=
    echo $LORA_PATH

    lora_path=${LORA_PATH:-""}
    lora_args=()
    lora_tag=base
    if [ -n "$lora_path" ]; then
        lora_args=(--lora-path "$lora_path")
        lora_tag=$(printf '%s' "$(basename "$lora_path")" | tr -c '[:alnum:]_-' '_')
    fi

    # Qwen training
    python traintest_qwen.py \
    --model-name Qwen/Qwen2.5-Omni-7B \
    --train-jsonl $wav_dir/../train_2class_sft_audio_only.jsonl \
    --test-jsonl $wav_dir/../test_2class_sft_audio_only.jsonl \
    --cache-dir $wav_dir/../mlp_qwen_audio_embeddings_${lora_tag} \
    --exp-dir $wav_dir/../mlp_qwen_audio_${lora_tag}_class${class_tag} \
    --class_set $class_set \
    "${lora_args[@]}"
    cd ..
fi

if [ $stage -le 50 ] && [ $stop_stage -ge 50 ]; then
    class_set="0 1"
    class_tag=$(echo "$class_set" | tr ' ' '_')
    cd llm
    echo with gold reasoning traces, $class_set

    LORA_PATH=
    lora_path=${LORA_PATH:-""}
    lora_args=()
    lora_tag=base
    if [ -n "$lora_path" ]; then
        lora_args=(--lora-path "$lora_path")
        lora_tag=$(printf '%s' "$(basename "$lora_path")" | tr -c '[:alnum:]_-' '_')
    fi

    # Qwen training
    if [ ! -d $data_dir/gemini_gold_reasoning_traces ]; then
        mkdir -p $data_dir/gemini_gold_reasoning_traces
        cp $data_dir/{train,test}_gemini_gold_rationales.jsonl $data_dir/gemini_gold_reasoning_traces
    fi

    python traintest_qwen_with_reasoning.py \
    --train-jsonl $wav_dir/../train_2class_grpo_distill.jsonl \
    --test-jsonl $wav_dir/../test_2class_grpo_distill.jsonl \
    --reasoning-dir $wav_dir/../gemini_gold_reasoning_traces \
    --cache-dir $wav_dir/../mlp_qwen_omni_grpo_distill_embeddings_${lora_tag} \
    --exp-dir $wav_dir/../mlp_qwen_omni_grpo_distill_${lora_tag}_class${class_tag} \
    --overwrite-cache \
    --class_set $class_set \
    "${lora_args[@]}"
    cd ..
fi

if [ $stage -le 60 ] && [ $stop_stage -ge 60 ]; then
    class_set="0 1"
    class_tag=$(echo "$class_set" | tr ' ' '_')
    cd llm
    ckpt_dir=/data/sls/scratch/limingw/workplace/speaker_anonymization/DementiaTTRL/ms-swift/output/Qwen2.5-Omni-7B/adress/grpo_distill_2class_lora_rank_128
    last_ckpt_dir=$(ls -vd $ckpt_dir/v*-*-* | tail -n 1)
    LORA_PATH=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
    echo $LORA_PATH
    echo with reasoning traces, $class_set

    lora_path=${LORA_PATH:-""}
    lora_args=()
    lora_tag=base
    if [ -n "$lora_path" ]; then
        lora_args=(--lora-path "$lora_path")
        lora_tag=$(printf '%s' "$(basename "$lora_path")" | tr -c '[:alnum:]_-' '_')
    fi

    # Qwen training
    if [ ! -d $data_dir/gemini_reasoning_traces_no_label ]; then
        mkdir -p $data_dir/gemini_reasoning_traces_no_label
        cp $data_dir/{train,test}_gemini_rationales.jsonl $data_dir/gemini_reasoning_traces_no_label
    fi

    python traintest_qwen_with_reasoning.py \
    --train-jsonl $data_dir/train_2class_grpo_distill.jsonl \
    --test-jsonl $data_dir/test_2class_grpo_distill.jsonl \
    --reasoning-dir $data_dir/gemini_reasoning_traces_no_label \
    --cache-dir $data_dir/mlp_qwen_omni_grpo_distill_embeddings_${lora_tag}_no_label \
    --exp-dir $data_dir/mlp_qwen_omni_grpo_distill_${lora_tag}_class${class_tag}_no_label \
    --overwrite-cache \
    --class_set $class_set \
    "${lora_args[@]}"
    cd ..
fi
