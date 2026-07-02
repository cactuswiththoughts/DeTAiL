#!/bin/bash
#SBATCH -J qwen_text_mlp_leads
#SBATCH -o logs/%j_qwen_text_mlp_leads.out
#SBATCH -e logs/%j_qwen_text_mlp_leads.err
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
raw_wav_dir=/data/sls/scratch/limingw/workplace/speaker_anonymization/data/leads/AllAudiosCleaned4MIT-MGB
wav_dir=/data/sls/scratch/limingw/workplace/speaker_anonymization/data/leads/wavs

stage=30
stop_stage=30
if [ $stage -le 30 ] && [ $stop_stage -ge 30 ]; then
    for i in {0..9}; do
        data_dir=$wav_dir/../cv_folds/fold_0$i
        for x in train test
        do
        python utils/prepare_leads_ms_swift_format_zs.py --tsv $data_dir/$x.tsv --text $data_dir/${x}_whisper.wrd --lbl $data_dir/$x.lbl --output $data_dir/${x}_3class_zs_text_only.jsonl --root_dir $wav_dir --modality T
        done
        python utils/prepare_leads_ms_swift_format_sft.py --fold_dir $data_dir --zs_path {}_3class_zs_text_only.jsonl --out_path {}_3class_sft_text_only.jsonl
    done
fi

if [ $stage -le 40 ] && [ $stop_stage -ge 40 ]; then
    class_set=${CLASS_SET:-"0 1 2"}
    class_tag=$(echo "$class_set" | tr ' ' '_')
    cd llm
    for i in {0..9}; do
        LORA_PATH=
        lora_path=${LORA_PATH:-""}
        lora_args=()
        lora_tag=base
        if [ -n "$lora_path" ]; then
            lora_args=(--lora-path "$lora_path")
            lora_tag=$(printf '%s' "$(basename "$lora_path")" | tr -c '[:alnum:]_-' '_')
        fi

        # Qwen training
        python traintest_qwen_text_only.py \
        --model-name Qwen/Qwen2.5-7B-Instruct \
        --train-jsonl $wav_dir/../cv_folds/fold_0$i/train_3class_sft_text_only.jsonl \
        --test-jsonl $wav_dir/../cv_folds/fold_0$i/test_3class_sft_text_only.jsonl \
        --cache-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen_text_embeddings_${lora_tag} \
        --exp-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen_text_${lora_tag}_class${class_tag} \
        --class_set $class_set \
        "${lora_args[@]}"
    done
    cd ..
fi

if [ $stage -le 50 ] && [ $stop_stage -ge 50 ]; then
    class_set="0 1"
    class_tag=$(echo "$class_set" | tr ' ' '_')
    cd llm
    for i in {0..9}; do
        ckpt_dir=/data/sls/scratch/limingw/workplace/speaker_anonymization/DementiaTTRL/ms-swift/output/Qwen2.5-Omni-7B/leads/fold_0$i/sft_3class
        last_ckpt_dir=$(ls -vd $ckpt_dir/v*-*-* | tail -n 1)
        LORA_PATH=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
        echo fold $i, $class_set, $LORA_PATH

        lora_path=${LORA_PATH:-""}
        lora_args=()
        lora_tag=base
        if [ -n "$lora_path" ]; then
            lora_args=(--lora-path "$lora_path")
            lora_tag=$(printf '%s' "$(basename "$lora_path")" | tr -c '[:alnum:]_-' '_')
        fi

        # Qwen training
        python traintest_qwen.py \
        --train-jsonl $wav_dir/../cv_folds/fold_0$i/train_3class_sft.jsonl \
        --test-jsonl $wav_dir/../cv_folds/fold_0$i/test_3class_sft.jsonl \
        --cache-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen_omni_embeddings_${lora_tag} \
        --exp-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen_omni_${lora_tag}_class${class_tag} \
        --class_set $class_set \
        "${lora_args[@]}"
    done
    cd ..
fi

if [ $stage -le 60 ] && [ $stop_stage -ge 60 ]; then
    class_set="1 2"
    class_tag=$(echo "$class_set" | tr ' ' '_')
    cd llm
    for i in {0..9}; do
        ckpt_dir=/data/sls/scratch/limingw/workplace/speaker_anonymization/DementiaTTRL/ms-swift/output/Qwen2.5-Omni-7B/leads/fold_0$i/sft_3class
        last_ckpt_dir=$(ls -vd $ckpt_dir/v*-*-* | tail -n 1)
        LORA_PATH=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
        echo fold $i, $class_set, $LORA_PATH

        lora_path=${LORA_PATH:-""}
        lora_args=()
        lora_tag=base
        if [ -n "$lora_path" ]; then
            lora_args=(--lora-path "$lora_path")
            lora_tag=$(printf '%s' "$(basename "$lora_path")" | tr -c '[:alnum:]_-' '_')
        fi

        # Qwen training
        python traintest_qwen.py \
        --train-jsonl $wav_dir/../cv_folds/fold_0$i/train_3class_sft.jsonl \
        --test-jsonl $wav_dir/../cv_folds/fold_0$i/test_3class_sft.jsonl \
        --cache-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen_omni_embeddings_${lora_tag} \
        --exp-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen_omni_${lora_tag}_class${class_tag} \
        --class_set $class_set \
        "${lora_args[@]}"
    done
    cd ..
fi

if [ $stage -le 80 ] && [ $stop_stage -ge 80 ]; then
    class_set="0 1 2"
    class_tag=$(echo "$class_set" | tr ' ' '_')
    cd llm
    for i in {0..9}; do
        ckpt_dir=/data/sls/scratch/limingw/workplace/speaker_anonymization/DementiaTTRL/ms-swift/output/Qwen2.5-Omni-7B/leads/fold_0$i/sft_3class
        last_ckpt_dir=$(ls -vd $ckpt_dir/v*-*-* | tail -n 1)
        LORA_PATH=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
        echo fold $i, $class_set, $LORA_PATH

        lora_path=${LORA_PATH:-""}
        lora_args=()
        lora_tag=base
        if [ -n "$lora_path" ]; then
            lora_args=(--lora-path "$lora_path")
            lora_tag=$(printf '%s' "$(basename "$lora_path")" | tr -c '[:alnum:]_-' '_')
        fi

        # Qwen training
        python traintest_qwen.py \
        --train-jsonl $wav_dir/../cv_folds/fold_0$i/train_3class_sft.jsonl \
        --test-jsonl $wav_dir/../cv_folds/fold_0$i/test_3class_sft.jsonl \
        --cache-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen_omni_embeddings_${lora_tag} \
        --exp-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen_omni_${lora_tag}_class${class_tag} \
        --class_set $class_set \
        "${lora_args[@]}"
    done
    cd ..
fi

if [ $stage -le 90 ] && [ $stop_stage -ge 90 ]; then
    class_set="0 1"
    class_tag=$(echo "$class_set" | tr ' ' '_')
    cd llm
    for i in {8..9}; do
        ckpt_dir=/data/sls/scratch/limingw/workplace/speaker_anonymization/DementiaTTRL/ms-swift/output/Qwen2.5-Omni-7B/leads/fold_0$i/grpo_distill_3class_lora_rank_128
        last_ckpt_dir=$(ls -vd $ckpt_dir/v*-*-* | tail -n 1)
        LORA_PATH=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
        echo fold $i, $class_set, $LORA_PATH

        lora_path=${LORA_PATH:-""}
        lora_args=()
        lora_tag=base
        if [ -n "$lora_path" ]; then
            lora_args=(--lora-path "$lora_path")
            lora_tag=$(printf '%s' "$(basename "$lora_path")" | tr -c '[:alnum:]_-' '_')
        fi

        # Qwen training
        python traintest_qwen_with_reasoning.py \
        --train-jsonl $wav_dir/../cv_folds/fold_0$i/train_3class_grpo_distill.jsonl \
        --test-jsonl $wav_dir/../cv_folds/fold_0$i/test_3class_grpo_distill.jsonl \
        --reasoning-dir $LORA_PATH/infer_result \
        --cache-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen_omni_grpo_distill_embeddings_${lora_tag} \
        --exp-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen_omni_grpo_distill_${lora_tag}_class${class_tag} \
        --overwrite-cache \
        --class_set $class_set \
        "${lora_args[@]}"
    done
    cd ..
fi

if [ $stage -le 100 ] && [ $stop_stage -ge 100 ]; then
    class_set="1 2"
    class_tag=$(echo "$class_set" | tr ' ' '_')
    cd llm
    for i in {8..9}; do
        ckpt_dir=/data/sls/scratch/limingw/workplace/speaker_anonymization/DementiaTTRL/ms-swift/output/Qwen2.5-Omni-7B/leads/fold_0$i/grpo_distill_3class_lora_rank_128
        last_ckpt_dir=$(ls -vd $ckpt_dir/v*-*-* | tail -n 1)
        LORA_PATH=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
        echo fold $i, $class_set, $LORA_PATH

        lora_path=${LORA_PATH:-""}
        lora_args=()
        lora_tag=base
        if [ -n "$lora_path" ]; then
            lora_args=(--lora-path "$lora_path")
            lora_tag=$(printf '%s' "$(basename "$lora_path")" | tr -c '[:alnum:]_-' '_')
        fi

        # Qwen training
        python traintest_qwen_with_reasoning.py \
        --train-jsonl $wav_dir/../cv_folds/fold_0$i/train_3class_grpo_distill.jsonl \
        --test-jsonl $wav_dir/../cv_folds/fold_0$i/test_3class_grpo_distill.jsonl \
        --reasoning-dir $LORA_PATH/infer_result \
        --cache-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen_omni_grpo_distill_embeddings_${lora_tag} \
        --exp-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen_omni_grpo_distill_${lora_tag}_class${class_tag} \
        --overwrite-cache \
        --class_set $class_set \
        "${lora_args[@]}"
    done
    cd ..
fi

if [ $stage -le 110 ] && [ $stop_stage -ge 110 ]; then
    class_set="0 1 2"
    class_tag=$(echo "$class_set" | tr ' ' '_')
    cd llm
    for i in {8..9}; do
        ckpt_dir=/data/sls/scratch/limingw/workplace/speaker_anonymization/DementiaTTRL/ms-swift/output/Qwen2.5-Omni-7B/leads/fold_0$i/grpo_distill_3class_lora_rank_128
        last_ckpt_dir=$(ls -vd $ckpt_dir/v*-*-* | tail -n 1)
        LORA_PATH=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
        echo fold $i, $class_set, $LORA_PATH

        lora_path=${LORA_PATH:-""}
        lora_args=()
        lora_tag=base
        if [ -n "$lora_path" ]; then
            lora_args=(--lora-path "$lora_path")
            lora_tag=$(printf '%s' "$(basename "$lora_path")" | tr -c '[:alnum:]_-' '_')
        fi

        # Qwen training
        python traintest_qwen_with_reasoning.py \
        --train-jsonl $wav_dir/../cv_folds/fold_0$i/train_3class_grpo_distill.jsonl \
        --test-jsonl $wav_dir/../cv_folds/fold_0$i/test_3class_grpo_distill.jsonl \
        --reasoning-dir $LORA_PATH/infer_result \
        --cache-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen_omni_grpo_distill_embeddings_${lora_tag} \
        --exp-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen_omni_grpo_distill_${lora_tag}_class${class_tag} \
        --overwrite-cache \
        --class_set $class_set \
        "${lora_args[@]}"
    done
    cd ..
fi

if [ $stage -le 120 ] && [ $stop_stage -ge 120 ]; then
    class_set="0 1"
    class_tag=$(echo "$class_set" | tr ' ' '_')
    cd llm
    for i in {0..9}; do
        echo fold $i, $class_set

        lora_path=${LORA_PATH:-""}
        lora_args=()
        lora_tag=base
        if [ -n "$lora_path" ]; then
            lora_args=(--lora-path "$lora_path")
            lora_tag=$(printf '%s' "$(basename "$lora_path")" | tr -c '[:alnum:]_-' '_')
        fi

        # Qwen training
        if [ ! -d $wav_dir/../gemini_reasoning_traces ]; then
            mkdir -p $wav_dir/../gemini_reasoning_traces
            cp $wav_dir/../cv_folds/fold_00/{train,test}_gemini_gold_rationales.jsonl $wav_dir/../gemini_reasoning_traces
        fi

        python traintest_qwen_with_reasoning.py \
        --train-jsonl $wav_dir/../cv_folds/fold_0$i/train_3class_grpo_distill.jsonl \
        --test-jsonl $wav_dir/../cv_folds/fold_0$i/test_3class_grpo_distill.jsonl \
        --reasoning-dir $wav_dir/../gemini_reasoning_traces \
        --cache-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen_omni_grpo_distill_embeddings_${lora_tag} \
        --exp-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen_omni_grpo_distill_${lora_tag}_class${class_tag} \
        --overwrite-cache \
        --class_set $class_set \
        "${lora_args[@]}"
    done
    cd ..
fi

if [ $stage -le 130 ] && [ $stop_stage -ge 130 ]; then
    class_set="1 2"
    class_tag=$(echo "$class_set" | tr ' ' '_')
    cd llm
    for i in {0..9}; do
        ckpt_dir=/data/sls/scratch/limingw/workplace/speaker_anonymization/DementiaTTRL/ms-swift/output/Qwen2.5-Omni-7B/leads/fold_0$i/grpo_distill_3class_lora_rank_128
        last_ckpt_dir=$(ls -vd $ckpt_dir/v*-*-* | tail -n 1)
        LORA_PATH=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
        echo fold $i, $class_set, $LORA_PATH

        lora_path=${LORA_PATH:-""}
        lora_args=()
        lora_tag=base
        if [ -n "$lora_path" ]; then
            lora_args=(--lora-path "$lora_path")
            lora_tag=$(printf '%s' "$(basename "$lora_path")" | tr -c '[:alnum:]_-' '_')
        fi

        # Qwen training
        python traintest_qwen_with_reasoning.py \
        --train-jsonl $wav_dir/../cv_folds/fold_0$i/train_3class_grpo_distill.jsonl \
        --test-jsonl $wav_dir/../cv_folds/fold_0$i/test_3class_grpo_distill.jsonl \
        --reasoning-dir $wav_dir/../gemini_reasoning_traces \
        --cache-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen_omni_grpo_distill_embeddings_${lora_tag} \
        --exp-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen_omni_grpo_distill_${lora_tag}_class${class_tag} \
        --overwrite-cache \
        --class_set $class_set \
        "${lora_args[@]}"
    done
    cd ..
fi

if [ $stage -le 140 ] && [ $stop_stage -ge 140 ]; then
    class_set="0 1 2"
    class_tag=$(echo "$class_set" | tr ' ' '_')
    cd llm
    for i in {0..9}; do
        ckpt_dir=/data/sls/scratch/limingw/workplace/speaker_anonymization/DementiaTTRL/ms-swift/output/Qwen2.5-Omni-7B/leads/fold_0$i/grpo_distill_3class_lora_rank_128
        last_ckpt_dir=$(ls -vd $ckpt_dir/v*-*-* | tail -n 1)
        LORA_PATH=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
        echo fold $i, $class_set, $LORA_PATH

        lora_path=${LORA_PATH:-""}
        lora_args=()
        lora_tag=base
        if [ -n "$lora_path" ]; then
            lora_args=(--lora-path "$lora_path")
            lora_tag=$(printf '%s' "$(basename "$lora_path")" | tr -c '[:alnum:]_-' '_')
        fi

        # Qwen training
        python traintest_qwen_with_reasoning.py \
        --train-jsonl $wav_dir/../cv_folds/fold_0$i/train_3class_grpo_distill.jsonl \
        --test-jsonl $wav_dir/../cv_folds/fold_0$i/test_3class_grpo_distill.jsonl \
        --reasoning-dir $wav_dir/../gemini_reasoning_traces \
        --cache-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen_omni_grpo_distill_embeddings_${lora_tag} \
        --exp-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen_omni_grpo_distill_${lora_tag}_class${class_tag} \
        --overwrite-cache \
        --class_set $class_set \
        "${lora_args[@]}"
    done
    cd ..
fi

if [ $stage -le 150 ] && [ $stop_stage -ge 150 ]; then
    class_set="0 1"
    class_tag=$(echo "$class_set" | tr ' ' '_')
    cd llm
    for i in {0..9}; do
        echo fold $i, $class_set

        lora_path=${LORA_PATH:-""}
        lora_args=()
        lora_tag=base
        if [ -n "$lora_path" ]; then
            lora_args=(--lora-path "$lora_path")
            lora_tag=$(printf '%s' "$(basename "$lora_path")" | tr -c '[:alnum:]_-' '_')
        fi

        # Qwen training
        if [ ! -d $wav_dir/../gemini_reasoning_traces ]; then
            mkdir -p $wav_dir/../gemini_reasoning_traces
            cp $wav_dir/../cv_folds/fold_00/{train,test}_gemini_gold_rationales.jsonl $wav_dir/../gemini_reasoning_traces
        fi

        python traintest_qwen_with_reasoning.py \
        --train-jsonl $wav_dir/../cv_folds/fold_0$i/train_3class_grpo_distill.jsonl \
        --test-jsonl $wav_dir/../cv_folds/fold_0$i/test_3class_grpo_distill.jsonl \
        --reasoning-dir $wav_dir/../gemini_reasoning_traces \
        --cache-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen_omni_grpo_distill_embeddings_${lora_tag} \
        --exp-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen_omni_grpo_distill_${lora_tag}_class${class_tag} \
        --overwrite-cache \
        --class_set $class_set \
        "${lora_args[@]}"
    done
    cd ..
fi

if [ $stage -le 160 ] && [ $stop_stage -ge 160 ]; then
    class_set="1 2"
    class_tag=$(echo "$class_set" | tr ' ' '_')
    cd llm
    for i in {0..9}; do
        ckpt_dir=/data/sls/scratch/limingw/workplace/speaker_anonymization/DementiaTTRL/ms-swift/output/Qwen2.5-Omni-7B/leads/fold_0$i/grpo_distill_3class_lora_rank_128
        last_ckpt_dir=$(ls -vd $ckpt_dir/v*-*-* | tail -n 1)
        LORA_PATH=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
        echo fold $i, $class_set, $LORA_PATH

        lora_path=${LORA_PATH:-""}
        lora_args=()
        lora_tag=base
        if [ -n "$lora_path" ]; then
            lora_args=(--lora-path "$lora_path")
            lora_tag=$(printf '%s' "$(basename "$lora_path")" | tr -c '[:alnum:]_-' '_')
        fi

        # Qwen training
        python traintest_qwen_with_reasoning.py \
        --train-jsonl $wav_dir/../cv_folds/fold_0$i/train_3class_grpo_distill.jsonl \
        --test-jsonl $wav_dir/../cv_folds/fold_0$i/test_3class_grpo_distill.jsonl \
        --reasoning-dir $wav_dir/../gemini_reasoning_traces \
        --cache-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen_omni_grpo_distill_embeddings_${lora_tag} \
        --exp-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen_omni_grpo_distill_${lora_tag}_class${class_tag} \
        --overwrite-cache \
        --class_set $class_set \
        "${lora_args[@]}"
    done
    cd ..
fi

if [ $stage -le 170 ] && [ $stop_stage -ge 170 ]; then
    class_set="0 1 2"
    class_tag=$(echo "$class_set" | tr ' ' '_')
    cd llm
    for i in {0..9}; do
        ckpt_dir=/data/sls/scratch/limingw/workplace/speaker_anonymization/DementiaTTRL/ms-swift/output/Qwen2.5-Omni-7B/leads/fold_0$i/grpo_distill_3class_lora_rank_128
        last_ckpt_dir=$(ls -vd $ckpt_dir/v*-*-* | tail -n 1)
        LORA_PATH=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
        echo fold $i, $class_set, $LORA_PATH

        lora_path=${LORA_PATH:-""}
        lora_args=()
        lora_tag=base
        if [ -n "$lora_path" ]; then
            lora_args=(--lora-path "$lora_path")
            lora_tag=$(printf '%s' "$(basename "$lora_path")" | tr -c '[:alnum:]_-' '_')
        fi

        # Qwen training
        python traintest_qwen_with_reasoning.py \
        --train-jsonl $wav_dir/../cv_folds/fold_0$i/train_3class_grpo_distill.jsonl \
        --test-jsonl $wav_dir/../cv_folds/fold_0$i/test_3class_grpo_distill.jsonl \
        --reasoning-dir $wav_dir/../gemini_reasoning_traces \
        --cache-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen_omni_grpo_distill_embeddings_${lora_tag} \
        --exp-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen_omni_grpo_distill_${lora_tag}_class${class_tag} \
        --overwrite-cache \
        --class_set $class_set \
        "${lora_args[@]}"
    done
    cd ..
fi

if [ $stage -le 180 ] && [ $stop_stage -ge 180 ]; then
    class_set="0 1"
    class_tag=$(echo "$class_set" | tr ' ' '_')
    cd llm
    for i in {0..9}; do
        ckpt_dir=/data/sls/scratch/limingw/workplace/speaker_anonymization/DementiaTTRL/ms-swift/output/Qwen2.5-Omni-7B/leads/fold_0$i/grpo_distill_3class_lora_rank_128
        last_ckpt_dir=$(ls -vd $ckpt_dir/v*-*-* | tail -n 1)
        LORA_PATH=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
        echo fold $i, $class_set, $LORA_PATH

        lora_path=${LORA_PATH:-""}
        lora_args=()
        lora_tag=base
        if [ -n "$lora_path" ]; then
            lora_args=(--lora-path "$lora_path")
            lora_tag=$(printf '%s' "$(basename "$lora_path")" | tr -c '[:alnum:]_-' '_')
        fi

        # Qwen training
        if [ ! -d $wav_dir/../gemini_reasoning_traces_no_label ]; then
            mkdir -p $wav_dir/../gemini_reasoning_traces_no_label
            cp $wav_dir/../cv_folds/fold_00/{train,test}_gemini_rationales.jsonl $wav_dir/../gemini_reasoning_traces_no_label
        fi

        python traintest_qwen_with_reasoning.py \
        --train-jsonl $wav_dir/../cv_folds/fold_0$i/train_3class_grpo_distill.jsonl \
        --test-jsonl $wav_dir/../cv_folds/fold_0$i/test_3class_grpo_distill.jsonl \
        --reasoning-dir $wav_dir/../gemini_reasoning_traces_no_label \
        --cache-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen_omni_grpo_distill_embeddings_${lora_tag}_no_label \
        --exp-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen_omni_grpo_distill_${lora_tag}_class${class_tag}_no_label \
        --overwrite-cache \
        --class_set $class_set \
        "${lora_args[@]}"
    done
    cd ..
fi

if [ $stage -le 190 ] && [ $stop_stage -ge 190 ]; then
    class_set="1 2"
    class_tag=$(echo "$class_set" | tr ' ' '_')
    cd llm
    for i in {0..9}; do
        ckpt_dir=/data/sls/scratch/limingw/workplace/speaker_anonymization/DementiaTTRL/ms-swift/output/Qwen2.5-Omni-7B/leads/fold_0$i/grpo_distill_3class_lora_rank_128
        last_ckpt_dir=$(ls -vd $ckpt_dir/v*-*-* | tail -n 1)
        LORA_PATH=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
        echo fold $i, $class_set, $LORA_PATH

        lora_path=${LORA_PATH:-""}
        lora_args=()
        lora_tag=base
        if [ -n "$lora_path" ]; then
            lora_args=(--lora-path "$lora_path")
            lora_tag=$(printf '%s' "$(basename "$lora_path")" | tr -c '[:alnum:]_-' '_')
        fi

        # Qwen training
        if [ ! -d $wav_dir/../gemini_reasoning_traces_no_label ]; then
            mkdir -p $wav_dir/../gemini_reasoning_traces_no_label
            cp $wav_dir/../cv_folds/fold_00/{train,test}_gemini_rationales.jsonl $wav_dir/../gemini_reasoning_traces_no_label
        fi

        python traintest_qwen_with_reasoning.py \
        --train-jsonl $wav_dir/../cv_folds/fold_0$i/train_3class_grpo_distill.jsonl \
        --test-jsonl $wav_dir/../cv_folds/fold_0$i/test_3class_grpo_distill.jsonl \
        --reasoning-dir $wav_dir/../gemini_reasoning_traces_no_label \
        --cache-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen_omni_grpo_distill_embeddings_${lora_tag}_no_label \
        --exp-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen_omni_grpo_distill_${lora_tag}_class${class_tag}_no_label \
        --class_set $class_set \
        "${lora_args[@]}"
    done
    cd ..
fi

if [ $stage -le 200 ] && [ $stop_stage -ge 200 ]; then
    class_set="0 1 2"
    class_tag=$(echo "$class_set" | tr ' ' '_')
    cd llm
    for i in {0..9}; do
        ckpt_dir=/data/sls/scratch/limingw/workplace/speaker_anonymization/DementiaTTRL/ms-swift/output/Qwen2.5-Omni-7B/leads/fold_0$i/grpo_distill_3class_lora_rank_128
        last_ckpt_dir=$(ls -vd $ckpt_dir/v*-*-* | tail -n 1)
        LORA_PATH=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
        echo fold $i, $class_set, $LORA_PATH

        lora_path=${LORA_PATH:-""}
        lora_args=()
        lora_tag=base
        if [ -n "$lora_path" ]; then
            lora_args=(--lora-path "$lora_path")
            lora_tag=$(printf '%s' "$(basename "$lora_path")" | tr -c '[:alnum:]_-' '_')
        fi

        # Qwen training
        if [ ! -d $wav_dir/../gemini_reasoning_traces_no_label ]; then
            mkdir -p $wav_dir/../gemini_reasoning_traces_no_label
            cp $wav_dir/../cv_folds/fold_00/{train,test}_gemini_rationales.jsonl $wav_dir/../gemini_reasoning_traces_no_label
        fi

        python traintest_qwen_with_reasoning.py \
        --train-jsonl $wav_dir/../cv_folds/fold_0$i/train_3class_grpo_distill.jsonl \
        --test-jsonl $wav_dir/../cv_folds/fold_0$i/test_3class_grpo_distill.jsonl \
        --reasoning-dir $wav_dir/../gemini_reasoning_traces_no_label \
        --cache-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen_omni_grpo_distill_embeddings_${lora_tag}_no_label \
        --exp-dir $wav_dir/../cv_folds/fold_0$i/mlp_qwen_omni_grpo_distill_${lora_tag}_class${class_tag}_no_label \
        --class_set $class_set \
        "${lora_args[@]}"
    done
    cd ..
fi