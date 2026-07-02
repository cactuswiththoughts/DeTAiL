#!/bin/bash
#SBATCH -J qwen_text_mlp_adress
#SBATCH -o logs/%j_qwen_text_mlp_adress.out
#SBATCH -e logs/%j_qwen_text_mlp_adress.err
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

stage=40
stop_stage=40
if [ $stage -le 30 ] && [ $stop_stage -ge 30 ]; then
    for x in train test
    do
    python utils/prepare_adress_ms_swift_format_zs.py --tsv $data_dir/$x.tsv --text $data_dir/${x}_raw.wrd --lbl $data_dir/$x.label --output $data_dir/${x}_2class_zs_text_only.jsonl --root_dir $wav_dir --modality T
    done
    python utils/prepare_adress_ms_swift_format_sft.py --fold_dir $data_dir --zs_path {}_2class_zs_text_only.jsonl --out_path {}_2class_sft_text_only.jsonl
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
    python traintest_qwen_text_only.py \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --train-jsonl $wav_dir/../train_2class_sft_text_only.jsonl \
    --test-jsonl $wav_dir/../test_2class_sft_text_only.jsonl \
    --cache-dir $wav_dir/../mlp_qwen_text_embeddings_${lora_tag} \
    --exp-dir $wav_dir/../mlp_qwen_text_${lora_tag}_class${class_tag} \
    --class_set $class_set \
    "${lora_args[@]}"
    cd ..
fi