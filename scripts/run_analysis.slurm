#!/bin/bash
#SBATCH -J qwen_mlp_leads
#SBATCH -o logs/%j_qwen_mlp_leads.out
#SBATCH -e logs/%j_qwen_mlp_leads.err
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


stage=40
stop_stage=40
if [ $stage -le 31 ] && [ $stop_stage -ge 31 ]; then
    conda activate sylber_matcha
    python utils/combine_layer_results.py \
        --cv_folds_dir $wav_dir/../cv_folds \
        --folds fold_00,fold_01,fold_02,fold_03,fold_04,fold_05,fold_06,fold_07,fold_08,fold_09 \
        --pattern "mlp_qwen_omni_checkpoint-3438_class*_l*/result.csv" \
        --out_csv $wav_dir/../cv_folds/mlp_qwen_omni_embeddings_checkpoint-3438_l_combined.csv \
        --out_plot $wav_dir/../cv_folds/mlp_qwen_omni_embeddings_checkpoint-3438_l_combined.pdf
    conda deactivate
fi

if [ $stage -le 40 ] && [ $stop_stage -ge 40 ]; then
# /data/sls/scratch/limingw/workplace/speaker_anonymization/DeTAiL/ms-swift/output/Qwen2.5-Omni-7B/adress/grpo_distill_2class_lora_rank_128/v1-20260622-172435/checkpoint-746/infer_result/20260623-041023.evidence.jsonl
    output_dir=ms-swift/output/Qwen2.5-Omni-7B/adress/grpo_distill_2class_lora_rank_128
    last_ckpt_dir=$(ls -vd $output_dir/v*-*-* | tail -n 1)
    last_ckpt_dir=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
    infer_result=$(
        find $last_ckpt_dir/infer_result -maxdepth 1 -regextype posix-extended \
            -regex '.*/[0-9]{8}-[0-9]{6}\.evidence\.summary\.json' \
            | sort -V | tail -n 1
    )
    echo $infer_result
    python utils/estimate_rationale_linguistic_evidence.py \
        --plot-summaries $infer_result \
        --plot-labels qwen_grpo_distill \
        --plot-output $output_dir/evidence_reliability.pdf
fi
