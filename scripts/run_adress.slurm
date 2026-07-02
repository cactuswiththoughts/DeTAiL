#!/bin/bash
#SBATCH -J base_adress
#SBATCH -o logs/%j_base_adress.out
#SBATCH -e logs/%j_base_adress.err
##SBATCH --qos=priority
#SBATCH --qos=regular
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --partition=a5
#SBATCH --exclude=sls-a5-3

export HOME=/data/sls/scratch/limingw
export CONDA_ROOT=/data/sls/scratch/limingw/miniconda3
source $CONDA_ROOT/etc/profile.d/conda.sh
#conda activate /data/sls/scratch/sbhati/miniconda3/envs/knnvc 
conda activate knnvc_whisper

cd ..
wav_dir=/data/sls/scratch/limingw/workplace/speaker_anonymization/data/adress/wavs

stage=40
stop_stage=40
if [ $stage -le 20 ] && [ $stop_stage -ge 20 ]; then
    cd plm
    # BERT training
    python traintest_text_adress.py --data-dir $wav_dir/.. --exp-dir $wav_dir/../bert_2class --loss bce --n_class 1 --gt_transcripts --n-epochs 50
    cd ..
fi

if [ $stage -le 30 ] && [ $stop_stage -ge 30 ]; then
    python train_whisper_adress.py --manifest_dir $wav_dir/.. --exp_dir $wav_dir/.. --n_class 1 --n_epochs 50
fi

if [ $stage -le 40 ] && [ $stop_stage -ge 40 ]; then
    python train_whisper_adress.py \
        --manifest_dir $wav_dir/.. \
        --exp_dir $wav_dir/.. \
        --mode test \
        --dataset adress

#    cd plm
#    python traintest_text_adress.py --mode eval --data-dir $wav_dir/.. --exp-dir $wav_dir/../bert_2class --loss bce --n_class 1 --gt_transcripts
#    cd .. 

    python utils/audio_text_fusion.py \
        --audio-score-path $wav_dir/../test_pred_scores.npy \
        --text-score-path $wav_dir/../bert_2class/preds/top120_test_pred_scores.npy \
        --true-label-path $wav_dir/../bert_2class/preds/top120_test_gold_label.npy \
        --out-path $wav_dir/../test_fusion_result.tsv
fi

if [ $stage -le 110 ] && [ $stop_stage -ge 110 ]; then
    cd plm
    python utils/shap_analysis.py --dataset leads
    cd ..
fi

if [ $stage -le 120 ] && [ $stop_stage -ge 120 ]; then
    cd plm
    # BERT training
    python traintest_text_adress.py --data-dir $wav_dir/.. --exp-dir $wav_dir/../cbert_2class --loss bce --n_class 1 --gt_transcripts --n-epochs 50 --lm medicalai/ClinicalBERT
    cd ..
fi


