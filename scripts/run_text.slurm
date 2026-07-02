#!/bin/bash
#SBATCH -J base_leads_text
#SBATCH -o logs/%j_base_leads_text.out
#SBATCH -e logs/%j_base_leads_text.err
##SBATCH --qos=priority
#SBATCH --qos=regular
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --partition=2080

export HOME=/data/sls/scratch/limingw
export CONDA_ROOT=/data/sls/scratch/limingw/miniconda3
source $CONDA_ROOT/etc/profile.d/conda.sh
#conda activate /data/sls/scratch/sbhati/miniconda3/envs/knnvc 
conda activate knnvc_whisper

raw_wav_dir=/data/sls/scratch/limingw/workplace/speaker_anonymization/data/leads/AllAudiosCleaned4MIT-MGB
wav_dir=/data/sls/scratch/limingw/workplace/speaker_anonymization/data/leads/wavs

stage=20
stop_stage=20
if [ $stage -le 20 ] && [ $stop_stage -ge 20 ]; then
    conda activate sylber_matcha
    for i in {0..9}; do
        echo Fold ${i}
        for x in train test; do
            python faster_whisper_asr.py \
            --tsv_path $wav_dir/../cv_folds/fold_0$i/$x.tsv --out_path $wav_dir/../cv_folds/fold_0$i/${x}_whisper.wrd
        done
    done
    conda deactivate
fi
