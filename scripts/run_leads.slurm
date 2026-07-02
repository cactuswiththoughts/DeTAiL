#!/bin/bash
#SBATCH -J base_leads
#SBATCH -o logs/%j_base_leads.out
#SBATCH -e logs/%j_base_leads.err
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
raw_wav_dir=/data/sls/scratch/limingw/workplace/speaker_anonymization/data/leads/AllAudiosCleaned4MIT-MGB
wav_dir=/data/sls/scratch/limingw/workplace/speaker_anonymization/data/leads/wavs

stage=93
stop_stage=93
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    python vad_segment_audio.py --input-dir $raw_wav_dir --output-dir $wav_dir --glob *.mp3 
fi

if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
    python prepare_leads.py --audio-root $wav_dir --xlsx /data/sls/scratch/limingw/workplace/speaker_anonymization/data/leads/LEADS_MIT_MGB.xlsx --out-dir $wav_dir/.. 
fi

if [ $stage -le 20 ] && [ $stop_stage -ge 20 ]; then
    for i in {0..9}; do
        python train_whisper.py \
        --manifest_dir $wav_dir/../cv_folds/fold_0$i --exp_dir $wav_dir/../cv_folds/fold_0$i
    done
fi

if [ $stage -le 21 ] && [ $stop_stage -ge 21 ]; then
    for i in {0..9}; do
        python train_whisper.py \
        --manifest_dir $wav_dir/../cv_folds/fold_0$i --exp_dir $wav_dir/../cv_folds/fold_0$i/whisper_class12 --class_set 1 2
    done
fi

if [ $stage -le 30 ] && [ $stop_stage -ge 30 ]; then
    for i in {0..9}; do
        python train_whisper_3class.py \
        --manifest_dir $wav_dir/../cv_folds/fold_0$i --exp_dir $wav_dir/../cv_folds/fold_0$i/whisper_3class
    done
fi

if [ $stage -le 40 ] && [ $stop_stage -ge 40 ]; then
    cd plm
    for i in {0..9}; do
        # BERT training
        python traintest_text.py --data-dir $wav_dir/../cv_folds/fold_0$i --exp-dir $wav_dir/../cv_folds/fold_0$i/bert_2class --loss bce --n_class 1
    done
    cd ..
fi

if [ $stage -le 41 ] && [ $stop_stage -ge 41 ]; then
    cd plm
    for i in {0..9}; do
        # BERT training
        python traintest_text.py --data-dir $wav_dir/../cv_folds/fold_0$i --exp-dir $wav_dir/../cv_folds/fold_0$i/bert_class12 --loss bce --n_class 1 --class_set 1 2
    done
    cd ..
fi

if [ $stage -le 50 ] && [ $stop_stage -ge 50 ]; then
    cd plm
    for i in {0..9}; do
        # BERT training
        python traintest_text.py --data-dir $wav_dir/../cv_folds/fold_0$i --exp-dir $wav_dir/../cv_folds/fold_0$i/bert_3class --loss ce --n_class 3
    done
    cd ..
fi

if [ $stage -le 60 ] && [ $stop_stage -ge 60 ]; then
    for i in {0..9}; do
#        python train_whisper.py \
#            --manifest_dir $wav_dir/../cv_folds/fold_0$i \
#            --exp_dir $wav_dir/../cv_folds/fold_0$i \
#            --mode test

        cd plm
        python traintest_text.py --mode eval --data-dir $wav_dir/../cv_folds/fold_0$i --exp-dir $wav_dir/../cv_folds/fold_0$i/bert_2class --loss bce --n_class 1
        cd .. 

        python utils/audio_text_fusion.py \
            --audio-score-path $wav_dir/../cv_folds/fold_0$i/test_pred_scores.npy \
            --text-score-path $wav_dir/../cv_folds/fold_0$i/bert_2class/preds/top120_test_pred_scores.npy \
            --true-label-path $wav_dir/../cv_folds/fold_0$i/bert_2class/preds/top120_test_gold_label.npy \
            --out-path $wav_dir/../cv_folds/fold_0$i/test_fusion_result.tsv
    done
fi

if [ $stage -le 61 ] && [ $stop_stage -ge 61 ]; then
    for i in {0..9}; do
        python train_whisper.py \
            --manifest_dir $wav_dir/../cv_folds/fold_0$i \
            --exp_dir $wav_dir/../cv_folds/fold_0$i/whisper_class12 \
            --mode test --class_set 1 2

        cd plm
        python traintest_text.py --mode eval --data-dir $wav_dir/../cv_folds/fold_0$i --exp-dir $wav_dir/../cv_folds/fold_0$i/bert_class12 --loss bce --n_class 1 --class_set 1 2
        cd .. 

        python utils/audio_text_fusion.py \
            --audio-score-path $wav_dir/../cv_folds/fold_0$i/whisper_class12/test_pred_scores.npy \
            --text-score-path $wav_dir/../cv_folds/fold_0$i/bert_class12/preds/top120_test_pred_scores.npy \
            --true-label-path $wav_dir/../cv_folds/fold_0$i/bert_class12/preds/top120_test_gold_label.npy \
            --out-path $wav_dir/../cv_folds/fold_0$i/test_fusion_result.tsv
    done
fi

if [ $stage -le 62 ] && [ $stop_stage -ge 62 ]; then
    for i in {0..9}; do
        python train_whisper_3class.py \
            --manifest_dir $wav_dir/../cv_folds/fold_0$i \
            --exp_dir $wav_dir/../cv_folds/fold_0$i/whisper_3class \
            --mode test

        cd plm
        python traintest_text.py --mode eval --data-dir $wav_dir/../cv_folds/fold_0$i --exp-dir $wav_dir/../cv_folds/fold_0$i/bert_3class --loss ce --n_class 3
        cd .. 

        python utils/audio_text_fusion.py \
            --audio-score-path $wav_dir/../cv_folds/fold_0$i/whisper_3class/test_pred_scores.npy \
            --text-score-path $wav_dir/../cv_folds/fold_0$i/bert_3class/preds/top120_test_pred_scores.npy \
            --true-label-path $wav_dir/../cv_folds/fold_0$i/bert_3class/preds/top120_test_gold_label.npy \
            --out-path $wav_dir/../cv_folds/fold_0$i/test_fusion_result.tsv
    done
fi

if [ $stage -le 70 ] && [ $stop_stage -ge 70 ]; then
    for i in {0..9}; do
        for x in train test; do
            python utils/csv_to_wrd_from_template.py \
            --csv $wav_dir/../LEADS_MIT_MGB_Transcripts.csv \
            --template-wrd $wav_dir/../cv_folds/fold_0$i/${x}_whisper.wrd \
            --output $wav_dir/../cv_folds/fold_0$i/$x.wrd
        done
    done
fi

if [ $stage -le 80 ] && [ $stop_stage -ge 80 ]; then
    cd plm
    for i in {0..9}; do
        for n in 1; do
        # BERT training
        python traintest_text.py \
            --data-dir $wav_dir/../cv_folds/fold_0$i \
            --exp-dir $wav_dir/../cv_folds/fold_0$i/bert_${n}class_gt_transcripts \
            --n_class ${n} \
            --gt_transcripts
        # --loss ce
        done
    done
    cd ..
fi

if [ $stage -le 81 ] && [ $stop_stage -ge 81 ]; then
    cd plm
    for i in {0..9}; do
        for n in 1; do
        # BERT training
        python traintest_text.py \
            --data-dir $wav_dir/../cv_folds/fold_0$i \
            --exp-dir $wav_dir/../cv_folds/fold_0$i/bert_class12_gt_transcripts \
            --n_class ${n} \
            --class_set 1 2 \
            --gt_transcripts
        # --loss ce
        done
    done
    cd ..
fi

if [ $stage -le 82 ] && [ $stop_stage -ge 82 ]; then
    cd plm
    for i in {0..9}; do
        for n in 3; do
        # BERT training
        python traintest_text.py \
            --data-dir $wav_dir/../cv_folds/fold_0$i \
            --exp-dir $wav_dir/../cv_folds/fold_0$i/bert_3class_gt_transcripts \
            --n_class ${n} \
            --class_set 0 1 2 \
            --gt_transcripts \
            --loss ce \
            --dataset leads
        done
    done
    cd ..
fi

if [ $stage -le 90 ] && [ $stop_stage -ge 90 ]; then
    cd plm
    for i in {0..9}; do
        for n in 1; do
        # Clinical BERT training
        python traintest_text_clinical_bert.py \
            --lm medicalai/ClinicalBERT \
            --data-dir $wav_dir/../cv_folds/fold_0$i \
            --exp-dir $wav_dir/../cv_folds/fold_0$i/clinical_bert_${n}class \
            --n_class ${n} #\
#           --gt_transcripts
#           --loss ce
        done
    done
    cd ..
fi

if [ $stage -le 91 ] && [ $stop_stage -ge 91 ]; then
    cd plm
    for i in {0..9}; do
        for n in 1; do
        # Clinical BERT training
        python traintest_text_clinical_bert.py \
            --lm medicalai/ClinicalBERT \
            --data-dir $wav_dir/../cv_folds/fold_0$i \
            --exp-dir $wav_dir/../cv_folds/fold_0$i/clinical_bert_${n}class \
            --n_class ${n} --class_set 1 2 #\
#           --gt_transcripts
#           --loss ce
        done
    done
    cd ..
fi

if [ $stage -le 92 ] && [ $stop_stage -ge 92 ]; then
    cd plm
    for i in {0..9}; do
        for n in 3; do
        # Clinical BERT training
        python traintest_text_clinical_bert.py \
            --lm medicalai/ClinicalBERT \
            --data-dir $wav_dir/../cv_folds/fold_0$i \
            --exp-dir $wav_dir/../cv_folds/fold_0$i/clinical_bert_${n}class \
            --n_class ${n} --class_set 0 1 2 \
            --loss ce
#           --gt_transcripts
        done
    done
    cd ..
fi

if [ $stage -le 93 ] && [ $stop_stage -ge 93 ]; then
    cd plm
    for i in {0..9}; do
        echo fold $i
        python traintest_text_clinical_bert.py --mode eval --data-dir $wav_dir/../cv_folds/fold_0$i --exp-dir $wav_dir/../cv_folds/fold_0$i/clinical_bert_3class --loss ce --n_class 3 --class_set 0 1 2 --gt_transcripts
    done
    python utils/audio_text_fusion.py \
        --audio-score-path $wav_dir/../cv_folds/fold_0$i/whisper_3class/test_pred_scores.npy \
        --text-score-path $wav_dir/../cv_folds/fold_0$i/clinical_bert_3class/preds/top120_test_pred_scores.npy \
        --true-label-path $wav_dir/../cv_folds/fold_0$i/clinical_bert_3class/preds/top120_test_gold_label.npy \
        --out-path $wav_dir/../cv_folds/fold_0$i/test_3class_clinical_bert_fusion_result.tsv
    cd .. 
fi

if [ $stage -le 100 ] && [ $stop_stage -ge 100 ]; then
    for i in {0..9}; do
        echo fold $i
#        python train_whisper.py \
#            --manifest_dir $wav_dir/../cv_folds/fold_0$i \
#            --exp_dir $wav_dir/../cv_folds/fold_0$i \
#            --mode test

        cd plm
        python traintest_text.py --mode eval --data-dir $wav_dir/../cv_folds/fold_0$i --exp-dir $wav_dir/../cv_folds/fold_0$i/bert_1class_gt_transcripts --loss bce --n_class 1 --gt_transcripts
        cd ..

        python utils/audio_text_fusion.py \
            --audio-score-path $wav_dir/../cv_folds/fold_0$i/test_pred_scores.npy \
            --text-score-path $wav_dir/../cv_folds/fold_0$i/bert_1class_gt_transcripts/preds/top120_test_pred_scores.npy \
            --true-label-path $wav_dir/../cv_folds/fold_0$i/bert_1class_gt_transcripts/preds/top120_test_gold_label.npy \
            --out-path $wav_dir/../cv_folds/fold_0$i/test_gt_transcript_fusion_result.tsv
    done
fi

if [ $stage -le 101 ] && [ $stop_stage -ge 101 ]; then
    for i in {0..9}; do
        echo fold $i
#        python train_whisper.py \
#            --manifest_dir $wav_dir/../cv_folds/fold_0$i \
#            --exp_dir $wav_dir/../cv_folds/fold_0$i/whisper_class12 \
#            --mode test --class_set 1 2

        cd plm
        python traintest_text.py --mode eval --data-dir $wav_dir/../cv_folds/fold_0$i --exp-dir $wav_dir/../cv_folds/fold_0$i/bert_class12_gt_transcripts --loss bce --n_class 1 --class_set 1 2 --gt_transcripts
        cd .. 

        python utils/audio_text_fusion.py \
            --audio-score-path $wav_dir/../cv_folds/fold_0$i/whisper_class12/test_pred_scores.npy \
            --text-score-path $wav_dir/../cv_folds/fold_0$i/bert_class12_gt_transcripts/preds/top120_test_pred_scores.npy \
            --true-label-path $wav_dir/../cv_folds/fold_0$i/bert_class12_gt_transcripts/preds/top120_test_gold_label.npy \
            --out-path $wav_dir/../cv_folds/fold_0$i/test_12_gt_transcript_fusion_result.tsv
    done
fi

if [ $stage -le 102 ] && [ $stop_stage -ge 102 ]; then
    for i in {0..9}; do
        echo fold $i
        cd plm
        python traintest_text.py --mode eval --data-dir $wav_dir/../cv_folds/fold_0$i --exp-dir $wav_dir/../cv_folds/fold_0$i/bert_3class_gt_transcripts --loss ce --n_class 3 --class_set 0 1 2 --gt_transcripts
        cd .. 

        python utils/audio_text_fusion.py \
            --audio-score-path $wav_dir/../cv_folds/fold_0$i/whisper_3class/test_pred_scores.npy \
            --text-score-path $wav_dir/../cv_folds/fold_0$i/bert_3class_gt_transcripts/preds/top120_test_pred_scores.npy \
            --true-label-path $wav_dir/../cv_folds/fold_0$i/bert_3class_gt_transcripts/preds/top120_test_gold_label.npy \
            --out-path $wav_dir/../cv_folds/fold_0$i/test_3class_gt_transcript_fusion_result.tsv
    done
fi

if [ $stage -le 110 ] && [ $stop_stage -ge 110 ]; then
cd plm
python utils/shap_analysis.py --dataset leads
cd ..
fi
