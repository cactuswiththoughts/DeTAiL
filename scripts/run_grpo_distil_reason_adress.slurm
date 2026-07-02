#!/bin/bash
#SBATCH -J grpo_distil_reason_trace_adress
#SBATCH -o logs/%j_grpo_distil_reason_trace_adress.out
#SBATCH -e logs/%j_grpo_distil_reason_trace_adress.err
#SBATCH --qos=regular
##SBATCH --gres=gpu:4
#SBATCH --gres=gpu:2
##SBATCH --gres=gpu:1
#SBATCH --nodes=1
##SBATCH --partition=a5
#SBATCH --partition=a6
#SBATCH --requeue

export HOME=/data/sls/scratch/limingw
export CONDA_ROOT=/data/sls/scratch/limingw/miniconda3
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate ms_swift
ulimit -s unlimited
cd ..

stage=50
stop_stage=50
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]
then
FOLD_DIR="/data/sls/scratch/limingw/workplace/speaker_anonymization/data/adress"
for x in train test
do
python utils/build_ms_swift_rationale_dataset.py \
    --input_jsonl "/data/sls/scratch/limingw/workplace/speaker_anonymization/data/adress/${x}_2class_grpo.jsonl" \
    --output_jsonl "/data/sls/scratch/limingw/workplace/speaker_anonymization/data/adress/${x}_gemini_gold_rationale_prompts.jsonl" \
    --labels control,dementia

python utils/build_ms_swift_rationale_dataset_no_label.py \
    --input_jsonl "/data/sls/scratch/limingw/workplace/speaker_anonymization/data/adress/${x}_2class_grpo.jsonl" \
    --output_jsonl "/data/sls/scratch/limingw/workplace/speaker_anonymization/data/adress/${x}_gemini_rationale_prompts.jsonl" \
    --labels control,dementia
done
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]
then
pip install google-genai

FOLD_DIR="/data/sls/scratch/limingw/workplace/speaker_anonymization/data/adress"
for x in train test
do
#python utils/generate_ms_swift_rationales_gemini.py \
#  --input_jsonl $FOLD_DIR/${x}_gemini_gold_rationale_prompts.jsonl \
#  --output_jsonl $FOLD_DIR/${x}_gemini_gold_rationales.jsonl \
#  --model gemini-3.5-flash \
#  --sleep 1 \
#  --overwrite

python utils/generate_ms_swift_rationales_gemini.py \
  --input_jsonl $FOLD_DIR/${x}_gemini_rationale_prompts.jsonl \
  --output_jsonl $FOLD_DIR/${x}_gemini_rationales.jsonl \
  --model gemini-3.5-flash \
  --sleep 1 \
  --overwrite
done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]
then
FOLD_DIR="/data/sls/scratch/limingw/workplace/speaker_anonymization/data/adress"
for x in train test
do
python utils/generate_reasoning_trace_html.py \
  --input $FOLD_DIR/${x}_gemini_gold_rationales.jsonl \
  --metadata $FOLD_DIR/${x}_gemini_gold_rationale_prompts.jsonl \
  --output $FOLD_DIR/${x}_gemini_gold_rationales.html

python utils/generate_reasoning_trace_html.py \
  --input $FOLD_DIR/${x}_gemini_rationales.jsonl \
  --metadata $FOLD_DIR/${x}_gemini_rationale_prompts.jsonl \
  --output $FOLD_DIR/${x}_gemini_rationales.html
done
fi

if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]
then
FOLD_DIR="/data/sls/scratch/limingw/workplace/speaker_anonymization/data/adress"
for x in train test
do
python utils/prepare_leads_ms_swift_format_sft_with_rationales.py \
    --fold_dir $FOLD_DIR \
    --metadata $FOLD_DIR/${x}_2class_grpo.jsonl \
    --rationales $FOLD_DIR/train_gemini_gold_rationales.jsonl $FOLD_DIR/test_gemini_gold_rationales.jsonl \
    --split $x \
    --output_name $FOLD_DIR/${x}_2class_student_sft_with_rationales.jsonl

python utils/prepare_leads_ms_swift_format_grpo.py \
    --fold_dir $FOLD_DIR \
    --metadata $FOLD_DIR/${x}_2class_grpo.jsonl \
    --split $x \
    --output_name $FOLD_DIR/${x}_2class_grpo_distill.jsonl
done
fi

if [ $stage -le 20 ] && [ $stop_stage -ge 20 ]
then
lora_rank=128
FOLD_DIR="/data/sls/scratch/limingw/workplace/speaker_anonymization/data/adress"

echo SFT with rationales
swift sft \
  --model Qwen/Qwen2.5-Omni-7B \
  --tuner_type lora \
  --dataset $FOLD_DIR/train_2class_student_sft_with_rationales.jsonl \
  --torch_dtype bfloat16 \
  --num_train_epochs 20 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --lora_rank $lora_rank \
  --lora_alpha 32 \
  --target_modules all-linear \
  --max_length 4096 \
  --eval_steps 50 \
  --save_steps 50 \
  --save_total_limit 2 \
  --logging_steps 5 \
  --output_dir output/Qwen2.5-Omni-7B/adress/sft_2class_with_rationale_lora_rank_$lora_rank \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 4 \
  --dataset_num_proc 4
fi

if [ $stage -le 21 ] && [ $stop_stage -ge 21 ]
then
#lora_rank=32
lora_rank=128
echo Inference for SFT with rationales
output_dir=output/Qwen2.5-Omni-7B/adress/sft_2class_with_rationale_lora_rank_$lora_rank
#output_dir=output/Qwen2.5-Omni-7B/adress/sft_2class_with_rationale
last_ckpt_dir=$(ls -vd $output_dir/v*-*-* | tail -n 1)
last_ckpt_dir=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
echo $last_ckpt_dir

swift infer \
--adapters $last_ckpt_dir \
--stream true \
--temperature 0 \
--max_new_tokens 4096 \
--val_dataset "$PWD/../data/adress/test_2class_student_sft_with_rationales.jsonl"
fi

if [ $stage -le 30 ] && [ $stop_stage -ge 30 ]
then
#lora_rank=32
lora_rank=128
for n in 2
do
echo $n class, LLM probs
output_dir=output/Qwen2.5-Omni-7B/adress/sft_2class_with_rationale_lora_rank_$lora_rank
#output_dir=output/Qwen2.5-Omni-7B/adress/sft_2class_with_rationale
last_ckpt_dir=$(ls -vd $output_dir/v*-*-* | tail -n 1)
last_ckpt_dir=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
infer_result=$(ls -vd $last_ckpt_dir/infer_result/*.jsonl | tail -n 1)

python utils/eval_leads_grpo_llm_prob.py \
--response_path $infer_result \
--lbl_path $PWD/../data/adress/test.label \
--lora_path $last_ckpt_dir \
--n_class $n \
--out_dir $last_ckpt_dir/..
done
fi

if [ $stage -le 40 ] && [ $stop_stage -ge 40 ]
then
cd ms-swift
model_name=Qwen2.5-Omni-7B
#lora_rank=32
lora_rank=128

echo 2 class
FOLD_DIR="/data/sls/scratch/limingw/workplace/speaker_anonymization/data/adress"
output_dir=output/$model_name/adress/grpo_distill_2class_lora_rank_$lora_rank
#init_model_dir=../output/$model_name/adress/sft_2class_with_rationale
sft_model_dir=../output/$model_name/adress/sft_2class_with_rationale_lora_rank_$lora_rank
sft_ckpt_dir=$(ls -vd $sft_model_dir/v*-*-* | tail -n 1)
sft_ckpt_dir=$(ls -vd $sft_ckpt_dir/checkpoint-* | tail -n 1)
echo $sft_ckpt_dir
num_gen=4
n_epoch=1

swift rlhf \
    --rlhf_type grpo \
    --model Qwen/$model_name \
    --reward_funcs external_r1v_acc format \
    --reward_weights 1 0.5 \
    --tuner_type lora \
    --adapters $sft_ckpt_dir \
    --lora_rank $lora_rank \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset $FOLD_DIR/train_2class_grpo_distill.jsonl \
    --load_from_cache_file true \
    --external_plugins "examples/train/grpo/plugin/plugin.py" \
    --max_completion_length 4096 \
    --num_train_epochs $n_epoch \
    --per_device_eval_batch_size 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 8 \
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
    --log_completions true #\
#    --resume_from_checkpoint $last_ckpt_dir
#    --per_device_train_batch_size 2 \
#    --num_generations 8 \
#    --deepspeed zero2 \

cd ..
fi

if [ $stage -le 41 ] && [ $stop_stage -ge 41 ]; then
    cd ms-swift
    lora_rank=128
#    lora_rank=32

    for x in test train
    do
    output_dir=output/Qwen2.5-Omni-7B/adress/grpo_distill_2class_lora_rank_$lora_rank
    last_ckpt_dir=$(ls -vd $output_dir/v*-*-* | tail -n 1)
    last_ckpt_dir=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
    echo $last_ckpt_dir

    swift infer \
    --adapters $last_ckpt_dir \
    --stream true \
    --temperature 0 \
    --max_new_tokens 4096 \
    --val_dataset "$PWD/../../data/adress/${x}_2class_grpo_distill.jsonl"
    done
    cd ..
fi

if [ $stage -le 42 ] && [ $stop_stage -ge 42 ]; then
    lora_rank=128
#    lora_rank=32

    echo 2 class
    output_dir=ms-swift/output/Qwen2.5-Omni-7B/adress/grpo_distill_2class_lora_rank_$lora_rank
    last_ckpt_dir=$(ls -vd $output_dir/v*-*-* | tail -n 1)
    last_ckpt_dir=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
    infer_result=$(ls -vd $last_ckpt_dir/infer_result/* | head -n 1)
    echo $infer_result

    python utils/eval_leads_grpo_llm_prob.py \
        --lora_path $last_ckpt_dir \
        --response_path $infer_result \
        --lbl_path $PWD/../data/adress/test.label \
    	--n_class 2 \
        --out_dir $last_ckpt_dir/../
fi

if [ $stage -le 50 ] && [ $stop_stage -ge 50 ]; then
    lora_rank=128
    output_dir=ms-swift/output/Qwen2.5-Omni-7B/adress/grpo_distill_2class_lora_rank_$lora_rank
    last_ckpt_dir=$(ls -vd $output_dir/v*-*-* | tail -n 1)
    last_ckpt_dir=$(ls -vd $last_ckpt_dir/checkpoint-* | tail -n 1)
    infer_result=$(ls -vd $last_ckpt_dir/infer_result/* | head -n 1)
    echo $infer_result
    python utils/generate_reasoning_trace_html.py --input $infer_result --output $output_dir/visualize_reasoning_trace.html
fi