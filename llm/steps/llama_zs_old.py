import argparse
import numpy as np
from collections import defaultdict
import json
import os
import os.path as osp
from pathlib import Path
import pandas as pd
import random
from tqdm import tqdm
import transformers
from transformers import pipeline, AutoModel, AutoProcessor, AutoTokenizer
import torch
import torch.nn.functional as F
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to: {seed}")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='meta-llama/Llama-3.1-70B-Instruct')
    parser.add_argument('--tsv_path', default='/data/sls/scratch/limingw/workplace/speaker_anonymization/data/fhs/test_100.tsv')
    parser.add_argument('--label_path', default='/data/sls/scratch/limingw/workplace/speaker_anonymization/data/fhs/test_100.lbl')
    parser.add_argument('--label_set', default='normal,dementia')
    parser.add_argument('--text_path', default='/data/sls/scratch/limingw/workplace/speaker_anonymization/data/fhs/test_100_diarized_whisper.wrd')
    parser.add_argument('--out_path', default='/data/sls/scratch/limingw/workplace/speaker_anonymization/fhs_dementia_classifier/llm/exp/test_100_Llama-3.1-70B-Instruct')
    parser.add_argument('--prompt_type', default='simple')
    parser.add_argument('--max_new_token', type=int, default=32)
    return parser

def create_prompt(text, label_set, prompt_type='simple'):
    if prompt_type == 'simple':
        prompt = (
            "You are a helpful assistant that classifies if a participant in an interview has dementia.\n\n"
            f"Possible labels: {', '.join(label_set)}.\n"
            f"Interview transcript: \"{text}\"\n"
            "Label:"
        )
    elif prompt_type == 'cot':
        prompt = (
            "You are a helpful assistant that classifies if a participant in an interview has dementia.\n\n"
            f"Possible labels: {', '.join(label_set)}.\n"
            f"Interview transcript: \"{text}\"\n"
            "Answer: let's think step by step."
        )
    else: raise ValueError(f'Unknown prompt type: {prompt_type}')
    
    return prompt

def classify(pipe, tok, text, label_set, args):
    prompt = create_prompt(text, label_set=label_set, prompt_type=args.prompt_type)
    out = pipe(prompt, return_full_text=False)[0]["generated_text"].strip()
    print('out:', out, flush=True)
    label_set = [l.lower() for l in label_set]
    if args.prompt_type == 'cot':
        label = 'unknown'
        explanation = out
        probs = F.one_hot(
            torch.tensor(len(label_set)), num_classes=len(label_set)+1)
        for lbl in label_set:
            if lbl in out.lower():
                label = lbl
                probs = F.one_hot(
                    torch.tensor(label_set.index(lbl)), 
                    num_classes=len(label_set)+1)
    else:
        label = out.split()[0].lower() if len(out.split()) else 'unknown' # first token is the label
        explanation = ''
        if len(out.split()) > 1:
            explanation = ' '.join(out.split()[1:]).lower()
        #label_ids = tok(label_set, add_special_tokens=False).input_ids
        #label_ids = [lab[0] for lab in label_ids]
        #logits = out.logits[:, -1]
        #label_logits = logits[0, label_ids]
        #probs = label_logits.softmax(-1)
        if not label in label_set:
            label = 'unknown'
            probs = F.one_hot(
                torch.tensor(len(label_set)),
                num_classes=len(label_set)+1,
            )
        else:
            probs = F.one_hot(
                torch.tensor(label_set.index(label)),
                num_classes=len(label_set)+1,
            )

    return label, probs, explanation

def main():
    set_seed()
    parser = get_parser()
    args = parser.parse_args()
    Path(args.out_path).parent.mkdir(exist_ok=True, parents=True)
    out_label_path = args.out_path+'.lbl'
    out_explanation_path = args.out_path+'_explanation.lbl'
    out_gold_label_path = args.out_path+'_gold.lbl'
    out_score_path = args.out_path+'.npy'
    out_res_path = args.out_path+'_result.txt'
    label_set = args.label_set.split(',')

    print(args.model_name)
    pipe = pipeline(
        "text-generation",
        model=args.model_name,
        tokenizer=args.model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        max_new_tokens=args.max_new_token,
        device_map='auto',
    )
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    fns = []
    with open(args.tsv_path, 'r') as f_tsv,\
        open(args.text_path, 'r') as f,\
        open(args.label_path, 'r') as f_lab:

        lines = f_tsv.read().strip().split('\n')
        _ = lines.pop(0)
        texts = f.read().strip().split('\n')
        labels = f_lab.read().strip().split('\n')
        wav2text = defaultdict(list)
        wav2lab = {}
        root = lines.pop(0)
        for l, y in tqdm(list(zip(lines, labels))):
            fn = l.strip().split('\t')[0]
            if not fn in fns:
                fns.append(fn)
                wav2lab[fn] = label_set[int(y)]

        for l in tqdm(texts):
            fn = l.strip().split('\t')[0] 
            text = ''.join(l.strip().split('\t')[3:])
            wav2text[fn].append(text)

    scores = []
    pred_labels = []
    gold_labels = []
    with open(out_label_path, 'w') as f_out,\
        open(out_gold_label_path, 'w') as f_out_gold,\
        open(out_explanation_path, 'w') as f_out_exp,\
        open(out_res_path, 'w') as f_out_res:
        for fn in tqdm(fns):
            text = '\n'.join(wav2text[fn])
            label, score, explanation = classify(pipe, tok, text, label_set, args)
            print(f'{fn}\t{label}', file=f_out)
            print(f'{fn}\t{wav2lab[fn]}', file=f_out_gold)
            print(f'{fn}\t{explanation}', file=f_out_exp)
            pred_labels.append(label)
            gold_labels.append(wav2lab[fn])
            scores.append(score.detach().cpu().numpy())
        correct = [y_p == y for y_p, y in zip(pred_labels, gold_labels)]
        acc = np.asarray(correct).mean()
        print(f'Accuracy: {acc}')
        print(acc, file=f_out_res)

    scores = np.stack(scores)
    np.save(out_score_path, scores)

if __name__ == '__main__':
    main()
