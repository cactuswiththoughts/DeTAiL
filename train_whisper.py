import argparse
import copy
import functools
import json
import os
import math
import numpy as np
import random
import soundfile as sf
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from typing import Tuple
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.functional as T 

import params
from model.classifier import WhisperMLPClassifier
from model.utils import MelSpectrogramFixed
from leads_data import LEADS


def set_seed(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def majority_vote(pred_labels, pred_scores, topk, vote_type='hard', n_class=1):
    if n_class == 1:
        top_indices = sorted(np.arange(len(pred_scores)), key=lambda x:max(pred_scores[x], 1-pred_scores[x]), reverse=True)[:topk]
        vote = np.zeros(2)
        prob = 0.0
        for i in top_indices:
            vote[int(pred_labels[i])] += 1
            prob += pred_scores[i]
    else:
        n_class = pred_scores[0].shape[-1]
        top_indices = sorted(np.arange(len(pred_scores)), key=lambda x:pred_scores[x].max(), reverse=True)[:topk]
        vote = np.zeros(n_class)
        prob = np.zeros(n_class)
        for i in top_indices:
            vote[int(pred_labels[i])] += 1
            prob += pred_scores[i]
    return int(vote.argmax()), prob / len(top_indices)


def compute_dementia_scores(pred_labels_dict, pred_scores_dict, gold_labels_dict, return_prob=False, n_class=1):
    pred_labels_list = [y for spk in sorted(pred_labels_dict) for y in pred_labels_dict[spk]]
    gold_labels_list = [y for spk in sorted(pred_labels_dict) for y in gold_labels_dict[spk]]
    _, _, utt_acc, _ = precision_recall_fscore_support(
        gold_labels_list, pred_labels_list, average='micro',
    )
 
    aucs, precisions, recalls, micro_f1s, macro_f1s = [], [], [], [], [] 
    for topk in [5, 10, 50, 100]: 
        all_pred_scores = []
        all_pred_labels = []
        all_gold_labels = []
        for spk, gold_label in sorted(gold_labels_dict.items()):
            pred_label, pred_score = majority_vote(pred_labels_dict[spk], pred_scores_dict[spk], n_class=1, topk=10)
            all_pred_labels.append(pred_label)
            all_pred_scores.append(pred_score)
            all_gold_labels.append(gold_label[0])

        pred_labels = np.asarray(all_pred_labels)
        pred_scores = np.asarray(all_pred_scores)
        gold_labels = np.asarray(all_gold_labels)

        if len(set(gold_labels.tolist())) == 1:
            auc = 0
        elif n_class == 2 and pred_scores.ndim == 1:
            auc = roc_auc_score(gold_labels, pred_scores)
        else:
            auc = roc_auc_score(
                gold_labels, pred_scores, multi_class='ovr', average='macro',
            )
        _, _, micro_f1, _ = precision_recall_fscore_support(
            gold_labels, pred_labels, average='micro',
        )
        precision, recall, macro_f1, _ = precision_recall_fscore_support(
            gold_labels, pred_labels, average='macro',
        )
        aucs.append(auc)
        precisions.append(precision)
        recalls.append(recall)
        micro_f1s.append(micro_f1)
        macro_f1s.append(macro_f1)
    
    res = {'auc': aucs, 'precision': precisions, 'recall': recalls, 'macro_f1': macro_f1s, 'micro_f1': micro_f1s, 'utt_acc': utt_acc}
    if return_prob:
        return res, pred_scores, gold_labels
    return res


def train_classifier(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    outdir = Path(args.exp_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    set_seed(42)

    audio_model = WhisperMLPClassifier('small').to(device)
    train_data = LEADS(args.manifest_dir, 'train', merge_dur=30, class_set=args.class_set, dataset=args.dataset)
    val_data = LEADS(args.manifest_dir, 'test', merge_dur=30, class_set=args.class_set, dataset=args.dataset)

    n_mels = params.n_mels
    sampling_rate = params.sampling_rate
    n_fft = params.n_fft
    hop_size = params.hop_size
    f_max = params.f_max

    mel_fn = MelSpectrogramFixed(
        sample_rate=sampling_rate,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_size,
        f_min=0,
        f_max=f_max,
        n_mels=n_mels,
        window_fn=torch.hann_window 
    ).cuda()

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=train_data.collate_fn) 
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=val_data.collate_fn) 
   
    print('running on ' + str(device))
    torch.set_grad_enabled(True)

    pred_scores_dict = defaultdict(list)
    pred_labels_dict = defaultdict(list)
    gold_labels_dict = defaultdict(list)
    exp_dir = Path(args.exp_dir) 

    # Set up the optimizer
    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(20, 100, 5)), gamma=0.5, last_epoch=-1)

    loss_fn = nn.BCEWithLogitsLoss()

    begin_time = time.time()
    total_loss = 0
    epoch = 0
    global_step = 0
    best_auc = 0
    
    print("current #steps=%s, #epochs=%s" % (global_step, epoch), flush=True)
    print("start training...", flush=True)

    for epoch in range(args.n_epochs):
        audio_model.train()
        for i, batch in enumerate(train_loader):
            wav = batch['wavs'].to(device, non_blocking=True)
            mel = mel_fn(wav)[...,:-1]
            feat_lengths = batch['feat_lengths'].to(device, non_blocking=True)
            label = batch['dementia_labels'].to(device, non_blocking=True)
            # warmup
            warm_up_step = 100
            if global_step <= warm_up_step and global_step % 5 == 0:
                warm_lr = (global_step / warm_up_step) * args.lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warm_lr
                print('warm-up learning rate is {:f}'.format(optimizer.param_groups[0]['lr']), flush=True)
        
            logits = audio_model(wav, feat_lengths)
            loss = loss_fn(logits.flatten(), label.flatten().float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            total_loss += loss.detach().cpu().item()
        
        # Validation
        audio_model.eval()
        res_all, pred_scores_all, gold_labels_all = test_classifier(args, audio_model, {'adress': val_loader})
        auc = sum([max(res['auc']) for k, res in res_all.items()]) / len(res_all)

        if best_auc < auc:
            best_auc = auc
            for name, res in res_all.items():
                auc = max(res['auc'])
                precision = max(res['precision'])
                recall = max(res['recall'])
                micro_f1 = max(res['micro_f1'])
                macro_f1 = max(res['macro_f1'])
                utt_acc = res['utt_acc']        
                with open(exp_dir / 'dementia_classifier_result.txt', 'a') as f_res:

                    print(f'Dataset: {name}, Per-utterance Micro F1: {utt_acc:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, Micro F1: {micro_f1:.3f}, Macro F1: {macro_f1:.3f}, AUC: {auc:.3f}, Best AUC: {best_auc:.3f}', file=f_res)
                print(f'Dataset: {name}, Per-utterance Micro F1: {utt_acc:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, Micro F1: {micro_f1:.3f}, Macro F1: {macro_f1:.3f}, AUC: {auc:.3f}, Best AUC: {best_auc:.3f}', flush=True)
        
            # Save model
            print(f'Save model with AUC = {best_auc}')
            torch.save(audio_model.state_dict(), exp_dir / 'dementia_classifier.pt')
            np.save(str(exp_dir / 'test_pred_scores.npy'), pred_scores_all)
            np.save(str(exp_dir / 'test_labels.npy'), gold_labels_all)

        audio_model.train()
    return audio_model


def test_classifier(args, audio_model, test_loaders):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    set_seed(42)

    n_mels = params.n_mels
    sampling_rate = params.sampling_rate
    n_fft = params.n_fft
    hop_size = params.hop_size
    f_max = params.f_max

    mel_fn = MelSpectrogramFixed(
        sample_rate=sampling_rate,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_size,
        f_min=0,
        f_max=f_max,
        n_mels=n_mels,
        window_fn=torch.hann_window 
    ).cuda()

    print('running on ' + str(device))
    torch.set_grad_enabled(True)

    pred_scores_dict = defaultdict(list)
    pred_labels_dict = defaultdict(list)
    gold_labels_dict = defaultdict(list)

    audio_model.eval()
    res_all = {}
    with torch.no_grad():
        for name, test_loader in test_loaders.items():
            for i, batch in enumerate(test_loader):
                wav = batch['wavs']
                wav = wav.to(device, non_blocking=True)
                mel = mel_fn(wav)[...,:-1]
                feat_lengths = batch['feat_lengths'].to(device, non_blocking=True)
                dementia_labels = batch['dementia_labels'].to(device, non_blocking=True)
                spks = batch['speaker_labels']
                logits = audio_model(wav, feat_lengths)
                pred_labels = (logits > 0).long()
                for logit, label, pred_label, spk in zip(logits, dementia_labels, pred_labels, spks):
                    pred_scores_dict[spk].append(logit.sigmoid().detach().cpu().numpy())
                    pred_labels_dict[spk].append(pred_label.detach().cpu().item())
                    gold_labels_dict[spk].append(label.detach().cpu().item()) 
            
            # Accuracy at the utterance level
            res, pred_scores, gold_labels = compute_dementia_scores(pred_labels_dict, pred_scores_dict, gold_labels_dict, return_prob=True, n_class=args.n_class)
            res_all[name] = res
    return res_all, pred_scores, gold_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest_dir', default='/data/sls/scratch/limingw/workplace/speaker_anonymization/data/fhs')
    parser.add_argument('--exp_dir', default='')
    parser.add_argument('--n_epochs', type=int, default=50) 
    parser.add_argument('--n_class', type=int, default=1)
    parser.add_argument('--num_all_per_class', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=int, default=1e-4)
    parser.add_argument('--seed_model_dir', default='')
    parser.add_argument('--dataset', default='leads')
    parser.add_argument('--mode', type=str, choices={'train', 'test'}, default='train')
    parser.add_argument('--class_set', nargs='+', type=int, choices=[0, 1, 2], default=[0, 1, 2])
    args = parser.parse_args()
   
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    audio_model = WhisperMLPClassifier('small').to(device)
    exp_dir = Path(args.exp_dir)
    if not args.seed_model_dir:
        seed_dir = exp_dir
    else:
        seed_dir = Path(args.seed_model_dir)
    if (seed_dir / 'dementia_classifier.pt').exists():
        audio_model.load_state_dict(torch.load(str(seed_dir / 'dementia_classifier.pt')))

    print(f'args.manifest_dir: {args.manifest_dir}')  # XXX
    if args.mode == 'test':        
        test_data_dict = {
            'leads': LEADS(args.manifest_dir, 'test', class_set=args.class_set, dataset=args.dataset),
        }

        test_loaders = {
            k: DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=test_data.collate_fn) 
            for k, test_data in test_data_dict.items()
        }

        res, pred_scores_all, gold_labels_all = test_classifier(args, audio_model, test_loaders)
        np.save(str(exp_dir / 'test_pred_scores.npy'), pred_scores_all)
        np.save(str(exp_dir / 'test_labels.npy'), gold_labels_all)

        for name in res:
            auc = max(res[name]['auc'])
            precision = max(res[name]['precision'])
            recall = max(res[name]['recall'])
            micro_f1 = max(res[name]['micro_f1'])
            macro_f1 = max(res[name]['macro_f1'])
            utt_acc = res[name]['utt_acc']
            with open(exp_dir / 'dementia_classifier_result.txt', 'a') as f_res:
                 print(f'Dataset: {name}, Per-utterance Micro F1: {utt_acc:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, Micro F1: {micro_f1:.3f}, Macro F1: {macro_f1:.3f}, AUC: {auc:.3f}', file=f_res)     
            print(f'Dataset: {name}, Per-utterance Micro F1: {utt_acc:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, Micro F1: {micro_f1:.3f}, Macro F1: {macro_f1:.3f}, AUC: {auc:.3f}', flush=True)
    else:
        train_classifier(args)
