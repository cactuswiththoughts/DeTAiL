#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a classifier on frozen Qwen prompt, reasoning, and audio embeddings."""

import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


LABELS = [
    "control",
    "alzheimers_disease",
    "non_ad_cognitive_impairment",
]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
DEFAULT_TEST_JSONL = (
    "/data/sls/scratch/limingw/workplace/speaker_anonymization/data/leads/"
    "cv_folds/fold_00/test_3class_sft.jsonl"
)
DEFAULT_TRAIN_JSONL = (
    "/data/sls/scratch/limingw/workplace/speaker_anonymization/data/leads/"
    "cv_folds/fold_00/train_3class_grpo_distill.jsonl"
)
DEFAULT_REASONING_DIR = (
    "/data/sls/scratch/limingw/workplace/speaker_anonymization/DementiaTTRL/"
    "ms-swift/output/Qwen2.5-Omni-7B/leads/fold_00/"
    "grpo_distill_3class_lora_rank_128/v1-20260607-115026/checkpoint-578/"
    "infer_result"
)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("train", "eval", "extract"), default="train")
    parser.add_argument("--test-jsonl", default=DEFAULT_TEST_JSONL)
    parser.add_argument(
        "--train-jsonl",
        default=DEFAULT_TRAIN_JSONL,
        help=(
            "JSONL supplying labels; its prompt is replaced by the matching "
            "GRPO inference prompt."
        ),
    )
    parser.add_argument(
        "--reasoning-dir",
        default=DEFAULT_REASONING_DIR,
        help=(
            "Directory (or JSONL file) containing generated GRPO responses. "
            "Used for training and, unless overridden, testing."
        ),
    )
    parser.add_argument(
        "--test-reasoning-dir",
        "--test_reasoning_dir",
        default=None,
        help=(
            "Optional reasoning directory (or JSONL file) for the test split. "
            "Defaults to --reasoning-dir."
        ),
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Omni-7B")
    parser.add_argument(
        "--lora-path",
        "--lora_path",
        default="",
        help="Optional PEFT/LoRA adapter path or Hugging Face model ID.",
    )
    parser.add_argument("--exp-dir", default="./exp/mlp_qwen_omni")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Embedding cache directory; defaults to EXP_DIR/embeddings.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-epochs", type=int, default=40)
    parser.add_argument("--lr", "--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-7)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--classifier-depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--class_set",
        "--class-set",
        type=int,
        nargs="+",
        choices=(0, 1, 2),
        default=[0, 1, 2],
        help=(
            "0 1 2: three-class cross entropy; 0 1: control vs merged "
            "Alzheimer/non-AD impairment with BCE; 1 2: filter control and "
            "classify Alzheimer vs non-AD impairment with BCE."
        ),
    )
    parser.add_argument(
        "--topk",
        default="2,4,6,8,10,20,40,80,120,160",
        help="Numbers of most-confident segments used for speaker voting.",
    )
    parser.add_argument(
        "--vote-type", choices=("hard", "soft"), default="hard"
    )
    parser.add_argument(
        "--pooling", choices=("mean", "last"), default="mean"
    )
    parser.add_argument(
        "--layer-index",
        type=int,
        default=-1,
        help=(
            "Hidden-state index to pool embeddings from. "
            "-1 selects the last layer (default); 0 is the embedding layer."
        ),
    )
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--audio-sampling-rate", type=int, default=16000)
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default=None,
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map used while extracting embeddings.",
    )
    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Recompute embeddings even when a cache exists.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Classifier checkpoint; defaults to EXP_DIR/models/best_text_model.pth.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def speaker_from_audio(path):
    return Path(path).name.split("_")[0]


def normalized_audio_path(path):
    return str(Path(path).expanduser().resolve())


def reasoning_jsonl_paths(path):
    path = Path(path).expanduser()
    if path.is_file():
        return [path.resolve()]
    if not path.is_dir():
        raise FileNotFoundError(f"Reasoning path does not exist: {path}")
    files = sorted(file.resolve() for file in path.glob("*.jsonl"))
    if not files:
        raise ValueError(f"No JSONL files found in reasoning directory {path}.")
    return files


def extract_thinking_prefix(response, source):
    match = re.search(
        r"<think>(.*?)</think>",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        match = re.search(
            r"<text>(.*?)</text>",
            response,
            flags=re.IGNORECASE | re.DOTALL,
        ) 
        
    if not match:
        #raise ValueError(f"{source} response has no complete <think> block.")
        print(f"Warning: {source} response has no complete <think> block.")
        reasoning = "" 
    else:
        reasoning = match.group(1).strip()
        if not reasoning:
            raise ValueError(f"{source} response has an empty <think> block.")
    return f"<think>\n{reasoning}\n</think>\n<answer>"


def load_reasoning_index(path):
    index = {}
    for jsonl_path in reasoning_jsonl_paths(path):
        with open(jsonl_path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                audios = record.get("audios", [])
                if not audios:
                    raise ValueError(
                        f"{jsonl_path}:{line_number} has no audio path."
                    )
                key = normalized_audio_path(audios[0])
                # XXX if key in index:
                #    raise ValueError(
                #        f"Duplicate generated trace for audio {key}: "
                #        f"{jsonl_path}:{line_number}"
                #    )
                response = record.get("response")
                if not isinstance(response, str):
                    raise ValueError(
                        f"{jsonl_path}:{line_number} has no generated response."
                    )
                prompt_messages = [
                    {"role": message["role"], "content": message["content"]}
                    for message in record.get("messages", [])
                    if message.get("role") != "assistant"
                ]
                index[key] = {
                    "messages": prompt_messages or None,
                    "thinking_prefix": extract_thinking_prefix(
                        response, f"{jsonl_path}:{line_number}"
                    ),
                }
    return index


def label_from_example(example, path, line_number):
    solution = example.get("solution")
    if isinstance(solution, str):
        return solution.strip()
    assistant_messages = [
        message
        for message in example.get("messages", [])
        if message.get("role") == "assistant"
    ]
    if not assistant_messages:
        raise ValueError(
            f"{path}:{line_number} has neither solution nor assistant label."
        )
    return assistant_messages[-1]["content"].strip()


def read_jsonl(path, reasoning_index):
    examples = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            example = json.loads(line)
            messages = example.get("messages", [])
            audios = example.get("audios", [])
            if not messages or not audios:
                raise ValueError(
                    f"{path}:{line_number} must contain messages and audios."
                )
            label_name = label_from_example(example, path, line_number)
            if label_name not in LABEL_TO_ID:
                raise ValueError(
                    f"{path}:{line_number} has unsupported label {label_name!r}."
                )
            missing = [audio for audio in audios if not os.path.isfile(audio)]
            if missing:
                raise FileNotFoundError(
                    f"{path}:{line_number} audio does not exist: {missing[0]}"
                )
            audio_key = normalized_audio_path(audios[0])
            trace = reasoning_index.get(audio_key)
            if trace is None:
                raise ValueError(
                    f"{path}:{line_number} has no generated trace for {audio_key}."
                )
            prompt_messages = trace["messages"]
            if prompt_messages is None:
                prompt_messages = [
                    {"role": message["role"], "content": message["content"]}
                    for message in messages
                    if message.get("role") != "assistant"
                ]
            examples.append(
                {
                    "messages": prompt_messages
                    + [
                        {
                            "role": "assistant",
                            "content": trace["thinking_prefix"],
                        }
                    ],
                    "audios": audios,
                    "label": LABEL_TO_ID[label_name],
                    "speaker": speaker_from_audio(audios[0]),
                }
            )
    if not examples:
        raise ValueError(f"No examples found in {path}.")
    return examples


def torch_dtype_from_name(name):
    if name == "auto":
        return "auto"
    return getattr(torch, name)


def lora_fingerprint(lora_path):
    if not lora_path:
        return ""
    path = Path(lora_path).expanduser()
    if not path.exists():
        return lora_path
    path = path.resolve()
    if path.is_file():
        stat = path.stat()
        return f"{path}:{stat.st_size}:{stat.st_mtime_ns}"

    parts = [str(path)]
    adapter_files = sorted(path.glob("adapter_config.json"))
    adapter_files.extend(sorted(path.glob("adapter_model*")))
    for file_path in adapter_files:
        if file_path.is_file():
            stat = file_path.stat()
            parts.append(
                f"{file_path.name}:{stat.st_size}:{stat.st_mtime_ns}"
            )
    return "|".join(parts)


def load_omni(
    model_name,
    lora_path,
    torch_dtype,
    device_map,
    attn_implementation,
):
    try:
        from transformers import (
            Qwen2_5OmniForConditionalGeneration,
            Qwen2_5OmniProcessor,
        )
    except ImportError as exc:
        raise ImportError(
            "Qwen2.5-Omni requires a recent transformers installation. "
            "Install the Qwen2.5-Omni requirements before running this script."
        ) from exc

    processor = Qwen2_5OmniProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    model_kwargs = {
        "torch_dtype": torch_dtype_from_name(torch_dtype),
        "device_map": device_map,
        "trust_remote_code": True,
    }
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_name,
        **model_kwargs,
    )
    if lora_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError(
                "Loading --lora-path requires the peft package."
            ) from exc
        print(f"Loading LoRA adapter from {lora_path}", flush=True)
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return processor, model.thinker


def model_input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except (AttributeError, StopIteration):
        return next(model.parameters()).device


def render_prompt(processor, messages):
    if messages and messages[-1].get("role") == "assistant":
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def load_audio_files(paths, sampling_rate):
    try:
        import librosa
    except ImportError as exc:
        raise ImportError(
            "Audio embedding extraction requires librosa. Install the "
            "Qwen2.5-Omni audio dependencies in the training environment."
        ) from exc
    return [
        librosa.load(path, sr=sampling_rate, mono=True)[0]
        for path in paths
    ]


def pool_hidden_state(hidden_state, attention_mask, pooling):
    if pooling == "last":
        lengths = attention_mask.long().sum(dim=1).clamp_min(1) - 1
        row_indices = torch.arange(hidden_state.size(0), device=hidden_state.device)
        return hidden_state[row_indices, lengths]

    mask = attention_mask.to(hidden_state.device).unsqueeze(-1)
    mask = mask.to(hidden_state.dtype)
    return (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


@torch.inference_mode()
def extract_embeddings(
    examples,
    processor,
    model,
    pooling,
    layer_index,
    sampling_rate,
    max_length,
    description,
):
    embeddings = []
    input_device = model_input_device(model)

    for example in tqdm(examples, desc=description):
        prompt = render_prompt(processor, example["messages"])
        audios = load_audio_files(example["audios"], sampling_rate)
        processor_kwargs = {
            "text": prompt,
            "audio": audios,
            "return_tensors": "pt",
            "padding": True,
        }
        if max_length is not None:
            processor_kwargs.update(
                max_length=max_length,
                truncation=True,
            )
        inputs = processor(**processor_kwargs)
        inputs = {
            key: value.to(input_device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        if outputs.hidden_states is None:
            raise RuntimeError("Qwen thinker did not return hidden states.")
        hidden_states = outputs.hidden_states
        if layer_index < -len(hidden_states) or layer_index >= len(hidden_states):
            raise ValueError(
                f"layer_index {layer_index} out of range for hidden_states "
                f"(length {len(hidden_states)}, valid range: "
                f"{-len(hidden_states)}..{len(hidden_states) - 1})"
            )
        embedding = pool_hidden_state(
            hidden_states[layer_index],
            inputs["attention_mask"],
            pooling,
        )
        embeddings.append(embedding[0].float().cpu())

    return torch.stack(embeddings)


def cache_path(cache_dir, split):
    return Path(cache_dir) / f"{split}_qwen_omni_embeddings.pt"


def reasoning_sources(path):
    return [
        {
            "path": str(file_path),
            "size": file_path.stat().st_size,
            "mtime_ns": file_path.stat().st_mtime_ns,
        }
        for file_path in reasoning_jsonl_paths(path)
    ]


def cache_matches(cache_file, jsonl_path, reasoning_dir, args):
    if args.overwrite_cache or not cache_file.exists():
        return False
    try:
        payload = torch.load(cache_file, map_location="cpu")
        example_count = sum(
            bool(line.strip())
            for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines()
        )
        return (
            payload.get("source") == str(Path(jsonl_path).resolve())
            and payload.get("source_size") == Path(jsonl_path).stat().st_size
            and payload.get("source_mtime_ns")
            == Path(jsonl_path).stat().st_mtime_ns
            and payload.get("model_name") == args.model_name
            and payload.get("lora_fingerprint")
            == lora_fingerprint(args.lora_path)
            and payload.get("pooling") == args.pooling
            and payload.get("layer_index", -1) == args.layer_index
            and payload.get("max_length") == args.max_length
            and payload.get("audio_sampling_rate") == args.audio_sampling_rate
            and payload.get("reasoning_sources")
            == reasoning_sources(reasoning_dir)
            and payload["embeddings"].shape[0] == example_count
        )
    except (KeyError, OSError, RuntimeError, TypeError):
        return False


def load_or_extract(
    jsonl_path,
    split,
    args,
    processor=None,
    model=None,
    reasoning_index=None,
    reasoning_dir=None,
):
    if reasoning_dir is None:
        reasoning_dir = args.reasoning_dir
    if reasoning_index is None:
        reasoning_index = load_reasoning_index(reasoning_dir)
    examples = read_jsonl(jsonl_path, reasoning_index)
    output_path = cache_path(args.cache_dir, split)
    # XXX if cache_matches(output_path, jsonl_path, reasoning_dir, args):
    if output_path.exists():
        payload = torch.load(output_path, map_location="cpu")
        print(f"Loaded {split} embeddings from {output_path}", flush=True)
        return payload
    if output_path.exists():
        print(f"Ignoring stale embedding cache {output_path}", flush=True)

    if processor is None or model is None:
        raise RuntimeError("The Qwen model is required to create an embedding cache.")
    embeddings = extract_embeddings(
        examples=examples,
        processor=processor,
        model=model,
        pooling=args.pooling,
        layer_index=args.layer_index,
        sampling_rate=args.audio_sampling_rate,
        max_length=args.max_length,
        description=f"Extracting {split}",
    )
    payload = {
        "embeddings": embeddings,
        "labels": torch.tensor(
            [example["label"] for example in examples], dtype=torch.long
        ),
        "speakers": [example["speaker"] for example in examples],
        "source": str(Path(jsonl_path).resolve()),
        "source_size": Path(jsonl_path).stat().st_size,
        "source_mtime_ns": Path(jsonl_path).stat().st_mtime_ns,
        "model_name": args.model_name,
        "lora_path": args.lora_path,
        "lora_fingerprint": lora_fingerprint(args.lora_path),
        "pooling": args.pooling,
        "layer_index": args.layer_index,
        "max_length": args.max_length,
        "audio_sampling_rate": args.audio_sampling_rate,
        "reasoning_sources": reasoning_sources(reasoning_dir),
        "label_names": LABELS,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"Saved {split} embeddings to {output_path}", flush=True)
    return payload


class EmbeddingDataset(Dataset):
    def __init__(self, payload, split):
        self.embeddings = payload["embeddings"].float()
        self.labels = payload["labels"].long()
        self.speakers = payload["speakers"]
        self.split = split

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.embeddings[index], self.labels[index], self.speakers[index]


def classification_config(class_set):
    class_set = tuple(class_set)
    if class_set == (0, 1, 2):
        return {
            "binary": False,
            "n_outputs": 3,
            "labels": LABELS,
        }
    if class_set == (0, 1):
        return {
            "binary": True,
            "n_outputs": 1,
            "labels": ["control", "cognitive_impairment"],
        }
    if class_set == (1, 2):
        return {
            "binary": True,
            "n_outputs": 1,
            "labels": [
                "alzheimers_disease",
                "non_ad_cognitive_impairment",
            ],
        }
    raise ValueError(
        "--class_set must be '0 1 2', '0 1', or '1 2'. "
        "Class IDs are 0=control, 1=Alzheimer's disease, "
        "2=non-AD cognitive impairment."
    )


def apply_class_set(payload, class_set):
    config = classification_config(class_set)
    labels = payload["labels"].long()

    if tuple(class_set) == (0, 1):
        # Control stays 0; both impairment classes become positive.
        keep = torch.ones_like(labels, dtype=torch.bool)
        mapped_labels = (labels != 0).long()
    elif tuple(class_set) == (1, 2):
        # Remove controls, then map Alzheimer=0 and non-AD impairment=1.
        keep = labels != 0
        mapped_labels = (labels[keep] == 2).long()
    else:
        keep = torch.ones_like(labels, dtype=torch.bool)
        mapped_labels = labels

    indices = keep.nonzero(as_tuple=False).flatten()
    transformed = dict(payload)
    transformed["embeddings"] = payload["embeddings"][indices]
    transformed["labels"] = mapped_labels
    transformed["speakers"] = [payload["speakers"][index] for index in indices.tolist()]
    transformed["label_names"] = config["labels"]
    return transformed


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth, n_class, dropout):
        super().__init__()
        if depth < 0:
            raise ValueError("--classifier-depth must be non-negative.")
        layers = []
        current_dim = input_dim
        for _ in range(depth):
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, n_class))
        self.classifier = nn.Sequential(*layers)

    def forward(self, embeddings):
        return self.classifier(embeddings)


def majority_vote(segment_probs, topk, vote_type, n_class):
    probs = np.asarray(segment_probs)
    count = min(topk, len(probs))
    indices = np.argsort(probs.max(axis=1))[::-1][:count]
    selected = probs[indices]
    if vote_type == "soft":
        score = selected.mean(axis=0)
        return int(score.argmax()), score
    votes = np.bincount(selected.argmax(axis=1), minlength=n_class)
    return int(votes.argmax()), selected.mean(axis=0)


def safe_auc(gold_labels, pred_scores, n_class):
    gold = np.asarray(gold_labels)
    scores = np.asarray(pred_scores)
    if n_class == 2:
        if np.unique(gold).size < 2:
            return float("nan")
        return float(roc_auc_score(gold, scores[:, 1]))

    one_hot = np.eye(n_class, dtype=np.int64)[gold]
    aucs = []
    for class_index in range(n_class):
        if np.unique(one_hot[:, class_index]).size < 2:
            continue
        aucs.append(roc_auc_score(one_hot[:, class_index], scores[:, class_index]))
    return float(np.mean(aucs)) if aucs else float("nan")


@torch.inference_mode()
def validate(classifier, loader, args):
    device = next(classifier.parameters()).device
    config = classification_config(args.class_set)
    n_class = len(config["labels"])
    classifier.eval()
    speaker_probs = defaultdict(list)
    speaker_gold = {}

    for embeddings, labels, speakers in loader:
        logits = classifier(embeddings.to(device))
        if config["binary"]:
            positive_probs = logits.flatten().sigmoid()
            probs = torch.stack((1.0 - positive_probs, positive_probs), dim=-1)
        else:
            probs = logits.softmax(dim=-1)
        probs = probs.cpu().numpy()
        for prob, label, speaker in zip(probs, labels.numpy(), speakers):
            speaker_probs[speaker].append(prob)
            previous = speaker_gold.setdefault(speaker, int(label))
            if previous != int(label):
                raise ValueError(f"Speaker {speaker} has inconsistent labels.")

    metrics = {}
    pred_dir = Path(args.exp_dir) / "preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for topk in [int(value) for value in args.topk.split(",")]:
        gold_labels = []
        pred_labels = []
        pred_scores = []
        speaker_ids = []
        for speaker in sorted(speaker_gold):
            prediction, score = majority_vote(
                speaker_probs[speaker], topk, args.vote_type, n_class
            )
            speaker_ids.append(speaker)
            gold_labels.append(speaker_gold[speaker])
            pred_labels.append(prediction)
            pred_scores.append(score)

        precision, recall, macro_f1, _ = precision_recall_fscore_support(
            gold_labels,
            pred_labels,
            average="macro",
            zero_division=0,
        )
        _, _, micro_f1, _ = precision_recall_fscore_support(
            gold_labels,
            pred_labels,
            average="micro",
            zero_division=0,
        )
        auc = safe_auc(gold_labels, pred_scores, n_class)
        metrics[topk] = {
            "precision": precision,
            "recall": recall,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "auc": auc,
        }
        prefix = pred_dir / f"top{topk}_{loader.dataset.split}"
        np.save(f"{prefix}_speaker.npy", np.asarray(speaker_ids))
        np.save(f"{prefix}_gold_label.npy", np.asarray(gold_labels))
        np.save(f"{prefix}_pred_label.npy", np.asarray(pred_labels))
        np.save(f"{prefix}_pred_scores.npy", np.asarray(pred_scores))
        np.save(
            f"{prefix}_confusion.npy",
            confusion_matrix(gold_labels, pred_labels, labels=range(n_class)),
        )
    return metrics


def print_metrics(metrics, best_auc=None):
    for topk in sorted(metrics):
        row = metrics[topk]
        message = (
            f"Top {topk}, Precision: {row['precision']:.3f}, "
            f"Recall: {row['recall']:.3f}, Micro F1: {row['micro_f1']:.3f}, "
            f"Macro F1: {row['macro_f1']:.3f}, AUC: {row['auc']:.3f}"
        )
        if best_auc is not None:
            message += f", Best AUC: {best_auc:.3f}"
        print(message, flush=True)


def checkpoint_path(args):
    if args.checkpoint:
        return Path(args.checkpoint)
    return Path(args.exp_dir) / "models" / "best_text_model.pth"


def save_checkpoint(classifier, path, args, input_dim, epoch, best_auc):
    config = classification_config(args.class_set)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": classifier.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "depth": args.classifier_depth,
            "dropout": args.dropout,
            "n_class": config["n_outputs"],
            "epoch": epoch,
            "best_auc": best_auc,
            "labels": config["labels"],
            "class_set": list(args.class_set),
        },
        path,
    )


def build_classifier_from_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    classifier = MLPClassifier(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        depth=checkpoint["depth"],
        n_class=checkpoint["n_class"],
        dropout=checkpoint.get("dropout", 0.0),
    )
    classifier.load_state_dict(checkpoint["model_state_dict"])
    return classifier.to(device), checkpoint


def train_classifier(train_payload, test_payload, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = classification_config(args.class_set)
    train_dataset = EmbeddingDataset(train_payload, "train")
    test_dataset = EmbeddingDataset(test_payload, "test")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    input_dim = train_dataset.embeddings.shape[1]
    classifier = MLPClassifier(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        depth=args.classifier_depth,
        n_class=config["n_outputs"],
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.95, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(range(20, 100, 5)),
        gamma=0.5,
    )
    loss_fn = nn.BCEWithLogitsLoss() if config["binary"] else nn.CrossEntropyLoss()
    best_auc = -float("inf")
    rows = []
    start_time = time.time()

    print(f"Training classifier on {device}; embedding dimension={input_dim}")
    for epoch in range(args.n_epochs):
        classifier.train()
        epoch_loss = 0.0
        for embeddings, labels, _ in train_loader:
            embeddings = embeddings.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(embeddings)
            if config["binary"]:
                loss = loss_fn(logits.flatten(), labels.float())
            else:
                loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(labels)
        scheduler.step()

        metrics = validate(classifier, test_loader, args)
        finite_aucs = [
            row["auc"] for row in metrics.values() if np.isfinite(row["auc"])
        ]
        epoch_auc = max(finite_aucs, default=-float("inf"))
        if epoch_auc > best_auc:
            best_auc = epoch_auc
            save_checkpoint(
                classifier,
                checkpoint_path(args),
                args,
                input_dim,
                epoch,
                best_auc,
            )
        average_loss = epoch_loss / len(train_dataset)
        print(
            f"Epoch {epoch + 1}/{args.n_epochs}, loss={average_loss:.6f}, "
            f"lr={optimizer.param_groups[0]['lr']:.6g}",
            flush=True,
        )
        print_metrics(metrics, best_auc)
        for topk, metric in metrics.items():
            rows.append(
                {
                    "Epoch": epoch,
                    "Train Loss": average_loss,
                    "Top k": topk,
                    "Test Precision": metric["precision"],
                    "Test Recall": metric["recall"],
                    "Test Micro F1": metric["micro_f1"],
                    "Test Macro F1": metric["macro_f1"],
                    "Test AUC": metric["auc"],
                    "Best AUC": best_auc,
                }
            )
        pd.DataFrame(rows).to_csv(Path(args.exp_dir) / "result.csv", index=False)

    print(f"Total training time: {time.time() - start_time:.1f}s")


def evaluate_classifier(test_payload, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier, checkpoint = build_classifier_from_checkpoint(
        checkpoint_path(args), device
    )
    checkpoint_class_set = checkpoint.get("class_set", [0, 1, 2])
    if list(args.class_set) != checkpoint_class_set:
        raise ValueError(
            f"Checkpoint class_set={checkpoint_class_set}, but evaluation "
            f"requested class_set={list(args.class_set)}."
        )
    dataset = EmbeddingDataset(test_payload, "test")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    metrics = validate(classifier, loader, args)
    print(
        f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}",
        flush=True,
    )
    print_metrics(metrics)


def main():
    args = parse_args()
    set_seed(args.seed)
    args.test_jsonl = str(Path(args.test_jsonl).resolve())
    args.train_jsonl = str(Path(args.train_jsonl).resolve())
    args.reasoning_dir = str(Path(args.reasoning_dir).resolve())
    if args.test_reasoning_dir is None:
        args.test_reasoning_dir = args.reasoning_dir
    else:
        args.test_reasoning_dir = str(Path(args.test_reasoning_dir).resolve())
    Path(args.exp_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir is None:
        args.cache_dir = str(Path(args.exp_dir) / "embeddings")

    required_splits = ["test"]
    if args.mode in ("train", "extract"):
        required_splits.insert(0, "train")
    paths = {"train": args.train_jsonl, "test": args.test_jsonl}
    reasoning_dirs = {
        "train": args.reasoning_dir,
        "test": args.test_reasoning_dir,
    }
    need_model = any(
        not cache_matches(
            cache_path(args.cache_dir, split),
            paths[split],
            reasoning_dirs[split],
            args,
        )
        for split in required_splits
    )
    processor = model = None
    reasoning_indexes = {
        reasoning_dir: load_reasoning_index(reasoning_dir)
        for reasoning_dir in {
            reasoning_dirs[split] for split in required_splits
        }
    }
    if need_model:
        processor, model = load_omni(
            model_name=args.model_name,
            lora_path=args.lora_path,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            attn_implementation=args.attn_implementation,
        )

    payloads = {
        split: load_or_extract(
            paths[split],
            split,
            args,
            processor=processor,
            model=model,
            reasoning_index=reasoning_indexes[reasoning_dirs[split]],
            reasoning_dir=reasoning_dirs[split],
        )
        for split in required_splits
    }
    payloads = {
        split: apply_class_set(payload, args.class_set)
        for split, payload in payloads.items()
    }
    config = classification_config(args.class_set)
    for split, payload in payloads.items():
        counts = np.bincount(
            payload["labels"].numpy(),
            minlength=len(config["labels"]),
        )
        distribution = ", ".join(
            f"{label}={count}"
            for label, count in zip(config["labels"], counts)
        )
        print(f"{split}: {len(payload['labels'])} segments ({distribution})")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.mode == "train":
        train_classifier(payloads["train"], payloads["test"], args)
    elif args.mode == "eval":
        evaluate_classifier(payloads["test"], args)
    else:
        print("Embedding extraction complete.", flush=True)


if __name__ == "__main__":
    main()
