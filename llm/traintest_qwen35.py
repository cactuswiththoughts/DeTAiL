#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a dementia classifier on frozen Qwen prompt/media embeddings.

This script supports either audio input or image input, but not both in the
same JSONL file/run.

Expected JSONL example format, audio:
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "<audio>\nTranscript: ..."},
    {"role": "assistant", "content": "control"}
  ],
  "audios": ["/path/to/audio.wav"]
}

Expected JSONL example format, image:
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "<image>\nTranscript: ..."},
    {"role": "assistant", "content": "control"}
  ],
  "images": ["/path/to/mel.png"]
}
"""

import argparse
import json
import os
import random
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


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("train", "eval", "extract"), default="train")
    parser.add_argument("--test-jsonl", default=DEFAULT_TEST_JSONL)
    parser.add_argument(
        "--train-jsonl",
        default=None,
        help="Defaults to train_3class_sft.jsonl beside --test-jsonl.",
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Omni-7B")
    parser.add_argument(
        "--lora-path",
        "--lora_path",
        default="",
        help="Optional PEFT/LoRA adapter path or Hugging Face model ID.",
    )
    parser.add_argument("--exp-dir", default="./exp/mlp_qwen_media")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Embedding cache directory; defaults to EXP_DIR/embeddings.",
    )
    parser.add_argument(
        "--modality",
        choices=("auto", "audio", "image"),
        default="auto",
        help=(
            "Input modality. 'auto' infers from JSONL keys. The run must be "
            "audio-only or image-only; mixed image/audio examples are rejected."
        ),
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
    parser.add_argument("--vote-type", choices=("hard", "soft"), default="hard")
    parser.add_argument("--pooling", choices=("mean", "last"), default="mean")
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


def speaker_from_media(path):
    return Path(path).name.split("_")[0]


def get_example_modality(example, path, line_number, requested_modality):
    audios = example.get("audios") or []
    images = example.get("images") or []

    has_audio = len(audios) > 0
    has_image = len(images) > 0

    if has_audio and has_image:
        raise ValueError(
            f"{path}:{line_number} contains both 'audios' and 'images'. "
            "Each example must use exactly one modality."
        )
    if not has_audio and not has_image:
        raise ValueError(
            f"{path}:{line_number} must contain exactly one of 'audios' or 'images'."
        )

    inferred = "audio" if has_audio else "image"
    if requested_modality != "auto" and inferred != requested_modality:
        raise ValueError(
            f"{path}:{line_number} has modality {inferred!r}, but "
            f"--modality={requested_modality!r} was requested."
        )

    media_paths = audios if inferred == "audio" else images
    return inferred, media_paths


def read_jsonl(path, requested_modality="auto"):
    examples = []
    file_modality = None

    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue

            example = json.loads(line)
            messages = example.get("messages", [])
            if not messages:
                raise ValueError(f"{path}:{line_number} must contain messages.")

            assistant_messages = [
                message for message in messages if message.get("role") == "assistant"
            ]
            if not assistant_messages:
                raise ValueError(f"{path}:{line_number} has no assistant label.")

            label_name = assistant_messages[-1]["content"].strip()
            if label_name not in LABEL_TO_ID:
                raise ValueError(
                    f"{path}:{line_number} has unsupported label {label_name!r}."
                )

            modality, media_paths = get_example_modality(
                example=example,
                path=path,
                line_number=line_number,
                requested_modality=requested_modality,
            )

            if file_modality is None:
                file_modality = modality
            elif file_modality != modality:
                raise ValueError(
                    f"{path}:{line_number} has modality {modality!r}, but earlier "
                    f"examples use {file_modality!r}. Mixed-modality files are not supported."
                )

            missing = [media for media in media_paths if not os.path.isfile(media)]
            if missing:
                raise FileNotFoundError(
                    f"{path}:{line_number} media file does not exist: {missing[0]}"
                )

            examples.append(
                {
                    # The assistant message is deliberately excluded to avoid leakage.
                    "messages": [
                        message
                        for message in messages
                        if message.get("role") != "assistant"
                    ],
                    "media": media_paths,
                    "modality": modality,
                    "label": LABEL_TO_ID[label_name],
                    "speaker": speaker_from_media(media_paths[0]),
                }
            )

    if not examples:
        raise ValueError(f"No examples found in {path}.")

    return examples, file_modality


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
            parts.append(f"{file_path.name}:{stat.st_size}:{stat.st_mtime_ns}")
    return "|".join(parts)


def load_qwen_model(
    model_name,
    lora_path,
    torch_dtype,
    device_map,
    attn_implementation,
    modality,
):
    """Load a Qwen-family model and return processor, model_for_embedding.

    Notes:
      - For audio, use an Omni model such as Qwen2.5-Omni or Qwen3-Omni.
      - For image, use a vision-capable Qwen model such as Qwen3.5/Qwen3.6-VL
        or another compatible AutoProcessor + model pair.
    """
    from transformers import AutoProcessor

    model_name_lower = model_name.lower()

    model_kwargs = {
        "torch_dtype": torch_dtype_from_name(torch_dtype),
        "device_map": device_map,
        "trust_remote_code": True,
    }
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    if "qwen2.5-omni" in model_name_lower or "qwen2_5-omni" in model_name_lower:
        try:
            from transformers import Qwen2_5OmniForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Qwen2.5-Omni requires a recent transformers installation "
                "with Qwen2_5OmniForConditionalGeneration."
            ) from exc

        full_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_name,
            **model_kwargs,
        )
        model_for_embedding = full_model.thinker

    elif "qwen3-omni" in model_name_lower:
        try:
            from transformers import Qwen3OmniMoeForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Qwen3-Omni requires a recent transformers installation "
                "with Qwen3OmniMoeForConditionalGeneration."
            ) from exc

        full_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_name,
            **model_kwargs,
        )
        model_for_embedding = full_model.thinker if hasattr(full_model, "thinker") else full_model

    elif "qwen3.5" in model_name_lower or "qwen3.6" in model_name_lower:
        # Class name may differ across transformers versions/checkpoints.
        # Try the explicit Qwen3_5 class first, then fall back to AutoModel.
        try:
            from transformers import Qwen3_5ForConditionalGeneration

            full_model = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_name,
                **model_kwargs,
            )
        except (ImportError, AttributeError):
            from transformers import AutoModelForCausalLM

            full_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs,
            )
        model_for_embedding = full_model.thinker if hasattr(full_model, "thinker") else full_model

    else:
        # Generic fallback for compatible Qwen-family checkpoints.
        try:
            from transformers import AutoModelForImageTextToText

            full_model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                **model_kwargs,
            )
        except (ImportError, ValueError, OSError):
            from transformers import AutoModelForCausalLM

            full_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs,
            )
        model_for_embedding = full_model.thinker if hasattr(full_model, "thinker") else full_model

    if modality == "audio" and "omni" not in model_name_lower:
        print(
            "WARNING: --modality=audio is usually only supported by Qwen Omni models. "
            f"Current --model-name={model_name!r}.",
            flush=True,
        )

    if lora_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("Loading --lora-path requires the peft package.") from exc

        print(f"Loading LoRA adapter from {lora_path}", flush=True)
        model_for_embedding = PeftModel.from_pretrained(model_for_embedding, lora_path)

    model_for_embedding.eval()
    for parameter in model_for_embedding.parameters():
        parameter.requires_grad_(False)

    return processor, model_for_embedding


def model_input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except (AttributeError, StopIteration):
        return next(model.parameters()).device


def load_audio_files(paths, sampling_rate):
    try:
        import librosa
    except ImportError as exc:
        raise ImportError(
            "Audio embedding extraction requires librosa. Install the "
            "Qwen audio dependencies in the training environment."
        ) from exc

    return [librosa.load(path, sr=sampling_rate, mono=True)[0] for path in paths]


def load_image_files(paths):
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Image embedding extraction requires pillow. Install it with: pip install pillow"
        ) from exc

    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
    return images


def pool_hidden_state(hidden_state, attention_mask, pooling):
    if pooling == "last":
        lengths = attention_mask.long().sum(dim=1).clamp_min(1) - 1
        row_indices = torch.arange(hidden_state.size(0), device=hidden_state.device)
        return hidden_state[row_indices, lengths]

    mask = attention_mask.to(hidden_state.device).unsqueeze(-1)
    mask = mask.to(hidden_state.dtype)
    return (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

def normalize_multimodal_messages(messages, modality):
    normalized = []

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")

        if role != "user" or not isinstance(content, str):
            normalized.append(message)
            continue

        if modality == "image" and "<image>" in content:
            text = content.replace("<image>", "").strip()
            normalized.append(
                {
                    "role": role,
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": text},
                    ],
                }
            )
        elif modality == "audio" and "<audio>" in content:
            text = content.replace("<audio>", "").strip()
            normalized.append(
                {
                    "role": role,
                    "content": [
                        {"type": "audio"},
                        {"type": "text", "text": text},
                    ],
                }
            )
        else:
            normalized.append(message)

    return normalized


def render_prompt(processor, messages, modality):
    messages = normalize_multimodal_messages(messages, modality)

    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    rendered = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")

        if isinstance(content, list):
            text_parts = []
            for item in content:
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "image":
                    text_parts.append("<image>")
                elif item.get("type") == "audio":
                    text_parts.append("<audio>")
            content = "\n".join(text_parts)

        rendered.append(f"{role}: {content}")

    rendered.append("assistant:")
    return "\n".join(rendered)


def build_processor_inputs(
    processor,
    prompt,
    media_paths,
    modality,
    sampling_rate,
    max_length,
):
    processor_kwargs = {
        "text": prompt,
        "return_tensors": "pt",
        "padding": True,
    }

    if modality == "audio":
        processor_kwargs["audio"] = load_audio_files(media_paths, sampling_rate)
    elif modality == "image":
        processor_kwargs["images"] = load_image_files(media_paths)
    else:
        raise ValueError(f"Unsupported modality: {modality!r}")

    if max_length is not None:
        processor_kwargs.update(max_length=max_length, truncation=True)

    try:
        return processor(**processor_kwargs)
    except TypeError as exc:
        # Some processors use singular "image" rather than "images".
        if modality == "image" and "images" in processor_kwargs:
            processor_kwargs["image"] = processor_kwargs.pop("images")
            return processor(**processor_kwargs)
        raise exc


def model_compute_dtype(model):
    for p in model.parameters():
        if p.is_floating_point():
            return p.dtype
    return torch.float32


def move_inputs_to_model(inputs, device, dtype):
    moved = {}

    for key, value in inputs.items():
        if torch.is_tensor(value):
            value = value.to(device)

            # Cast only floating tensors.
            # Do NOT cast input_ids, attention_mask, position_ids, etc.
            if value.is_floating_point():
                value = value.to(dtype)

        moved[key] = value

    return moved


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
        prompt = render_prompt(
            processor,
            example["messages"],
            example["modality"],
        )
        inputs = build_processor_inputs(
            processor=processor,
            prompt=prompt,
            media_paths=example["media"],
            modality=example["modality"],
            sampling_rate=sampling_rate,
            max_length=max_length,
        )

        compute_dtype = model_compute_dtype(model)
        inputs = move_inputs_to_model(
            inputs=inputs,
            device=input_device,
            dtype=compute_dtype,
        )

        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        if outputs.hidden_states is None:
            raise RuntimeError("Model did not return hidden states.")

        if "attention_mask" not in inputs:
            raise RuntimeError(
                "Processor output has no attention_mask, so pooling cannot be computed."
            )

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


def safe_model_tag(model_name):
    return (
        model_name.replace("/", "_")
        .replace(".", "_")
        .replace("-", "_")
        .replace(":", "_")
    )


def cache_path(cache_dir, split, model_name, modality):
    model_tag = safe_model_tag(model_name)
    return Path(cache_dir) / f"{split}_{model_tag}_{modality}_embeddings.pt"


def count_jsonl_examples(jsonl_path):
    return sum(
        bool(line.strip())
        for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines()
    )


def cache_matches(cache_file, jsonl_path, args):
    if args.overwrite_cache or not cache_file.exists():
        return False

    try:
        payload = torch.load(cache_file, map_location="cpu")
        return (
            payload.get("source") == str(Path(jsonl_path).resolve())
            and payload.get("source_size") == Path(jsonl_path).stat().st_size
            and payload.get("source_mtime_ns") == Path(jsonl_path).stat().st_mtime_ns
            and payload.get("model_name") == args.model_name
            and payload.get("modality") in ("audio", "image")
            and (args.modality == "auto" or payload.get("modality") == args.modality)
            and payload.get("lora_fingerprint") == lora_fingerprint(args.lora_path)
            and payload.get("pooling") == args.pooling
            and payload.get("layer_index", -1) == args.layer_index
            and payload.get("max_length") == args.max_length
            and payload.get("audio_sampling_rate") == args.audio_sampling_rate
            and payload["embeddings"].shape[0] == count_jsonl_examples(jsonl_path)
        )
    except (KeyError, OSError, RuntimeError, TypeError):
        return False


def load_or_extract(
    jsonl_path,
    split,
    args,
    processor=None,
    model=None,
):
    examples, detected_modality = read_jsonl(jsonl_path, args.modality)
    output_path = cache_path(
        args.cache_dir,
        split,
        args.model_name,
        detected_modality,
    )

    if cache_matches(output_path, jsonl_path, args):
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
        "labels": torch.tensor([example["label"] for example in examples], dtype=torch.long),
        "speakers": [example["speaker"] for example in examples],
        "source": str(Path(jsonl_path).resolve()),
        "source_size": Path(jsonl_path).stat().st_size,
        "source_mtime_ns": Path(jsonl_path).stat().st_mtime_ns,
        "model_name": args.model_name,
        "modality": detected_modality,
        "lora_path": args.lora_path,
        "lora_fingerprint": lora_fingerprint(args.lora_path),
        "pooling": args.pooling,
        "layer_index": args.layer_index,
        "max_length": args.max_length,
        "audio_sampling_rate": args.audio_sampling_rate,
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
                speaker_probs[speaker],
                topk,
                args.vote_type,
                n_class,
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

    print(f"Training classifier on {device}; embedding dimension={input_dim}", flush=True)

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
        finite_aucs = [row["auc"] for row in metrics.values() if np.isfinite(row["auc"])]
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

    print(f"Total training time: {time.time() - start_time:.1f}s", flush=True)


def evaluate_classifier(test_payload, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier, checkpoint = build_classifier_from_checkpoint(checkpoint_path(args), device)

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


def infer_required_modality(paths, required_splits, requested_modality):
    detected_modalities = []
    for split in required_splits:
        _, modality = read_jsonl(paths[split], requested_modality)
        detected_modalities.append(modality)

    unique = sorted(set(detected_modalities))
    if len(unique) != 1:
        raise ValueError(
            f"Train/test modalities do not match: {dict(zip(required_splits, detected_modalities))}. "
            "A single run must be audio-only or image-only."
        )

    return unique[0]


def main():
    args = parse_args()
    set_seed(args.seed)

    args.test_jsonl = str(Path(args.test_jsonl).resolve())
    if args.train_jsonl is None:
        args.train_jsonl = str(Path(args.test_jsonl).with_name("train_3class_sft.jsonl"))
    else:
        args.train_jsonl = str(Path(args.train_jsonl).resolve())

    Path(args.exp_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir is None:
        args.cache_dir = str(Path(args.exp_dir) / "embeddings")

    required_splits = ["test"]
    if args.mode in ("train", "extract"):
        required_splits.insert(0, "train")

    paths = {"train": args.train_jsonl, "test": args.test_jsonl}

    run_modality = infer_required_modality(
        paths=paths,
        required_splits=required_splits,
        requested_modality=args.modality,
    )
    print(f"Detected modality: {run_modality}", flush=True)

    need_model = any(
        not cache_matches(
            cache_path(args.cache_dir, split, args.model_name, run_modality),
            paths[split],
            args,
        )
        for split in required_splits
    )

    processor = model = None
    if need_model:
        processor, model = load_qwen_model(
            model_name=args.model_name,
            lora_path=args.lora_path,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            attn_implementation=args.attn_implementation,
            modality=run_modality,
        )

    payloads = {
        split: load_or_extract(
            paths[split],
            split,
            args,
            processor=processor,
            model=model,
        )
        for split in required_splits
    }

    # Enforce same modality for loaded/cached payloads too.
    cached_modalities = {split: payload.get("modality") for split, payload in payloads.items()}
    if len(set(cached_modalities.values())) != 1:
        raise ValueError(f"Loaded payloads have inconsistent modalities: {cached_modalities}")

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
            f"{label}={count}" for label, count in zip(config["labels"], counts)
        )
        print(f"{split}: {len(payload['labels'])} segments ({distribution})", flush=True)

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
