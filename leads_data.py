import os
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as T
from torch.utils.data import Dataset
from tqdm import tqdm

from params import seed as random_seed
from utils.prepare_leads import materialize_leads_mit_mgb_fold_manifests
random.seed(random_seed)
DOWNSAMPLE_FACTOR = 320

# Default MIT-MGB spreadsheet (used when CV fold manifests are missing on disk)
DEFAULT_LEADS_MIT_MGB_XLSX = (
    "/data/sls/scratch/limingw/workplace/speaker_anonymization/data/leads/LEADS_MIT_MGB.xlsx"
)


def _utterance_duration_sec(fn: str, dataset: str='leads') -> float:
    parts = Path(fn).stem.split("_")
    if len(parts) >= 3:
        return float(parts[-2]) / 16000.0, float(parts[-1]) / 16000.0
    return float(parts[-2]) / 1000.0, float(parts[-1]) / 1000.0


def _speaker_from_path(fn: str) -> str:
    return Path(fn).name.split("_")[0]


class BaseAudioDataset(Dataset):
    def __init__(self, sample_rate=16000, hop_length=320, n_mels=80, n_class=2):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_class = n_class
        self.multiple = 64 * hop_length  # UNet alignment

    def load_wav(self, fn, start=None, end=None):
        wav, sr = torchaudio.load(fn, normalize=True)
        if start is not None:
            assert end is not None
            wav = wav[:, start:end]

        if sr != self.sample_rate:
            wav = T.resample(wav, orig_freq=sr, new_freq=self.sample_rate)
        wav, _ = self.pad_to_multiple(wav)
        feat_len = wav.shape[-1] // DOWNSAMPLE_FACTOR - 1
        return wav, feat_len

    def pad_to_multiple(self, x):
        Tlen = x.shape[-1]
        pad = (self.multiple - Tlen % self.multiple) % self.multiple + DOWNSAMPLE_FACTOR
        return F.pad(x, (0, pad)), pad

    def __getitem__(self, idx):
        spk, start, end, fns, lbl = self.data[idx]
        if self.n_class == 2:
            lbl = int(self.class_set.index(lbl) > 0)
        else:
            lbl = int(lbl)
        
        wavs = []
        for fn in fns:
            wav, feat_len = self.load_wav(fn)
            wavs.append(wav)
        wav = torch.cat(wavs, dim=-1)

        return {
            "wav": wav,
            "feature_length": feat_len,
            "speaker_label": spk,
            "dementia_label": lbl,
        }

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        B = len(batch)
        wav_list = [sample["wav"] for sample in batch]
        feat_lengths = [sample["feature_length"] for sample in batch]
        spk_labels = [sample["speaker_label"] for sample in batch]
        dementia_labels = [sample["dementia_label"] for sample in batch]
        train_frames = max(sample["wav"].shape[-1] for sample in batch)

        wavs = torch.zeros(B, train_frames, dtype=torch.float32)
        for i, wav in enumerate(wav_list):
            wav_len = wav.shape[-1]
            wavs[i, :wav_len] = wav[..., :wav_len]

        feat_lengths = torch.LongTensor(feat_lengths)
        dementia_labels = torch.LongTensor(dementia_labels)
        return {
            "wavs": wavs,
            "feat_lengths": feat_lengths,
            "speaker_labels": spk_labels,
            "dementia_labels": dementia_labels,
        }


class LEADS(BaseAudioDataset):
    """LEADS manifest layout matches ADReSS: {split}.tsv + {split}.label / {split}.lbl.

    For 10-fold CV, pass ``fold`` in ``0..n_splits-1`` and set ``data_dir`` to the dataset root
    (manifests live under ``data_dir/cv_subdir/fold_XX/``).

    If ``{split}.tsv`` is missing there, manifests are generated from the MIT-MGB spreadsheet
    (default: ``DEFAULT_LEADS_MIT_MGB_XLSX``), using speaker-grouped ``n_splits``-fold CV;
    override the file with ``xlsx_path``.
    """

    def __init__(
        self,
        data_dir,
        split,
        tgt_spks=None,
        n_class=2,
        min_sec=1,
        max_sec=10,
        sample_rate=16000,
        hop_length=320,
        fold=None,
        cv_subdir="cv_folds",
        n_splits=10,
        xlsx_path=None,
        merge_dur=30,
        class_set=[0, 1, 2],
        dataset='leads',
    ):
        super().__init__(sample_rate, hop_length, n_class=n_class)
        self.class_set = class_set
        assert len(class_set) >= n_class

        if tgt_spks is None:
            tgt_spks = []
        self.multiple = 64 * hop_length
        self.max_frame = ((max_sec * sample_rate) // self.multiple) * self.multiple

        manifest_dir = os.path.abspath(data_dir)
        if fold is not None:
            manifest_dir = os.path.join(manifest_dir, cv_subdir, f"fold_{int(fold):02d}")

        tsv_path = os.path.join(manifest_dir, f"{split}.tsv")
        with open(os.path.join(manifest_dir, f"{split}.tsv"), "r") as f:
            lines = f.read().strip().split("\n")
            root = lines.pop(0)

        label_path = os.path.join(manifest_dir, f"{split}.label")
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                labels = f.read().strip().split("\n")
        else:
            with open(os.path.join(manifest_dir, f"{split}.lbl"), "r") as f:
                labels = f.read().strip().split("\n")

        fns = [os.path.join(root, l.strip().split("\t")[0]) for l in lines]
        spks = [_speaker_from_path(fn) for fn in fns]
        start_ends = [_utterance_duration_sec(fn, dataset=dataset) for fn in fns]
        data_dict = defaultdict(list) 
        for fn, spk, lbl, (start, end) in zip(fns, spks, labels, start_ends):
            data_dict[spk].append([start, end, [fn], int(lbl)])
        
        if merge_dur > 0:
            data_dict = self.merge(data_dict, merge_dur)

        self.data = [[spk]+x for spk in sorted(data_dict) for x in data_dict[spk] if x[-1] in self.class_set]
        print(
            f"[LEADS] manifest_dir={manifest_dir} split={split} "
            f"Create {len(self.data)} samples from {len(fns)} utterances, min_sec={min_sec}, max_sec={max_sec}"
        )

    def merge(self, data_dict, merge_dur):
        new_data_dict = defaultdict(list)
        total = 0
        total_merged = 0
        for id_date in tqdm(list(sorted(data_dict))):
            flist = sorted(data_dict[id_date], key=lambda x:x[0])
            start, end, fns, label = flist[0]
            new_fn = [start, end, fns, label]
            new_flist = []
            for start, end, fns, label in flist[1:]:
                total += 1
                if end - new_fn[0] > merge_dur * 16e3:
                    total_merged += 1
                    new_flist.append(new_fn)
                    if end - start > merge_dur * 16e3:
                        end = int(start + merge_dur * 16e3)
                    new_fn = [start, end, fns, label]
                else:
                    new_fn[1] = end
                    new_fn[2].extend(fns)

            new_flist.append(new_fn)
            # print('new_flist[:5]:', new_flist[:5])
            new_data_dict[id_date].extend(new_flist)
            
        return new_data_dict
