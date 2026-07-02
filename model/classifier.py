import os
os.environ['TORCH_HOME'] = '../../pretrained_models'
import math
from pathlib import Path

import timm
from timm.models.layers import to_2tuple,trunc_normal_

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import torchvision
from torch.cuda.amp import autocast
from torch import Tensor
from torchvision.models.feature_extraction import create_feature_extractor
from typing import List


def fast_cosine_dist(source_feats: Tensor, matching_pool: Tensor, device: str = 'cpu') -> Tensor:
    """ Like torch.cdist, but fixed dim=-1 and for cosine distance."""
    source_norms = torch.norm(source_feats, p=2, dim=-1).to(device)
    matching_norms = torch.norm(matching_pool, p=2, dim=-1)
    dotprod = -torch.cdist(source_feats[None].to(device), matching_pool[None], p=2)[0]**2 + source_norms[:, None]**2 + matching_norms[None]**2
    dotprod /= 2

    dists = 1 - ( dotprod / (source_norms[:, None] * matching_norms[None]) )
    return dists


class MLPAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # First MLP layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Second MLP layer
        self.fc2 = nn.Linear(hidden_dim, 1)
        # Softmax for attention weights
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query_key, value):
        hidden = F.relu(self.fc1(query_key))
        attention_scores = self.fc2(hidden).squeeze(-1)
        # (batch size, n tgt spk)
        attention_weights = self.softmax(attention_scores)
        
        # (batch size, seq len, n tgt spk, feat dim) -> (batch size, seq_len, feat dim)
        output = torch.matmul(
            attention_weights[:, None], value
        ).squeeze(-2)

        return output


class EfficientNetClassifier(nn.Module):
    def __init__(self, label_dim=1, pretrain=True, model_id=0):
        super().__init__()
        b = int(model_id)
        print('now train a effnet-b{:d} model'.format(b))
        if b == 7:
            self.model = torchvision.models.efficientnet_b7(pretrained=pretrain)
        elif b == 6:
            self.model = torchvision.models.efficientnet_b6(pretrained=pretrain)
        elif b == 5:
            self.model = torchvision.models.efficientnet_b5(pretrained=pretrain)
        elif b == 4:
            self.model = torchvision.models.efficientnet_b4(pretrained=pretrain)
        elif b == 3:
            self.model = torchvision.models.efficientnet_b3(pretrained=pretrain)
        elif b == 2:
            self.model = torchvision.models.efficientnet_b2(pretrained=pretrain)
        elif b == 1:
            self.model = torchvision.models.efficientnet_b1(pretrained=pretrain)
        elif b == 0:
            self.model = torchvision.models.efficientnet_b0(pretrained=pretrain)

        new_proj = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        print('conv1 get from pretrained model.')
        new_proj.weight = torch.nn.Parameter(torch.sum(self.model.features[0][0].weight, dim=1).unsqueeze(1))
        new_proj.bias = self.model.features[0][0].bias
        self.model.features[0][0] = new_proj
        self.model = create_feature_extractor(self.model, {'features.8': 'mout'})
        hid_dim = [1280, 1280, 1408, 1536, 1792, 2048, 2304, 2560]
        self.cla = torch.nn.Sequential(nn.LayerNorm(hid_dim[int(model_id)]), nn.Linear(hid_dim[int(model_id)], label_dim))

    def forward(self, x, valid_len):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        x = self.model(x)['mout']
        for i in range(x.shape[0]):
            x[i, :, :, 0] = torch.mean(x[i, :, :, :math.ceil(valid_len[i] / 33)], dim=-1)
        x = x[:, :, :, 0]
        x = torch.mean(x, dim=[2])
        out = self.cla(x)
        return out
    

class ECAPASpeakerVerifier(nn.Module):
    def __init__(self, device):
        super().__init__()
        from speechbrain.inference import EncoderClassifier
        self.classifier = EncoderClassifier.from_hparams(
            source='speechbrain/spkrec-ecapa-voxceleb', run_opts={"device": device},
        )
        self.classifier.hparams.label_encoder.ignore_len()

    # Function to extract MFCC features from audio files
    def forward(self, y1, y2):
        if y1.ndim == 1:
            y1 = y1.unsqueeze(0)
        if y2.ndim == 1:
            y2 = y2.unsqueeze(0)

        with torch.no_grad():
            embed1 = self.classifier.encode_batch(y1)
            embed2 = self.classifier.encode_batch(y2)
            similarity = F.cosine_similarity(embed1, embed2, dim=-1)
        
        return similarity

    def embed(self, x):
        with torch.no_grad():
            if x.ndim == 1:
                x = x.unsqueeze(0)
            embed = self.classifier.encode_batch(x)
        return embed


class ECAPAMLPClassifier(nn.Module):
    def __init__(self, device, embed_dim=1280, depth=2):
        super().__init__()
        from speechbrain.inference import EncoderClassifier
        self.enc = EncoderClassifier.from_hparams(
            source='speechbrain/spkrec-ecapa-voxceleb', run_opts={"device": device},
        )
        self.enc.hparams.label_encoder.ignore_len()
        dims = [192]+[embed_dim]*depth
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(out_dim))
        layers.append(nn.Linear(embed_dim, 1))
        self.mlp = nn.Sequential(*layers)
        self.mlp.to(device) 

    def forward(self, x, size):
        with torch.no_grad():
            embed = self.enc.encode_batch(x)
        logit = self.mlp(embed).squeeze(-1)
        return logit


class WhisperReader(object):
    def __init__(self, model_name, download_root='/data/sls/scratch/limingw/models/whisper', device='cuda'):
        import whisper
        self.model = whisper.load_model(
            model_name, download_root=download_root).to(device)

        spec_map = {'small':[768, list(range(12))], 'medium':[1024, list(range(24))], 'large':[1280, list(range(32))]}
        self.feat_dim = spec_map[model_name][0]
        self.layers = spec_map[model_name][1]

    def read_audio(self, fname, sr=16e3):
        import whisper
        if isinstance(fname, str):
            audio = whisper.load_audio(fname)
        else:
            audio = fname
        audio_len = audio.shape[-1] // 320
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        return mel, audio_len

    def normalize(self, w):
        w = w.strip()
        w = ''.join([c for c in w if c.isalpha()])
        return w

    def extract_features(self, loc):
        '''Return encoder features, word-level labels and segmentations'''
        mel, audio_len = self.read_audio(loc)

        layer_results = {}
        def get_layer_results(name):
            def hook(model, input, output):
                layer_results[name] = output.detach()
            return hook

        for l in self.layers:
            self.model.encoder.blocks[l].register_forward_hook(get_layer_results(f'block_{l}'))
    
        with torch.no_grad():
            if mel.ndim == 2:
                mel = mel[None]
            _ = self.model.encoder(mel)
            feats = []
            for l in self.layers:
                feat = layer_results[f'block_{l}'].contiguous() #XXX.squeeze(0)
                feats.append(feat[:, :audio_len])

        return torch.stack(feats, dim=-1)


class TemporalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, size=None):
        """
        x: [B, T, D]
        mask: [B, T] (optional, True for valid positions)
        """
        B, T, D = x.shape

        h = self.proj(x)    # [B, T, D]
        scores = torch.einsum("btd,d->bt", h, self.query)   # [B, T]

        if size is not None:
            device = x.device
            mask = (
                torch.arange(T, device=device)
                .unsqueeze(0)
                .expand(B, T)
                < size.unsqueeze(1)
            )  # [B, T], True = valid 
            scores = scores.masked_fill(~mask, -1e9)

        attn = torch.softmax(scores, dim=-1)
        pooled = torch.einsum("bt,btd->bd", attn, x)
        return pooled


class WhisperMLPClassifier(nn.Module):
    def __init__(self, model_name, embed_dim=1280, depth=2, device='cuda', num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.whisper = WhisperReader(model_name, device=device)
        self.layers = self.whisper.layers

        self.weighted_sum = nn.Linear(len(self.layers), 1)

        self.temporal_attn = TemporalAttention(self.whisper.feat_dim)

        dims = [self.whisper.feat_dim]+[embed_dim]*depth
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(out_dim))
        
        if num_classes <= 2:
            layers.append(nn.Linear(embed_dim, 1))
        else:
            layers.append(nn.Linear(embed_dim, num_classes))
        self.classifier = nn.Sequential(*layers)      

    def forward(self, x, size):
        if self.whisper is not None:
            feats = self.whisper.extract_features(x)
            # expected: [B, T, L, D]
        else:
            feats = x

        if feats.ndim == 4:
            # layer attention first
            feats = self.weighted_sum(feats).squeeze(-1)       
    
        # temporal attention instead of mean pooling
        feats = self.temporal_attn(feats, size=size)  # [B, D]
        
        logits = self.classifier(feats)
        if self.num_classes == 1:
            return logits.squeeze(-1)
        return logits


'''
class DemenbaClassifier(nn.Module):
    """
    """
    def __init__(self, args):
        ckpt_path = args.ckpt_path
        self.classifier = DASS(label_dim=args.n_dementia_class, imagenet_pretrain=True,
            audionet_pretrain=True, model_size='medium', verbose=True)

from .vmamba import VSSM
class DASS(nn.Module):
    """
    The MAST model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
    """
    def __init__(self, label_dim=527, imagenet_pretrain=True,
                  audioset_pretrain=False, model_size='medium', verbose=True):

        super(DASS, self).__init__()
        if verbose == True:
            print('---------------DASS Model Summary---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(imagenet_pretrain),str(audioset_pretrain)))
        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
        if (1): #audioset_pretrain == False:
            if model_size == 'small':
                self.v = VSSM(drop_path_rate=0.2,embed_dim=96,depths=[2, 2, 8, 2],
                           ssm_d_state=1,ssm_ratio=1.0,ssm_conv_bias=False,
                           forward_type='v05_noz',downsample_version="v3",patchembed_version="v2",
                           norm_layer="LN2D",num_classes=1000)

                if imagenet_pretrain:
                    checkpoint = torch.hub.load_state_dict_from_url(
                    url='https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm1_tiny_0230s_ckpt_epoch_264.pth',
                    map_location="cpu", check_hash=True
                    )
                    self.v.load_state_dict(checkpoint["model"],strict=True)

            elif model_size == 'medium': 
                self.v = VSSM(drop_path_rate=0.3,embed_dim=96,depths=[2, 2, 20, 2],
                           ssm_d_state=1,ssm_ratio=1.0,ssm_conv_bias=False,
                           forward_type='v05_noz',downsample_version="v3",patchembed_version="v2",
                           norm_layer="LN2D",num_classes=1000)
                
                if imagenet_pretrain:
                    checkpoint = torch.hub.load_state_dict_from_url(
                    url='https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm1_small_0229s_ckpt_epoch_240.pth',
                    map_location="cpu", check_hash=True
                    )
                    self.v.load_state_dict(checkpoint["model"],strict=True)
            else:
                raise Exception('Model size must be one of small, middle.')

            if verbose == True:
                print(self.v)

            self.v.classifier.head = nn.Linear(self.v.classifier.head.in_features, label_dim)

            new_proj = torch.nn.Conv2d(1, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            if imagenet_pretrain == True:
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed[0].weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed[0].bias
            
            self.v.patch_embed[0] = new_proj
            del new_proj
        if audioset_pretrain == True:
            #if not isinstance(audio_model, nn.DataParallel):
            #    audio_model = nn.DataParallel(audio_model)
            if model_size == 'small':
                checkpoint_path = '../../pretrained_models/DASS_small.pth'
                checkpoint = torch.load(checkpoint_path)
                mod_checkpoint = {}
                for k,v in checkpoint.items():
                    if 'classifier.head' in k:
                        continue
                    mod_checkpoint[k.replace('module.v.','')] = v
                self.v.load_state_dict(mod_checkpoint,strict=False)
                
            elif model_size == 'medium':
                checkpoint_path = '../../pretrained_models/DASS_medium.pth'
                checkpoint = torch.load(checkpoint_path)
                mod_checkpoint = {}
                for k,v in checkpoint.items():
                    if 'classifier.head' in k:
                        continue
                    mod_checkpoint[k.replace('module.v.','')] = v
                self.v.load_state_dict(mod_checkpoint,strict=False)

    def assign_devices(self, ngpu):
        self.v.assign_devices(ngpu)

    @autocast()
    def forward(self, x):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: prediction
        """
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        x = self.v(x)
        return x
'''
