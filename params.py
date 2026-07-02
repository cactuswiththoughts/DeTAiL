# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

# data parameters
n_mels = 80
n_feat = 1024
sampling_rate = 16000
n_fft = 1280
hop_size = 320
f_max = 8000

# wavlm knn-based encoder parameters
# synthesis_layer = 6
# matching_layer = 6
synthesis_layer = 18
matching_layer = 18

# diffusion-based decoder parameters
dec_dim = 256
spk_dim = 192
use_ref_t = True
beta_min = 0.05
beta_max = 20.0

# training parameters
seed = 37
test_size = 5
train_frames = 64
