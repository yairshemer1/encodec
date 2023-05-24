# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Original copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""STFT-based Loss modules."""

import torch
import torchaudio


class MelLoss(torch.nn.Module):
    """Mel loss module."""

    def __init__(self, fft_size=1024, hop_length=120, win_length=600, n_mels=64, normalized=True):
        """Initialize Mel loss module."""
        super(MelLoss, self).__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(n_fft=fft_size,
                                                                  hop_length=hop_length,
                                                                  win_length=win_length,
                                                                  n_mels=n_mels,
                                                                  normalized=normalized)

        self.l1_loss = torch.nn.L1Loss()
        self.l2_loss = torch.nn.MSELoss()

    def forward(self, y_pred, y):
        """Calculate forward propagation.
        Args:
            y_pred (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: L1 loss.
            Tensor: L2 loss.
        """
        y_pred_spec = self.mel_transform(y_pred)
        y_spec = self.mel_transform(y)

        return self.l1_loss(y_pred_spec, y_spec), self.l2_loss(y_pred_spec, y_spec)


class MultiResolutionMelLoss(torch.nn.Module):
    """Multi resolution Mel loss module."""

    def __init__(self,
                 scales=[5, 6, 7, 8, 9, 10, 11]):
        """Initialize Multi resolution Mel loss module.
        Args:
            scales (list): List of scales of spectrogram.
        """
        super(MultiResolutionMelLoss, self).__init__()
        self.mel_losses = torch.nn.ModuleList()
        for scale in scales:
            hl = 2 ** scale
            wl = hl // 4
            fft = hl*2+1
            self.mel_losses += [MelLoss(hop_length=hl, win_length=wl, fft_size=fft)]

    def forward(self, y_pred, y):
        """Calculate forward propagation.
        Args:
            y_pred (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Total multi resolution loss value.
            Tensor: Multi resolution l1 loss value.
            Tensor: Multi resolution l2 loss value.
        """
        total_l1_loss = 0.0
        total_l2_loss = 0.0
        for f in self.mel_losses:
            l1_loss, l2_loss = f(y_pred, y)
            total_l1_loss += l1_loss
            total_l2_loss += l2_loss
        total_l1_loss /= len(self.mel_losses)
        total_l2_loss /= len(self.mel_losses)

        return total_l1_loss + total_l2_loss, total_l1_loss, total_l2_loss
