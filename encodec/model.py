# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

import math
import time
import typing as tp

import numpy as np
import torch
import torchaudio
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from encodec.modules import NormConv2d
from . import modules as m
from . import quantization as qt
from .resample import downsample2, upsample2
from .utils import _linear_overlap_add, _get_checkpoint_url
from .utils import capture_init

EncodedFrame = tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]

ROOT_URL = "https://dl.fbaipublicfiles.com/encodec/v0/"
FeatureMapType = tp.List[torch.Tensor]
LogitsType = torch.Tensor
DiscriminatorOutput = tp.Tuple[tp.List[LogitsType], tp.List[FeatureMapType]]


def get_2d_padding(kernel_size: tp.Tuple[int, int], dilation: tp.Tuple[int, int] = (1, 1)):
    return (((kernel_size[0] - 1) * dilation[0]) // 2, ((kernel_size[1] - 1) * dilation[1]) // 2)


class LMModel(nn.Module):
    """Language Model to estimate probabilities of each codebook entry.
    We predict all codebooks in parallel for a given time step.

    Args:
        n_q (int): number of codebooks.
        card (int): codebook cardinality.
        dim (int): transformer dimension.
        **kwargs: passed to `encodec.modules.transformer.StreamingTransformerEncoder`.
    """

    def __init__(self, n_q: int = 32, card: int = 1024, dim: int = 200, **kwargs):
        super().__init__()
        self.card = card
        self.n_q = n_q
        self.dim = dim
        self.transformer = m.StreamingTransformerEncoder(dim=dim, **kwargs)
        self.emb = nn.ModuleList([nn.Embedding(card + 1, dim) for _ in range(n_q)])
        self.linears = nn.ModuleList([nn.Linear(dim, card) for _ in range(n_q)])

    def forward(self, indices: torch.Tensor, states: tp.Optional[tp.List[torch.Tensor]] = None, offset: int = 0):
        """
        Args:
            indices (torch.Tensor): indices from the previous time step. Indices
                should be 1 + actual index in the codebook. The value 0 is reserved for
                when the index is missing (i.e. first time step). Shape should be
                `[B, n_q, T]`.
            states: state for the streaming decoding.
            offset: offset of the current time step.

        Returns a 3-tuple `(probabilities, new_states, new_offset)` with probabilities
        with a shape `[B, card, n_q, T]`.

        """
        B, K, T = indices.shape
        input_ = sum([self.emb[k](indices[:, k]) for k in range(K)])
        out, states, offset = self.transformer(input_, states, offset)
        logits = torch.stack([self.linears[k](out) for k in range(K)], dim=1).permute(0, 3, 1, 2)
        return torch.softmax(logits, dim=1), states, offset


def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference) ** 0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)


def fast_conv(conv, x):
    """
    Faster convolution evaluation if either kernel size is 1
    or length of sequence is 1.
    """
    batch, chin, length = x.shape
    chout, chin, kernel = conv.weight.shape
    assert batch == 1
    if kernel == 1:
        x = x.view(chin, length)
        out = torch.addmm(conv.bias.view(-1, 1), conv.weight.view(chout, chin), x)
    elif length == kernel:
        x = x.view(chin * kernel, 1)
        out = torch.addmm(conv.bias.view(-1, 1), conv.weight.view(chout, chin * kernel), x)
    else:
        out = conv(x)
    return out.view(batch, chout, -1)


class EncodecModel(nn.Module):
    """EnCodec model operating on the raw waveform.
    Args:
        target_bandwidths (list of float): Target bandwidths.
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.
        sample_rate (int): Audio sample rate.
        channels (int): Number of audio channels.
        normalize (bool): Whether to apply audio normalization.
        segment (float or None): segment duration in sec. when doing overlap-add.
        overlap (float): overlap between segment, given as a fraction of the segment duration.
        name (str): name of the model, used as metadata when compressing audio.
    """

    @capture_init
    def __init__(
        self,
        encoder: m.SEANetEncoder,
        decoder: m.SEANetDecoder,
        quantizer: qt.ResidualVectorQuantizer,
        target_bandwidths: tp.List[float],
        sample_rate: int,
        channels: int,
        normalize: bool = False,
        segment: tp.Optional[float] = None,
        overlap: float = 0.01,
        name: str = "unset",
    ):
        super().__init__()
        self.bandwidth: tp.Optional[float] = None
        self.target_bandwidths = target_bandwidths
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder
        self.sample_rate = sample_rate
        self.channels = channels
        self.normalize = normalize
        self.segment = segment
        self.overlap = overlap
        self.frame_rate = math.ceil(self.sample_rate / np.prod(self.encoder.ratios))
        self.name = name
        self.bits_per_codebook = int(math.log2(self.quantizer.bins))
        assert 2 ** self.bits_per_codebook == self.quantizer.bins, \
            "quantizer bins must be a power of 2."

    @property
    def segment_length(self) -> tp.Optional[int]:
        if self.segment is None:
            return None
        return int(self.segment * self.sample_rate)

    @property
    def segment_stride(self) -> tp.Optional[int]:
        segment_length = self.segment_length
        if segment_length is None:
            return None
        return max(1, int((1 - self.overlap) * segment_length))

    def encode(self, x: torch.Tensor) -> tp.List[EncodedFrame]:
        """Given a tensor `x`, returns a list of frames containing
        the discrete encoded codes for `x`, along with rescaling factors
        for each segment, when `self.normalize` is True.

        Each frames is a tuple `(codebook, scale)`, with `codebook` of
        shape `[B, K, T]`, with `K` the number of codebooks.
        """
        assert x.dim() == 3
        _, channels, length = x.shape
        assert 0 < channels <= 2
        segment_length = self.segment_length
        if segment_length is None:
            segment_length = length
            stride = length
        else:
            stride = self.segment_stride  # type: ignore
            assert stride is not None

        encoded_frames: tp.List[EncodedFrame] = []
        for offset in range(0, length, stride):
            frame = x[:, :, offset : offset + segment_length]
            encoded_frames.append(self._encode_frame(frame))
        return encoded_frames

    def _encode_frame(self, x: torch.Tensor) -> EncodedFrame:
        x, scale = self.preprocess(x)

        emb = self.encoder(x)
        codes = self.quantizer.encode(emb, self.frame_rate, self.bandwidth)

        codes = codes.transpose(0, 1)
        # codes is [B, K, T], with T frames, K nb of codebooks.
        return codes, scale

    def decode(self, encoded_frames: tp.List[EncodedFrame]) -> torch.Tensor:
        """Decode the given frames into a waveform.
        Note that the output might be a bit bigger than the input. In that case,
        any extra steps at the end can be trimmed.
        """
        segment_length = self.segment_length
        if segment_length is None:
            assert len(encoded_frames) == 1
            return self._decode_frame(encoded_frames[0])

        frames = [self._decode_frame(frame) for frame in encoded_frames]
        return _linear_overlap_add(frames, self.segment_stride or 1)

    def _decode_frame(self, encoded_frame: EncodedFrame) -> torch.Tensor:
        codes, scale = encoded_frame
        codes = codes.transpose(0, 1)
        emb = self.quantizer.decode(codes)
        out = self.decoder(emb)
        self.postprocess(out, scale)
        return out

    def forward(self, x: torch.Tensor):
        length = x.shape[-1]
        duration = length / self.sample_rate
        assert self.segment is None or duration <= 1e-5 + self.segment

        if self.normalize:
            mono = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.view(-1, 1)
        else:
            scale = None

        emb = self.encoder(x)
        out = self.quantizer.forward(emb, self.frame_rate, self.bandwidth)
        pred = self.decoder(out.quantized)[:, :, :x.shape[-1]]
        if scale is not None:
            pred = pred * scale.view(-1, 1, 1)
        return pred, out.penalty

    def postprocess(self, x: torch.Tensor, scale: tp.Optional[torch.Tensor] = None) -> torch.Tensor:
        if scale is not None:
            assert self.normalize
            x = x * scale.view(-1, 1, 1)
        return x

    def preprocess(self, x):
        length = x.shape[-1]
        duration = length / self.sample_rate
        assert self.segment is None or duration <= 1e-5 + self.segment

        if self.normalize:
            mono = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.view(-1, 1)
        else:
            scale = None
        return x, scale

    def set_target_bandwidth(self, bandwidth: float):
        if bandwidth not in self.target_bandwidths:
            raise ValueError(
                f"This model doesn't support the bandwidth {bandwidth}. " f"Select one of {self.target_bandwidths}."
            )
        self.bandwidth = bandwidth

    def get_lm_model(self) -> LMModel:
        """Return the associated LM model to improve the compression rate."""
        device = next(self.parameters()).device
        lm = LMModel(
            self.quantizer.n_q, self.quantizer.bins, num_layers=5, dim=200, past_context=int(3.5 * self.frame_rate)
        ).to(device)
        checkpoints = {
            "encodec_24khz": "encodec_lm_24khz-1608e3c0.th",
            "encodec_48khz": "encodec_lm_48khz-7add9fc3.th",
        }
        try:
            checkpoint_name = checkpoints[self.name]
        except KeyError:
            raise RuntimeError("No LM pre-trained for the current Encodec model.")
        url = _get_checkpoint_url(ROOT_URL, checkpoint_name)
        state = torch.hub.load_state_dict_from_url(url, map_location="cpu", check_hash=True)  # type: ignore
        lm.load_state_dict(state)
        lm.eval()
        return lm

    @staticmethod
    def _get_model(
        target_bandwidths: tp.List[float],
        sample_rate: int = 24_000,
        channels: int = 1,
        causal: bool = True,
        model_norm: str = "weight_norm",
        audio_normalize: bool = False,
        segment: tp.Optional[float] = None,
        name: str = "unset",
    ):
        encoder = m.SEANetEncoder(channels=channels, norm=model_norm, causal=causal)
        decoder = m.SEANetDecoder(channels=channels, norm=model_norm, causal=causal)
        n_q = int(1000 * target_bandwidths[-1] // (math.ceil(sample_rate / encoder.hop_length) * 10))
        quantizer = qt.ResidualVectorQuantizer(
            dimension=encoder.dimension,
            n_q=n_q,
            bins=1024,
        )
        model = EncodecModel(
            encoder,
            decoder,
            quantizer,
            target_bandwidths,
            sample_rate,
            channels,
            normalize=audio_normalize,
            segment=segment,
            name=name,
        )
        return model


class DiscriminatorS(torch.nn.Module):
    def __init__(
        self,
        filters: int,
        in_channels: int = 1,
        out_channels: int = 1,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        max_filters: int = 1024,
        filters_scale: int = 1,
        kernel_size: tp.Tuple[int, int] = (3, 9),
        dilations: tp.List = [1, 2, 4],
        stride: tp.Tuple[int, int] = (1, 2),
        normalized: bool = True,
        norm: str = "weight_norm",
        activation: str = "LeakyReLU",
        activation_params: dict = {"negative_slope": 0.2},
    ):
        super(DiscriminatorS, self).__init__()
        assert len(kernel_size) == 2
        assert len(stride) == 2
        self.filters = filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized
        self.activation = getattr(torch.nn, activation)(**activation_params)
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window_fn=torch.hann_window,
            normalized=self.normalized,
            center=False,
            pad_mode=None,
            power=None,
        )
        spec_channels = 2 * self.in_channels
        self.convs = nn.ModuleList()
        self.convs.append(
            NormConv2d(spec_channels, self.filters, kernel_size=kernel_size, padding=get_2d_padding(kernel_size))
        )
        in_chs = min(filters_scale * self.filters, max_filters)
        for i, dilation in enumerate(dilations):
            out_chs = min((filters_scale ** (i + 1)) * self.filters, max_filters)
            self.convs.append(
                NormConv2d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=(dilation, 1),
                    padding=get_2d_padding(kernel_size, (dilation, 1)),
                    norm=norm,
                )
            )
            in_chs = out_chs
        out_chs = min((filters_scale ** (len(dilations) + 1)) * self.filters, max_filters)
        self.convs.append(
            NormConv2d(
                in_chs,
                out_chs,
                kernel_size=(kernel_size[0], kernel_size[0]),
                padding=get_2d_padding((kernel_size[0], kernel_size[0])),
                norm=norm,
            )
        )
        self.conv_post = NormConv2d(
            out_chs,
            self.out_channels,
            kernel_size=(kernel_size[0], kernel_size[0]),
            padding=get_2d_padding((kernel_size[0], kernel_size[0])),
            norm=norm,
        )

    def forward(self, x: torch.Tensor):
        fmap = []
        z = self.spec_transform(x)  # [B, 2, Freq, Frames, 2]
        z = torch.cat([z.real, z.imag], dim=1)
        z = rearrange(z, "b c w t -> b c t w")
        for i, layer in enumerate(self.convs):
            z = layer(z)
            z = self.activation(z)
            fmap.append(z)
        z = self.conv_post(z)
        return z, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    @capture_init
    def __init__(
        self,
        filters: int,
        in_channels: int = 1,
        out_channels: int = 1,
        n_ffts: tp.List[int] = [1024, 2048, 512],
        hop_lengths: tp.List[int] = [256, 512, 128],
        win_lengths: tp.List[int] = [1024, 2048, 512],
    ):
        super(MultiScaleDiscriminator, self).__init__()
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorS(
                    filters,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    n_fft=n_ffts[i],
                    win_length=win_lengths[i],
                    hop_length=hop_lengths[i],
                )
                for i in range(len(n_ffts))
            ]
        )

    def forward(self, x: torch.Tensor) -> DiscriminatorOutput:
        logits = []
        fmaps = []
        for disc in self.discriminators:
            logit, fmap = disc(x)
            logits.append(logit)
            fmaps.append(fmap)
        return logits, fmaps


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss


def total_gen_loss(fmap_real, logits_fake, fmap_fake, device="cuda"):
    relu = torch.nn.ReLU()
    l1Loss = torch.nn.L1Loss(reduction="mean")
    l_g = torch.tensor([0.0], device=device, requires_grad=True)
    l_feat = torch.tensor([0.0], device=device, requires_grad=True)

    # generator loss and feat loss, D_k(\hat x) = logits_fake[k], D_k^l(x) = fmap_real[k][l], D_k^l(\hat x) = fmap_fake[k][l]
    for tt1 in range(len(fmap_real)):
        l_g = l_g + torch.mean(relu(1 - logits_fake[tt1])) / len(logits_fake)
        for tt2 in range(len(fmap_real[tt1])):
            l_feat = l_feat + l1Loss(fmap_real[tt1][tt2], fmap_fake[tt1][tt2]) / torch.mean(
                torch.abs(fmap_real[tt1][tt2])
            )

    KL_scale = len(fmap_real) * len(fmap_real[0])
    K_scale = len(fmap_real)

    l_g = 3 * l_g / K_scale
    l_feat = 3 * l_feat / KL_scale

    return l_g, l_feat
