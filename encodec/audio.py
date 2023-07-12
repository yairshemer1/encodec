# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

from collections import namedtuple
import json
from pathlib import Path
import math
import os
import random
import sys
import librosa
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool

import torchaudio
from torch.nn import functional as F
import sys

sys.path.append("/content/drive/My Drive/AudioLab/encodec")
from encodec.dsp import convert_audio

Info = namedtuple("Info", ["length", "sample_rate", "channels"])


def get_info(path):
    info = torchaudio.info(path)
    if hasattr(info, "num_frames"):
        # new version of torchaudio
        return Info(info.num_frames, info.sample_rate, info.num_channels)
    else:
        siginfo = info[0]
        return Info(siginfo.length // siginfo.channels, siginfo.rate, siginfo.channels)


def find_audio_files(path, exts=[".wav"], progress=True):
    audio_files = []
    for root, folders, files in os.walk(path, followlinks=True):
        for file in files:
            file = Path(root) / file
            if file.suffix.lower() in exts:
                audio_files.append(str(file.resolve()))
    meta = []
    for idx, file in enumerate(audio_files):
        info = get_info(file)
        meta.append((file, info.length))
        if progress:
            print(format((1 + idx) / len(audio_files), " 3.1%"), end="\r", file=sys.stderr)
    meta.sort()
    return meta


class Audioset:
    def __init__(self, files=None, sample_rate=16_000, segment_len=32_000, saved_path=None):
        self.index_file = []
        self.segment_len = segment_len
        files = [file for file, length in files]
        load_ = partial(Audioset._load_sample_meta, sample_rate=sample_rate, segment_len=segment_len)
        with Pool() as p:
            self.index_file = list(tqdm(p.imap(load_, files), total=len(files)))
        self.index_file = [x for x in self.index_file if x is not None]

    @staticmethod
    def _load_sample_meta(f_path, sample_rate, segment_len):
        sr, dur = librosa.get_samplerate(f_path), librosa.get_duration(filename=f_path)
        assert sr == sample_rate
        if int(sr * dur) > segment_len:
            return f_path, int(sr * dur)

    def __len__(self):
        return len(self.index_file)

    def __getitem__(self, index):
        path, len_wav = self.index_file[index]
        offset = random.randint(0, len_wav - self.segment_len)
        wav, _ = torchaudio.load(path, frame_offset=offset, num_frames=self.segment_len)
        return wav


if __name__ == "__main__":
    meta = []
    for path in sys.argv[1:]:
        meta += find_audio_files(path)
    json.dump(meta, sys.stdout, indent=4)
