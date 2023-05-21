# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adiyoss
import wandb
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import logging
import sys
import numpy as np
from pesq import pesq
from pystoi import stoi
import torch

from .data import CleanSet
from .enhance import add_flags, get_estimate
from . import distrib, pretrained
from .utils import bold, LogProgress

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
        'denoiser.evaluate',
        description='Speech enhancement using Demucs - Evaluate model performance')
add_flags(parser)
parser.add_argument('--data_dir', help='directory including noisy.json and clean.json files')
parser.add_argument('--matching', default="sort", help='set this to dns for the dns dataset.')
parser.add_argument('--no_pesq', action="store_false", dest="pesq", default=True,
                    help="Don't compute PESQ.")
parser.add_argument('-v', '--verbose', action='store_const', const=logging.DEBUG,
                    default=logging.INFO, help="More loggging")


def evaluate(args, model=None, data_loader=None):
    total_pesq = 0
    total_stoi = 0
    total_cnt = 0
    updates = 5

    # Load model
    if not model:
        model = pretrained.get_model(args).to(args.device)
    model.eval()

    # Load data
    if data_loader is None:
        dataset = CleanSet(args.data_dir,
                                matching=args.matching, sample_rate=model.sample_rate)
        data_loader = distrib.loader(dataset, batch_size=1, num_workers=2)
    pendings = []
    with ProcessPoolExecutor(args.num_workers) as pool:
        with torch.no_grad():
            iterator = LogProgress(logger, data_loader, name="Eval estimates")
            for i, data in enumerate(iterator):
                # Get batch data
                clean = data.to(args.device)
                # If device is CPU, we do parallel evaluation in each CPU worker.
                if args.device == 'cpu':
                    pendings.append(
                        pool.submit(_estimate_and_run_metrics, clean, model, args, data_loader.epoch))
                else:
                    estimate = get_estimate(model, clean, args)
                    estimate = estimate.cpu()
                    clean = clean.cpu()
                    pendings.append(
                        pool.submit(_run_metrics, clean, estimate, args, model.sample_rate, data_loader.epoch))
                total_cnt += clean.shape[0]

        for pending in LogProgress(logger, pendings, updates, name="Eval metrics"):
            pending.result()


def _estimate_and_run_metrics(clean, model, args, epoch):
    estimate = get_estimate(model, clean, args)
    return _run_metrics(clean, estimate, args, sr=model.sample_rate, epoch=epoch)


def _run_metrics(clean, estimate, args, sr, epoch):
    estimate = estimate.numpy()[:, 0]
    clean = clean.numpy()[:, 0]
    if args.wandb:
        assert epoch, "epoch must not be None"
        wandb.log({"estimated": wandb.Audio(estimate.flatten(), caption="estimated", sample_rate=sr)}, step=epoch)
        wandb.log({"target": wandb.Audio(clean.flatten(), caption="target", sample_rate=sr)}, step=epoch)


def get_pesq(ref_sig, out_sig, sr):
    """Calculate PESQ.
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        PESQ
    """
    pesq_val = 0
    for i in range(len(ref_sig)):
        pesq_val += pesq(sr, ref_sig[i], out_sig[i], 'wb')
    return pesq_val


def get_stoi(ref_sig, out_sig, sr):
    """Calculate STOI.
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        STOI
    """
    stoi_val = 0
    for i in range(len(ref_sig)):
        stoi_val += stoi(ref_sig[i], out_sig[i], sr, extended=False)
    return stoi_val


def main():
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stderr, level=args.verbose)
    logger.debug(args)
    pesq, stoi = evaluate(args)
    json.dump({'pesq': pesq, 'stoi': stoi}, sys.stdout)
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
