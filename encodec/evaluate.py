# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adiyoss
import random

import wandb
import argparse
import logging
import sys
import torch.nn.functional as F

from .data import CleanSet
from .enhance import add_flags, get_pred
from . import distrib, pretrained

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    "denoiser.evaluate", description="Speech enhancement using Demucs - Evaluate model performance"
)
add_flags(parser)
parser.add_argument("--data_dir", help="directory including noisy.json and clean.json files")
parser.add_argument("--matching", default="sort", help="set this to dns for the dns dataset.")
parser.add_argument(
    "-v", "--verbose", action="store_const", const=logging.DEBUG, default=logging.INFO, help="More loggging"
)


def evaluate(args, model=None, data_loader=None, dset_name="test"):
    # Load model
    if not model:
        model = pretrained.get_model(args).to(args.device)
    model.eval()

    # Load data
    if data_loader is None:
        dataset = CleanSet(args.data_dir, matching=args.matching, sample_rate=model.sample_rate)
        data_loader = distrib.loader(dataset, batch_size=1, num_workers=2)

    dataset = data_loader.dataset
    for i, example in enumerate([dataset[0], dataset[1]]):
        example = example[None, :]
        example = example.to(args.device)
        if args.device == "cpu":
            _estimate_and_run_metrics(y=example, model=model, args=args, epoch=data_loader.epoch, example_ind=i, dset_name=dset_name)
        else:
            y_pred = get_pred(model, example, args)
            y_pred = y_pred.cpu()
            y = example.cpu()
            _run_metrics(y=y, y_pred=y_pred, args=args, epoch=data_loader.epoch, sr=model.sample_rate, example_ind=i, dset_name=dset_name)


def _estimate_and_run_metrics(y, model, args, epoch, example_ind, dset_name):
    y_pred = get_pred(model, y, args)
    return _run_metrics(y, y_pred, args, sr=model.sample_rate, epoch=epoch, example_ind=example_ind, dset_name=dset_name)


def _run_metrics(y, y_pred, args, sr, epoch, example_ind, dset_name):
    l1_loss = F.l1_loss(y_pred, y)
    y_pred = y_pred.numpy()[:, 0]
    y = y.numpy()[:, 0]

    if args.wandb:
        assert epoch, "epoch must not be None"
        wandb.log(
            {f"{dset_name}_prediction_{example_ind}": wandb.Audio(y_pred.flatten(), caption=f"{dset_name}_prediction", sample_rate=sr)},
            step=epoch,
        )
        wandb.log({f"target_{example_ind}": wandb.Audio(y.flatten(), caption="target", sample_rate=sr)}, step=epoch)
        wandb.log({f"{dset_name}_{example_ind} L1_loss": l1_loss}, step=epoch)


def main():
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stderr, level=args.verbose)
    logger.debug(args)
    evaluate(args)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
