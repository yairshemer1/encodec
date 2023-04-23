#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# authors: adiyoss and adefossez

import logging
import os
import wandb
import hydra

from encodec.executor import start_ddp_workers
from encodec.model import MultiScaleDiscriminator

logger = logging.getLogger(__name__)


def run(args):
    import torch

    from encodec import distrib
    from encodec.data import CleanSet
    from encodec.model import EncodecModel
    from encodec.solver import Solver
    import encodec.modules as m
    distrib.init(args)

    # torch also initialize cuda seed if available
    torch.manual_seed(args.seed)

    # model = Demucs(**args.demucs, sample_rate=args.sample_rate)
    encoder = m.SEANetEncoder(channels=1, norm='weight_norm', causal=True)
    decoder = m.SEANetDecoder(channels=1, norm='weight_norm', causal=True)
    model = EncodecModel(
            encoder,
            decoder,
            normalize=False,
            segment=None,
        )
    msd = MultiScaleDiscriminator()

    if args.show:
        logger.info(model)
        mb = sum(p.numel() for p in model.parameters()) * 4 / 2**20
        logger.info('Size: %.1f MB', mb)
        if hasattr(model, 'valid_length'):
            field = model.valid_length(1)
            logger.info('Field: %.1f ms', field / args.sample_rate * 1000)
        return

    assert args.batch_size % distrib.world_size == 0
    args.batch_size //= distrib.world_size

    length = int(args.segment * args.sample_rate)
    stride = int(args.stride * args.sample_rate)
    # Demucs requires a specific number of samples to avoid 0 padding during training
    if hasattr(model, 'valid_length'):
        length = model.valid_length(length)
    kwargs = {"matching": args.dset.matching, "sample_rate": args.sample_rate, "convert": args.convert}
    # Building datasets and loaders
    tr_dataset = CleanSet(
        args.dset.train, length=length, stride=stride, pad=args.pad, **kwargs)
    tr_loader = distrib.loader(
        tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    if args.dset.valid:
        cv_dataset = CleanSet(args.dset.valid, **kwargs)
        cv_loader = distrib.loader(cv_dataset, batch_size=1, num_workers=args.num_workers)
    else:
        cv_loader = None
    if args.dset.test:
        tt_dataset = CleanSet(args.dset.test, **kwargs)
        tt_loader = distrib.loader(tt_dataset, batch_size=1, num_workers=args.num_workers)
    else:
        tt_loader = None
    data = {"tr_loader": tr_loader, "cv_loader": cv_loader, "tt_loader": tt_loader}

    if torch.cuda.is_available():
        model.cuda()

    # optimizer
    if args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, args.beta2))
        optimizer_msd = torch.optim.Adam(msd.parameters(), lr=args.lr, betas=(0.9, args.beta2))
    else:
        logger.fatal('Invalid optimizer %s', args.optim)
        os._exit(1)

    # init WandB
    if args.wandb:
        from datetime import datetime
        name = datetime.strftime(datetime.now(), format="%Y%m%d%H%M%S")
        wandb.init(project="encodec-reconstruct", name=name)

    # Construct Solver
    solver = Solver(data=data, model=model, msd=msd, optimizer=optimizer, optimizer_msd=optimizer_msd, args=args)
    solver.train()
    if args.wandb:
        wandb.finish()


def _main(args):
    global __file__
    # Updating paths in config
    for key, value in args.dset.items():
        if isinstance(value, str) and key not in ["matching"]:
            args.dset[key] = hydra.utils.to_absolute_path(value)
    __file__ = hydra.utils.to_absolute_path(__file__)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("denoise").setLevel(logging.DEBUG)

    logger.info("For logs, checkpoints and samples check %s", os.getcwd())
    logger.debug(args)
    if args.ddp and args.rank is None:
        start_ddp_workers(args)
    else:
        run(args)


@hydra.main(config_path="conf/config.yaml")
def main(args):
    try:
        _main(args)
    except Exception:
        logger.exception("Some error happened")
        # Hydra intercepts exit code, fixed in beta but I could not get the beta to work
        os._exit(1)


if __name__ == "__main__":
    main()
