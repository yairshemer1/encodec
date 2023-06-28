# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

import logging
import os
import typing as tp
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Subset
from torch.nn.parallel.distributed import DistributedDataParallel

logger = logging.getLogger(__name__)


def rank():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0


def is_distributed():
    return world_size() > 1


def world_size():
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return 1


def init(args):
    """init.

    Initialize DDP using the given rendezvous file.
    """
    # if args.ddp:
    #     assert args.rank is not None and args.world_size is not None
    #     rank = args.rank
    #     world_size = args.world_size
    if world_size() == 1:
        return
    torch.cuda.set_device(rank())
    torch.distributed.init_process_group(
        backend=args.ddp_backend,
        init_method="file://" + os.path.abspath(args.rendezvous_file),
        world_size=world_size(),
        rank=rank(),
    )
    logger.debug("Distributed rendezvous went well, rank %d/%d", rank(), world_size())


def average(metrics, count=1.0):
    """average.

    Average all the relevant metrices across processes
    `metrics`should be a 1D float32 vector. Returns the average of `metrics`
    over all hosts. You can use `count` to control the weight of each worker.
    """
    if world_size() == 1:
        return metrics
    tensor = torch.tensor(list(metrics) + [1], device="cuda", dtype=torch.float32)
    tensor *= count
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return (tensor[:-1] / tensor[-1]).cpu().numpy().tolist()


def average_metrics(metrics: tp.Dict[str, float], count=1.0):
    """Average a dictionary of metrics across all workers, using the optional
    `count` as unnormalized weight.
    """
    if not is_distributed():
        return metrics
    keys, values = zip(*metrics.items())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor = torch.tensor(list(values) + [1], device=device, dtype=torch.float32)
    tensor *= count
    all_reduce(tensor)
    averaged = (tensor[:-1] / tensor[-1]).cpu().tolist()
    return dict(zip(keys, averaged))


def all_reduce(tensor: torch.Tensor, op=torch.distributed.ReduceOp.SUM):
    if is_distributed():
        return torch.distributed.all_reduce(tensor, op)


def wrap(model):
    """wrap.

    Wrap a model with DDP if distributed training is enabled.
    """
    if world_size() == 1:
        return model
    else:
        return DistributedDataParallel(
            model, device_ids=[torch.cuda.current_device()], output_device=torch.cuda.current_device()
        )


def barrier():
    if world_size() > 1:
        torch.distributed.barrier()


def loader(dataset, *args, shuffle=False, klass=DataLoader, **kwargs):
    """loader.

    Create a dataloader properly in case of distributed training.
    If a gradient is going to be computed you must set `shuffle=True`.

    :param dataset: the dataset to be parallelized
    :param args: relevant args for the loader
    :param shuffle: shuffle examples
    :param klass: loader class
    :param kwargs: relevant args
    """

    if world_size() == 1:
        return klass(dataset, *args, shuffle=shuffle, **kwargs)

    if shuffle:
        # train means we will compute backward, we use DistributedSampler
        sampler = DistributedSampler(dataset)
        # We ignore shuffle, DistributedSampler already shuffles
        return klass(dataset, *args, **kwargs, sampler=sampler)
    else:
        # We make a manual shard, as DistributedSampler otherwise replicate some examples
        dataset = Subset(dataset, list(range(rank(), len(dataset), world_size())))
        return klass(dataset, *args, shuffle=shuffle)


def _is_complex_or_float(tensor):
    return torch.is_floating_point(tensor) or torch.is_complex(tensor)


def _check_number_of_params(params: tp.List[torch.Tensor]):
    # utility function to check that the number of params in all workers is the same,
    # and thus avoid a deadlock with distributed all reduce.
    if not is_distributed() or not params:
        return
    tensor = torch.tensor([len(params)], device=params[0].device, dtype=torch.long)
    all_reduce(tensor)
    if tensor.item() != len(params) * world_size():
        # If not all the workers have the same number, for at least one of them,
        # this inequality will be verified.
        raise RuntimeError(
            f"Mismatch in number of params: ours is {len(params)}, " "at least one worker has a different one."
        )


def broadcast_tensors(tensors: tp.Iterable[torch.Tensor], src: int = 0):
    """Broadcast the tensors from the given parameters to all workers.
    This can be used to ensure that all workers have the same model to start with.
    """
    if not is_distributed():
        return
    tensors = [tensor for tensor in tensors if _is_complex_or_float(tensor)]
    _check_number_of_params(tensors)
    handles = []
    for tensor in tensors:
        handle = torch.distributed.broadcast(tensor.data, src=src, async_op=True)
        handles.append(handle)
    for handle in handles:
        handle.wait()


def sync_buffer(buffers, average=True):
    """
    Sync grad for buffers. If average is False, broadcast instead of averaging.
    """
    if not is_distributed():
        return
    handles = []
    for buffer in buffers:
        if torch.is_floating_point(buffer.data):
            if average:
                handle = torch.distributed.all_reduce(buffer.data, op=torch.distributed.ReduceOp.SUM, async_op=True)
            else:
                handle = torch.distributed.broadcast(buffer.data, src=0, async_op=True)
            handles.append((buffer, handle))
    for buffer, handle in handles:
        handle.wait()
        if average:
            buffer.data /= world_size


def sync_grad(params):
    """
    Simpler alternative to DistributedDataParallel, that doesn't rely
    on any black magic. For simple models it can also be as fast.
    Just call this on your model parameters after the call to backward!
    """
    if not is_distributed():
        return
    handles = []
    for p in params:
        if p.grad is not None:
            handle = torch.distributed.all_reduce(p.grad.data, op=torch.distributed.ReduceOp.SUM, async_op=True)
            handles.append((p, handle))
    for p, handle in handles:
        handle.wait()
        p.grad.data /= world_size()
