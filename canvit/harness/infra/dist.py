"""DDP utilities. Single source of truth for rank, world size, and device.

When SLURM env vars are absent, every helper returns rank=0 / world_size=1 and
`is_dist()=False`, so single-GPU code paths are unaffected.

Activation requires both `WORLD_SIZE` and `SLURM_PROCID` to be set, plus
`WORLD_SIZE > 1`. This matches `plan_dataloading/ddp/minimal_distributed_training.sh`.
"""

from __future__ import annotations

import atexit
import logging
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import Tensor, nn

log = logging.getLogger(__name__)

_initialized = False
_rank = 0
_world_size = 1
_local_rank = 0
_device = torch.device("cpu")


def init_dist() -> None:
    """Initialise the process group from SLURM env vars. Idempotent.

    No-op when WORLD_SIZE is unset or equals 1. Otherwise initialises NCCL,
    sets the CUDA device to local_rank, and caches rank/world_size globally.
    """
    global _initialized, _rank, _world_size, _local_rank, _device
    if _initialized:
        return
    _initialized = True

    world_size_env = os.environ.get("WORLD_SIZE")
    if world_size_env is None or int(world_size_env) <= 1:
        if torch.cuda.is_available():
            _device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            _device = torch.device("mps")
        else:
            _device = torch.device("cpu")
        log.info(f"DDP: not initialised (WORLD_SIZE={world_size_env}); single-process mode, device={_device}")
        return

    _world_size = int(world_size_env)
    _rank = int(os.environ["SLURM_PROCID"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    _local_rank = _rank - gpus_per_node * (_rank // gpus_per_node)

    torch.cuda.set_device(_local_rank)
    _device = torch.device("cuda", _local_rank)

    log.info(
        f"DDP init: rank={_rank}/{_world_size}, local_rank={_local_rank}, "
        f"device={_device}, MASTER_ADDR={os.environ.get('MASTER_ADDR')}, "
        f"MASTER_PORT={os.environ.get('MASTER_PORT')}"
    )
    # device_id pins this rank's NCCL communicator to a specific device,
    # avoiding PyTorch's "Guessing device ID based on global rank" warning
    # and the heterogeneous-mapping hang it warns about. Our rank → device
    # mapping is already established by torch.cuda.set_device(_local_rank)
    # above, so the explicit binding matches what PyTorch would have guessed.
    # timeout: validation runs on rank 0 only while the other ranks wait at a barrier, and
    # NCCL's default of 10 min aborts that wait. A multi-retina validation (one pass per
    # retina; ~4-7 min estimated for exp38's 9) comes close, so allow 15 min.
    dist.init_process_group(backend="nccl", rank=_rank, world_size=_world_size, device_id=_device,
                            timeout=timedelta(minutes=15))

    # Register cleanup at interpreter shutdown to silence PyTorch's
    # "destroy_process_group() was not called before program exit" warning.
    # atexit runs after all DDP collectives have completed, so it's safe; the
    # is_initialized() guard makes it idempotent if destroy is called manually.
    atexit.register(lambda: dist.destroy_process_group() if dist.is_initialized() else None)


def is_dist() -> bool:
    return _world_size > 1


def is_main() -> bool:
    return _rank == 0


def rank() -> int:
    return _rank


def world_size() -> int:
    return _world_size


def local_rank() -> int:
    return _local_rank


def device() -> torch.device:
    return _device


def barrier() -> None:
    if is_dist():
        dist.barrier()


def broadcast_module_buffers(module: nn.Module, src: int = 0) -> None:
    """Broadcast all buffers and parameters of a module from src rank to all others.

    Used after rank-0 initialises standardizer state from a single shard, to make
    sure every rank holds identical normalizer stats before the first step.
    """
    if not is_dist():
        return
    for tensor in list(module.buffers()) + list(module.parameters()):
        dist.broadcast(tensor.data, src=src)


def all_reduce_mean(t: Tensor) -> Tensor:
    """All-reduce mean across ranks. Returns input unchanged in single-process mode."""
    if not is_dist():
        return t
    out = t.detach().clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    out /= _world_size
    return out
