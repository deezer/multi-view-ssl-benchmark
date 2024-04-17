import logging
import os
from typing import Any, List, Tuple

import gin  # type: ignore
import GPUtil  # type: ignore
import numpy as np
import tensorflow as tf  # type: ignore
import torch
import torch.distributed as dist
from torch import nn


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx: Any, *grads: Any) -> Any:
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def find_free_gpus_manually() -> List[int]:
    """Not used! replaced by GPUtil"""
    os.system("nvidia-smi -q -d PIDs |grep Processes > tmp_free_gpu")
    memory_available = [True if len(x.split()) > 1 else False for x in open("tmp_free_gpu", "r").readlines()]
    os.system("rm tmp_free_gpu")
    return [int(i) for i in np.where(memory_available)[0]]


@gin.configurable  # type: ignore
def set_gpus(max_memory: float = 0.05) -> str:
    print("TORCH version: {}".format(torch.__version__))
    print("TF version: {}".format(tf.__version__))

    #  enable CuDNN benchmark
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    gpu_index = GPUtil.getAvailable(limit=4, maxMemory=max_memory)
    # setting gpu for tensorflow
    try:
        gpu_index = [gpu_index[0]]
    except:
        raise ValueError("No GPU available!!")
    print("\t Using GPUs: {}".format(gpu_index))
    # physical_devices = tf.config.list_physical_devices("GPU")
    # tf.config.set_visible_devices([physical_devices[gpu_index[0]]], "GPU")
    # tf.config.experimental.set_memory_growth(
    #     physical_devices[gpu_index[0]], True
    # )  # limit tf memory usage
    tf.config.set_visible_devices([], "GPU")
    device = "cuda:{}".format(",".join([str(i) for i in gpu_index]))
    return device


def set_cpu() -> str:
    print("TORCH version: {}".format(torch.__version__))
    print("TF version: {}".format(tf.__version__))
    print("\t Using only CPU")
    tf.config.set_visible_devices([], "GPU")
    return "cpu"


@gin.configurable  # type: ignore
def fix_random_seeds(seed: int = 31, extra: int = 0) -> None:
    """
    Fix random seeds.
    """
    seed += extra
    print("|---- SEED {} ----|".format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    return


def is_dist_avail_and_initialized() -> bool:
    output = True
    if not dist.is_available():
        output = False
    if not dist.is_initialized():
        output = False
    return output


def stop_process(condition: bool, device: Any) -> bool:
    flag_tensor = torch.zeros(1).to(device)
    stop = False
    if condition:
        # Conditions for breaking the loop
        flag_tensor += 1
    dist.all_reduce(flag_tensor, op=dist.ReduceOp.SUM)
    if flag_tensor != 0:
        print("Training stopped")
        stop = True
    return stop


def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    output = 0
    if is_dist_avail_and_initialized():
        output = dist.get_rank()
    return output


def is_main_process() -> bool:
    return get_rank() == 0


def cleanup() -> None:
    dist.destroy_process_group()


def setup_for_distributed(is_master: bool) -> None:
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args: Any, **kwargs: Any) -> None:
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)
        return

    __builtin__.print = print
    return


def init_distributed_mode(world_size: int, rank: int) -> None:
    """
    dist_url: url used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html
    limit_gpus: number of gpus to use
    """
    assert world_size >= 1
    assert torch.distributed.is_nccl_available()
    dist_url = "env://"
    # prevent tensorflow to use gpu
    tf.config.set_visible_devices([], "GPU")
    torch.cuda.set_device(rank)
    # use nccl as backend since it is currently the fastest and highly recommended backend when using GPUs
    dist.init_process_group(
        backend="nccl",
        init_method=dist_url,
        world_size=world_size,
        rank=rank,
    )
    print("| distributed init (rank {}): {}".format(rank, dist_url), flush=True)
    # dist.barrier()  # type: ignore
    setup_for_distributed(rank == 0)
    return


def define_world(limit_gpus: int) -> int:
    gpus = GPUtil.getAvailable(limit=limit_gpus, maxMemory=0.15)
    assert torch.distributed.is_available()
    assert len(gpus) == limit_gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in gpus])
    world_size = len(gpus)

    #  enable CuDNN benchmark
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    logging.info("Starting multi-gpu process:")
    logging.info("Using {} as GPUS \n".format(gpus))
    return world_size


def model_to_distributed(model: Any, device: int) -> Any:
    has_batch_norm = False
    bn_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
        nn.LayerNorm,
        nn.GroupNorm,
    )
    for _, module in model.named_modules():
        if isinstance(module, bn_types):
            has_batch_norm = True
    if has_batch_norm:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  # type: ignore
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device)
    return model
