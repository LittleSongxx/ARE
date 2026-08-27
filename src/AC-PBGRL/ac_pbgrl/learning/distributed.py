from __future__ import annotations

import contextlib
import math
import os
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class BatchSchedule:
    global_batch_size: int
    world_size: int
    rank: int
    micro_batch_size: int

    @property
    def local_samples(self) -> int:
        base, remainder = divmod(self.global_batch_size, self.world_size)
        return base + int(self.rank < remainder)

    @property
    def chunk_sizes(self) -> list[int]:
        maximum_local = math.ceil(self.global_batch_size / self.world_size)
        chunks = math.ceil(maximum_local / self.micro_batch_size)
        remaining = self.local_samples
        result = []
        for _ in range(chunks):
            size = min(self.micro_batch_size, remaining)
            result.append(size)
            remaining -= size
        return result

    def gradient_scale(self, chunk_size: int) -> float:
        # DDP averages rank gradients. This restores a true global sample mean even
        # for uneven partitions such as 43/43/42 with world_size=3.
        return float(chunk_size * self.world_size) / float(self.global_batch_size)


@dataclass
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    initialized: bool

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.initialized:
            dist.barrier()

    def broadcast_object(self, value, source: int = 0):
        if not self.initialized:
            return value
        values = [value]
        dist.broadcast_object_list(values, src=source)
        return values[0]

    def all_reduce_gradients(self, modules: Iterable[nn.Module], extra_parameters: Iterable[torch.Tensor] = ()) -> None:
        """Average explicitly accumulated gradients across ranks.

        AC-PBGRL accumulates an RL minibatch and an independent privileged-label
        minibatch in one optimizer update. Keeping every backward pass inside
        ``DDP.no_sync`` and reducing once here avoids an accidental extra sync or
        a different effective auxiliary weight when the world size changes.
        """

        if not self.initialized:
            return
        seen: set[int] = set()
        parameters = []
        for module in modules:
            for parameter in module.parameters():
                if parameter.grad is not None and id(parameter) not in seen:
                    parameters.append(parameter)
                    seen.add(id(parameter))
        for parameter in extra_parameters:
            if parameter.grad is not None and id(parameter) not in seen:
                parameters.append(parameter)
                seen.add(id(parameter))
        for parameter in parameters:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(self.world_size)

    def weighted_metric_mean(self, value: float, local_weight: int) -> float:
        if not self.initialized:
            return float(value)
        pair = torch.tensor(
            [float(value) * float(local_weight), float(local_weight)],
            dtype=torch.float64,
            device=self.device,
        )
        dist.all_reduce(pair, op=dist.ReduceOp.SUM)
        return float((pair[0] / pair[1].clamp_min(1.0)).item())

    def assert_parameters_synchronized(
        self,
        modules: Iterable[nn.Module],
        extra_parameters: Iterable[torch.Tensor] = (),
        tolerance: float = 1.0e-6,
    ) -> None:
        if not self.initialized:
            return
        values = []
        for module in modules:
            values.extend(parameter.detach() for parameter in module.parameters())
        values.extend(parameter.detach() for parameter in extra_parameters)
        checksum = torch.zeros(2, dtype=torch.float64, device=self.device)
        for index, value in enumerate(values, start=1):
            flattened = value.double().reshape(-1)
            checksum[0] += flattened.sum() * index
            checksum[1] += flattened.square().sum() * index
        lower, upper = checksum.clone(), checksum.clone()
        dist.all_reduce(lower, op=dist.ReduceOp.MIN)
        dist.all_reduce(upper, op=dist.ReduceOp.MAX)
        scale = upper.abs().clamp_min(1.0)
        if torch.any((upper - lower).abs() > float(tolerance) * scale):
            raise RuntimeError(f"DDP parameters diverged across ranks: min={lower.tolist()} max={upper.tolist()}")

    def close(self) -> None:
        if self.initialized and dist.is_initialized():
            dist.destroy_process_group()


def initialize_distributed(device_preference: str = "auto") -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = device_preference != "cpu" and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    initialized = world_size > 1
    if initialized and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if use_cuda else "gloo", init_method="env://")
    return DistributedContext(rank, local_rank, world_size, device, initialized)


def wrap_ddp(module: nn.Module, context: DistributedContext) -> nn.Module:
    module = module.to(context.device)
    if not context.initialized:
        return module
    kwargs = {"broadcast_buffers": False, "find_unused_parameters": True}
    if context.device.type == "cuda":
        kwargs.update(device_ids=[context.local_rank], output_device=context.local_rank)
    return DistributedDataParallel(module, **kwargs)


@contextlib.contextmanager
def maybe_no_sync(modules: Iterable[nn.Module], enabled: bool):
    if not enabled:
        yield
        return
    with contextlib.ExitStack() as stack:
        for module in modules:
            no_sync = getattr(module, "no_sync", None)
            if no_sync is not None:
                stack.enter_context(no_sync())
        yield


def unwrap(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, DistributedDataParallel) else module
