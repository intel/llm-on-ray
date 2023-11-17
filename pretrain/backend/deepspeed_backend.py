import logging
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import deepspeed
import torch.distributed as dist


import ray
from ray.train.torch.config import _TorchBackend
from ray.train.torch.config import TorchConfig as RayTorchConfig
from ray.train._internal.worker_group import WorkerGroup
from ray.train._internal.utils import get_address_and_port
from ray.train.constants import DEFAULT_NCCL_SOCKET_IFNAME
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TorchConfig(RayTorchConfig):

    @property
    def backend_cls(self):
        return DeepSpeedBackend

def _set_nccl_network_interface():
    """Set the appropriate NCCL network interface to use."""

    if "NCCL_SOCKET_IFNAME" not in os.environ:
        logger.debug(
            f"Setting NCCL_SOCKET_IFNAME to {DEFAULT_NCCL_SOCKET_IFNAME} "
            f"to prioritize ethernet connection. To override this behavior, set the "
            f"`NCCL_SOCKET_IFNAME` environment variable in your Ray runtime "
            "environment: "
            "`ray.init(runtime_env={{'env_vars': {'NCCL_SOCKET_IFNAME': 'ens5'}}}`"
        )
        os.environ["NCCL_SOCKET_IFNAME"] = DEFAULT_NCCL_SOCKET_IFNAME

def _setup_deepspeed_process_group(
    backend: str,
    world_rank: int,
    world_size: int,
    init_method: str,
    timeout_s: int = 1800,
):
    """Connects the distributed Deepspeed backend.

    Args:
        backend: The backend (nccl, gloo, etc.) to use for training.
        world_rank: Rank of the current worker.
        world_size: Number of workers participating in the job.
        init_method: URL specifying how to initialize the process group.
        timeout_s: Seconds for process group operations to timeout.
    """
    if world_rank == 0:
        logger.info(
            f"Setting up process group for: {init_method} [rank={world_rank}, "
            f"world_size={world_size}]"
        )
    else:
        logger.debug(
            f"Setting up process group for: {init_method} [rank={world_rank}, "
            f"world_size={world_size}]"
        )
    logger.debug(f"using {backend}")

    # See the `timeout` arg in https://pytorch.org/docs/master/
    # distributed.html#torch.distributed.init_process_group for description of
    # NCCL_ASYNC_ERROR_HANDLING. We do not use NCCL_BLOCKING_WAIT due to performance
    # overhead.
    if (
        backend == "nccl"
        and "NCCL_ASYNC_ERROR_HANDLING" not in os.environ
        and "NCCL_BLOCKING_WAIT" not in os.environ
    ):
        logger.debug(
            "Setting NCCL_ASYNC_ERROR_HANDLING to fail if NCCL collective "
            "communication operations are timing out. "
            "To override this behavior, you can set NCCL_ASYNC_ERROR_HANDLING=0."
        )
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"


    deepspeed.init_distributed(
        dist_backend=backend,
        auto_mpi_discovery=False,
        init_method=init_method,
        rank=world_rank,
        world_size=world_size,
        timeout=timedelta(seconds=timeout_s),
    )

class DeepSpeedBackend(_TorchBackend):

    def on_start(self, worker_group: WorkerGroup, backend_config: TorchConfig):
        if dist.is_available():
            # Set the appropriate training backend.
            if backend_config.backend is None:
                if worker_group.num_gpus_per_worker > 0:
                    backend = "nccl"
                else:
                    backend = "gloo"
            else:
                backend = backend_config.backend

            if backend == "nccl":
                worker_group.execute(_set_nccl_network_interface)

            master_addr, master_port = worker_group.execute_single(
                0, get_address_and_port
            )
            if backend_config.init_method == "env":

                def set_env_vars(addr, port):
                    os.environ["MASTER_ADDR"] = addr
                    os.environ["MASTER_PORT"] = str(port)

                worker_group.execute(set_env_vars, addr=master_addr, port=master_port)
                url = "env://"
            elif backend_config.init_method == "tcp":
                url = f"tcp://{master_addr}:{master_port}"
            else:
                raise ValueError(
                    f"The provided init_method ("
                    f"{backend_config.init_method}) is not supported. Must "
                    f"be either 'env' or 'tcp'."
                )

            setup_futures = []
            for i in range(len(worker_group)):
                setup_futures.append(
                    worker_group.execute_single_async(
                        i,
                        _setup_deepspeed_process_group,
                        backend=backend,
                        world_rank=i,
                        world_size=len(worker_group),
                        init_method=url,
                        timeout_s=backend_config.timeout_s,
                    )
                )
            ray.get(setup_futures)
        else:
            raise RuntimeError("Distributed torch is not available.")


