from ray.train.torch.config import _TorchBackend
from ray.train.torch.config import TorchConfig as RayTorchConfig
from ray.train._internal.worker_group import WorkerGroup
from dataclasses import dataclass
from typing import Optional
import os
import sys

# The package importlib_metadata is in a different place, depending on the Python version.
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


@dataclass
class TorchConfig(RayTorchConfig):
    device: Optional[str] = None

    @property
    def backend_cls(self):
        EnableCCLBackend.device = self.device
        return EnableCCLBackend


def libs_import():
    """try to import IPEX and oneCCL."""
    try:
        import intel_extension_for_pytorch  # noqa: F401
    except ImportError:
        raise ImportError("Please install intel_extension_for_pytorch")
    try:
        ccl_version = importlib_metadata.version("oneccl_bind_pt")
        if ccl_version >= "1.12":
            import oneccl_bindings_for_pytorch  # noqa: F401
        else:
            import torch_ccl  # noqa: F401
    except ImportError as ccl_not_exist:
        raise ImportError("Please install torch-ccl") from ccl_not_exist


def _set_torch_distributed_env_vars(device):
    if device is not None:
        os.environ["ACCELERATE_TORCH_DEVICE"] = device


class EnableCCLBackend(_TorchBackend):
    device: Optional[str] = None

    def on_start(self, worker_group: WorkerGroup, backend_config: RayTorchConfig):
        for i in range(len(worker_group)):
            worker_group.execute_single_async(i, libs_import)
        super().on_start(worker_group, backend_config)

    def on_training_start(self, worker_group: WorkerGroup, backend_config: RayTorchConfig):
        super().on_training_start(worker_group, backend_config)
        worker_group.execute(_set_torch_distributed_env_vars, self.device)
