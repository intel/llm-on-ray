from ray.train.torch.config import _TorchBackend
from ray.train.torch.config import TorchConfig as RayTorchConfig
from ray.train._internal.worker_group import WorkerGroup
from dataclasses import dataclass


@dataclass
class TorchConfig(RayTorchConfig):
    @property
    def backend_cls(self):
        return EnableHabanaBackend


def habana_import():
    try:
        pass
    except ImportError as habana_not_exist:
        raise ImportError("Please install habana_frameworks") from habana_not_exist


class EnableHabanaBackend(_TorchBackend):
    def on_start(self, worker_group: WorkerGroup, backend_config: RayTorchConfig):
        for i in range(len(worker_group)):
            worker_group.execute_single_async(i, habana_import)
        super().on_start(worker_group, backend_config)
