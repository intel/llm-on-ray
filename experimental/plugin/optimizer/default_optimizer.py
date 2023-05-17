import torch

from .optimizer import Optimizer

class DefaultOptimizer(Optimizer):
    def __call__(self, model, config):

        optimizer_name = config.get("name", "SGD")
        optimizer_config = config.get("config", {})
        optimizer_type = eval("torch.optim.%s"%(optimizer_name))

        optimizer = optimizer_type(**optimizer_config)

        return optimizer
