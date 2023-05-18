import torch

from .optimizer import Optimizer

class DefaultOptimizer(Optimizer):
    def __call__(self, model, config):

        optimizer_name = config.get("name", "SGD")
        optimizer_config = config.get("config", {})
        optimizer_type = eval("torch.optim.%s"%(optimizer_name))

        optimizer_grouped_parameters = self.get_grouped_parameters(model, config)
        optimizer = optimizer_type(optimizer_grouped_parameters, **optimizer_config)

        return optimizer

    def get_grouped_parameters(self, model, config):
        return model.parameters()