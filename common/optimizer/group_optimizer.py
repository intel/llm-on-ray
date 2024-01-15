import torch  # noqa: F401
from .optimizer import Optimizer


class GroupOptimizer(Optimizer):
    def __call__(self, model, config):
        optimizer_name = config.get("name", "SGD")
        optimizer_config = config.get("config", {})
        optimizer_type = eval("torch.optim.%s" % (optimizer_name))

        optimizer_grouped_parameters = self.get_grouped_parameters(model, config)
        optimizer = optimizer_type(optimizer_grouped_parameters, **optimizer_config)

        return optimizer

    def get_grouped_parameters(self, model, config):
        no_decay = ["bias", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.1,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters
