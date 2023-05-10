from .optimizer import Optimizer
import transformers
import torch

class DefaultOptimizer(Optimizer):
    def __call__(self, model, config):
        no_decay = ["bias", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.1,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        lr = config.get("lr", 1e-5)
        #optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr)

        return optimizer