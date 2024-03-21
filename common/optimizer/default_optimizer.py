import torch  # noqa: F401
from .optimizer import Optimizer


class DefaultOptimizer(Optimizer):
    def __call__(self, model, config):
        return self.create_optimizer(model)
        optimizer_name = config.get("name", "SGD")
        optimizer_config = config.get("config", {})
        optimizer_type = eval("torch.optim.%s" % (optimizer_name))

        optimizer_grouped_parameters = self.get_grouped_parameters(model, config)
        optimizer = optimizer_type(optimizer_grouped_parameters, **optimizer_config)

        return optimizer

    def get_grouped_parameters(self, model, config):
        return model.parameters()

    def get_decay_parameter_names(self, model):
        from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
        from transformers.trainer_pt_utils import get_parameter_names
        decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        return decay_parameters

    def create_optimizer(self, model):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        decay_parameters = self.get_decay_parameter_names(model)

        optimizer_grouped_parameters = []
        weight_decay = 0.0
        for t_params, t_weight_decay in zip(
            [
                [p for n, p in model.named_parameters() if n in decay_parameters and p.requires_grad],
                [p for n, p in model.named_parameters() if n not in decay_parameters and p.requires_grad],
            ],
            [weight_decay, 0.0],
        ):
            # Empty groups of parameters are filtered because they make FusedAdamW crash
            if t_params:
                optimizer_grouped_parameters.append(
                    {
                        "params": t_params,
                        "weight_decay": t_weight_decay,
                    }
                )

        try:
            from habana_frameworks.torch.hpex.optimizers import FusedAdamW
        except ImportError as error:
            error.msg = (
                f"Could not import 'FusedAdamW' from 'habana_frameworks.torch.hpex.optimizers'. {error.msg}."
            )
            raise error
        optimizer_cls = FusedAdamW
        optimizer_kwargs = {'lr': 1e-05, 'betas': (0.9, 0.999), 'eps': 1e-08}
        # print(f">>>>>>>>>>>>>>>>>>>>>>> optimizer_cls = {optimizer_cls}, optimizer_grouped_parameters = {optimizer_grouped_parameters}, len(optimizer_grouped_parameters) = {len(optimizer_grouped_parameters)}, optimizer_kwargs = {optimizer_kwargs}")

        return optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
