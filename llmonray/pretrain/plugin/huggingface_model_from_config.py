import torch
import math
import transformers
from llmonray.common.model.model import Model


# for huggingface model weight random initialization
class HuggingFaceModelFromConfig(Model):
    def __call__(self, config):
        name = config.get("name")
        self.model_config = config.get("config", {})
        self.auto_config = None
        if name is not None:
            self.auto_config = transformers.AutoConfig.from_pretrained(
                pretrained_model_name_or_path=name, **self.model_config
            )
        else:
            self.auto_config = transformers.AutoConfig.for_model(**self.model_config)
        self.model = transformers.AutoModelForCausalLM.from_config(self.auto_config)

        if config.get("deepspeed_zero_stage", None) != 3:
            self.init_weights()

        return self.model

    def init_weights(self):
        if self.model_config.get("init_method", None):
            init_method = self.get_init_methods(self.model_config)
            self.recursive_initialization(self.model, init_method, init_method)

        return self.model

    def recursive_initialization(
        self,
        module,
        init_method_linear,
        init_method_embedding,
    ):
        if isinstance(module, torch.nn.Linear):
            init_method_linear(module.weight)
            if module.bias is not None:
                # torch.nn.init.zeros_(module.bias)
                pass
        elif isinstance(module, torch.nn.Embedding):
            init_method_embedding(module.weight)

        for child in module.children():
            self.recursive_initialization(child, init_method_linear, init_method_embedding)

    def get_init_methods(self, init_config):
        init_method = init_config.get("init_method")
        init_method_std = init_config.get("init_method_std", 0.02)

        num_layers = self.auto_config.num_hidden_layers
        hidden_size = self.auto_config.hidden_size

        def _get(name):
            if name == "normal":
                return init_method_normal(
                    init_method_std,
                )
            elif name == "scaled_normal":
                return scaled_init_method_normal(
                    init_method_std,
                    num_layers,
                )
            elif name == "xavier_uniform":
                return xavier_uniform_init_method()
            elif name == "xavier_normal":
                return xavier_normal_init_method()
            elif name == "wang_init":
                return wang_init_method(
                    num_layers,
                    hidden_size,
                )
            elif name == "small_init":
                return small_init_init_method(
                    hidden_size,
                )
            else:
                raise NotImplementedError(f"Unknown init method {name}")

        return _get(init_method)


def init_method_normal(sigma, use_mup_outer=False, mup_init_scale=1.0):
    """Init method based on N(0, sigma)."""

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method_normal(sigma, num_layers, use_mup_outer=False, mup_init_scale=1.0):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def xavier_uniform_init_method(use_mup_outer=False, mup_init_scale=1.0):
    """Fills the input Tensor with values according to the method described in Understanding the difficulty of
    training deep feedforward neural networks - Glorot, X. & Bengio, Y. (2010), using a uniform distribution.
    """

    def init_(tensor):
        return torch.nn.init.xavier_uniform_(tensor)

    return init_


def xavier_normal_init_method(use_mup_outer=False, mup_init_scale=1.0):
    """Fills the input Tensor with values according to the method described in Understanding the difficulty of
    training deep feedforward neural networks - Glorot, X. & Bengio, Y. (2010), using a normal distribution.
    """

    def init_(tensor):
        return torch.nn.init.xavier_normal_(tensor)

    return init_


def small_init_init_method(dim, use_mup_outer=False, mup_init_scale=1.0):
    """Fills the input Tensor with values according to the method described in Transformers without Tears: Improving
    the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2010), using a normal distribution.
    """
    std = math.sqrt(2 / (5 * dim))

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def wang_init_method(n_layers, dim, use_mup_outer=False, mup_init_scale=1.0):
    std = 2 / n_layers / math.sqrt(dim)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_
