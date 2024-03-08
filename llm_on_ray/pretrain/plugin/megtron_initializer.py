from megatron.initialize import initialize_megatron
from llm_on_ray.common.initializer import Initializer
from llm_on_ray.common.logging import logger


class MegatronInitializer(Initializer):
    def __init__(self, config):
        self.config = config
        self.args = {}

    def init(self):
        # self._parse_arguments(ARGUMENTS_SCHEMA, config)
        args = None
        if "megatron_config" in self.config:
            args = self.config["megatron_config"]
            initialize_megatron(ignore_unknown_args=True, external_args=args, allow_no_cuda=True)
        else:
            logger.error("cannot initialize the megatron without the megatron_config")
