from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset
from megatron.initialize import initialize_megatron
from common.initializer import Initializer
from common.logging import logger


class MegatronInitializer(Initializer):
    def __init__(self, config):
        self.config = config
        self.args = {}

    def init(self):
        #self._parse_arguments(ARGUMENTS_SCHEMA, config)
        args = None
        if "megatron_config" in self.config :
            args = self.config["megatron_config"] 
            initialize_megatron(ignore_unknown_args=True, args_defaults=args, allow_no_cuda=True)
        else:
            logger.error("cannot initialize the megatron without the megatron_config")