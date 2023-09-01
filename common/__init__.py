from .config import Config, parse_args, parse_config
from .logging import get_logger, logger
from .load import *
from .init import init
from .common import import_all_module
from . import agentenv

@load_check_decorator
def get_agentenv(config: Dict[str, Any]):
    logger.info(f"{sys._getframe().f_code.co_name} config: {config}")
    agentenv_type = config.get("type", None)
    Factory = agentenv.AgentEnv.registory.get(agentenv_type)
    if Factory is None:
        raise ValueError(f"there is no {agentenv_type} AgentEnv.")
    try:
        _ = Factory(config)
    except Exception as e:
        logger.critical(f"{Factory.__name__} init error: {e}", exc_info=True)
        exit(1)
    return _