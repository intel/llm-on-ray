from .logging import logger
from .load import *  # noqa: F403 # unable to detect undefined names
from . import agentenv
from typing import Dict, Any
import sys


@load_check_decorator  # noqa: F405 # may be undefined, or defined from star imports
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
