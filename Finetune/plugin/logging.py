import sys
import logging
import logging.config
import traceback
import functools

from .config import Config

__all__ = ["logger", "get_logger"]

config = Config()

logging_config_file = config.get("logging_config_file", None)
if logging_config_file is not None:
    try:
        logging.config.fileConfig(logging_config_file)
    except Exception as e:
        traceback.print_exc()
        exit(1)
    

logging_config = config.get("logging_config", None)
if logging_config is not None:
    try:
        logging.config.dictConfig(logging_config)
    except Exception as e:
        traceback.print_exc()
        exit(1)

logger_name = config.get("logger_name", "root")
if config.get("use_accelerate_log", False):
    import accelerate
    get_logger = functools.partial(accelerate.logging.get_logger, name=logger_name)
else:
    get_logger = functools.partial(logging.getLogger, name=logger_name)

logger = get_logger()