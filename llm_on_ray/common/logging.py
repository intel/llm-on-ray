import logging
import logging.config
import traceback
import functools

__all__ = ["logger", "get_logger"]

use_accelerate_log = False
logger_name = "common"

logging_config = {
    "version": 1,
    "loggers": {
        "root": {"level": "INFO", "handlers": ["consoleHandler"]},
        "common": {
            "level": "INFO",
            "handlers": ["consoleHandler"],
            "qualname": "common",
            "propagate": 0,
        },
    },
    "handlers": {
        "consoleHandler": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standardFormatter",
        },
    },
    "formatters": {
        "standardFormatter": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "",
        }
    },
}

if logging_config is not None:
    try:
        logging.config.dictConfig(logging_config)
    except Exception:
        traceback.print_exc()
        print("pass here........................")
        exit(1)

if use_accelerate_log:
    import accelerate

    get_logger = functools.partial(accelerate.logging.get_logger, name=logger_name)
else:
    get_logger = functools.partial(logging.getLogger, name=logger_name)

logger = get_logger()
