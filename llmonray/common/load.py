import sys
from typing import Any, Dict

from .logging import logger
from . import dataset
from . import tokenizer
from . import model
from . import optimizer
from . import trainer
from . import initializer


def load_check_decorator(func):
    def wrapper(*args, **kwargs):
        try:
            logger.info(f"{func.__name__} start")
            ret = func(*args, **kwargs)
        except Exception as e:
            logger.critical(f"{func.__name__}: {e}", exc_info=True)
            exit(1)
        logger.info(f"{func.__name__} finish")
        if ret is None:
            logger.critical(f"{func.__name__} has wrong return type")
            exit(1)
        else:
            return ret

    return wrapper


@load_check_decorator
def load_dataset(config: Dict[str, Any]):
    logger.info(f"{sys._getframe().f_code.co_name} config: {config}")
    datasets_type = config.get("type", None)
    Factory = dataset.Dataset.registory.get(datasets_type)
    if Factory is None:
        raise ValueError(f"there is no {datasets_type} dataset.")
    else:
        try:
            _ = Factory()(config)
        except Exception as e:
            logger.critical(f"{Factory.__name__} call error: {e}", exc_info=True)
            exit(1)
        return _


@load_check_decorator
def load_tokenizer(config: Dict[str, Any]):
    logger.info(f"{sys._getframe().f_code.co_name} config: {config}")
    tokenizer_type = config.get("type", None)
    Factory = tokenizer.Tokenizer.registory.get(tokenizer_type)
    if Factory is None:
        raise ValueError(f"there is no {tokenizer_type} tokenizer.")
    else:
        try:
            _ = Factory()(config)
        except Exception as e:
            logger.critical(f"{Factory.__name__} call error: {e}", exc_info=True)
            exit(1)
        return _


@load_check_decorator
def load_model(config: Dict[str, Any]):
    logger.info(f"{sys._getframe().f_code.co_name} config: {config}")
    model_type = config.get("type", None)
    Factory = model.Model.registory.get(model_type)
    if Factory is None:
        raise ValueError(f"there is no {model_type} model.")
    else:
        try:
            _ = Factory()(config)
        except Exception as e:
            logger.critical(f"{Factory.__name__} call error: {e}", exc_info=True)
            exit(1)
        return _


@load_check_decorator
def load_optimizer(model, config: Dict[str, Any]):
    logger.info(f"{sys._getframe().f_code.co_name} config: {config}")
    optimizer_type = config.get("type", None)
    Factory = optimizer.Optimizer.registory.get(optimizer_type)
    if Factory is None:
        raise ValueError(f"there is no {optimizer_type} optimizer.")
    else:
        try:
            _ = Factory()(model, config)
        except Exception as e:
            logger.critical(f"{Factory.__name__} call error: {e}", exc_info=True)
            exit(1)
        return _


@load_check_decorator
def get_trainer(config: Dict[str, Any]):
    logger.info(f"{sys._getframe().f_code.co_name} config: {config}")
    trainer_type = config.get("type", None)
    Factory = trainer.Trainer.registory.get(trainer_type)
    if Factory is None:
        raise ValueError(f"there is no {trainer_type} trainer.")
    try:
        _ = Factory(config)
    except Exception as e:
        logger.critical(f"{Factory.__name__} init error: {e}", exc_info=True)
        exit(1)
    return _


@load_check_decorator
def get_initializer(config: Dict[str, Any]):
    logger.info(f"{sys._getframe().f_code.co_name} config: {config}")
    initializer_type = config.get("type", None)
    Factory = initializer.Initializer.registory.get(initializer_type)
    if Factory is None:
        raise ValueError(f"there is no {initializer_type} initializer.")
    try:
        _ = Factory(config)
    except Exception as e:
        logger.critical(f"{Factory.__name__} init error: {e}", exc_info=True)
        exit(1)
    return _
