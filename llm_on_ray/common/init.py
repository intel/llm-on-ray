import torch
import accelerate

from llm_on_ray.common.logging import logger


def check_config(config):
    logger.debug("check config start")
    if isinstance(config, dict):
        return True
    else:
        return False


def init(config):
    logger.debug("global init start")
    if not check_config(config):
        logger.critical("config must be a dict")
        raise
    logger.debug("check config finish")

    torch_thread_num = config.get("torch_thread_num")
    if torch_thread_num is not None:
        torch.set_num_threads(torch_thread_num)
        logger.info(f"torch_thread_num is set {torch_thread_num}")
    else:
        logger.info("torch_thread_num is not set, all cpu will be used")

    seed = config.get("seed")
    if seed is not None:
        accelerate.utils.set_seed(seed)
        logger.info(f"seed is set {seed}")
    else:
        logger.info("seed is not set")

    logger.debug("global init finish")
