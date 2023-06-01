#!/usr/bin/env python

import os
import time
import traceback
from typing import Any, Dict

import accelerate

import ray
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig
from raydp.torch.config import TorchConfig
from ray.air import RunConfig, FailureConfig

import plugin


def main():
    config = plugin.Config()
    agentenv = plugin.get_agentenv(config.get("agentenv"))

    pass

if __name__ == "__main__":
    main()

