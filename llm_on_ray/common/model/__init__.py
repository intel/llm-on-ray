import os
from llm_on_ray.common.model.model import Model
from llm_on_ray.common.common import import_all_module

realpath = os.path.realpath(__file__)
basedir = os.path.dirname(realpath)
import_all_module(basedir, "llm_on_ray.common.model")

__all__ = ["Model"]
