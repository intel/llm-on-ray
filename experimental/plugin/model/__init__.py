import os
from .model import Model
from ..common import import_all_module

realpath = os.path.realpath(__file__)
basedir = os.path.dirname(realpath)
import_all_module(basedir, "plugin.model")

__all__ = ["Model"]
