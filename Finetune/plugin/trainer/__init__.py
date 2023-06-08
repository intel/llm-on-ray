import os
from .trainer import Trainer
from ..common import import_all_module

realpath = os.path.realpath(__file__)
basedir = os.path.dirname(realpath)
import_all_module(basedir, "plugin.trainer")

__all__ =  ["Trainer"]
