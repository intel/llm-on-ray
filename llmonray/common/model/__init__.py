import os
from llmonray.common.model.model import Model
from llmonray.common.common import import_all_module

realpath = os.path.realpath(__file__)
basedir = os.path.dirname(realpath)
import_all_module(basedir, "llmonray.common.model")

__all__ = ["Model"]
