import os
from .tokenizer import Tokenizer
from ..common import import_all_module

realpath = os.path.realpath(__file__)
basedir = os.path.dirname(realpath)
import_all_module(basedir, "common.tokenizer")

__all__ = ["Tokenizer"]
