import os
import glob
import importlib

from llmonray.common.logging import logger


def import_all_module(basedir, prefix=None):
    all_py_files = glob.glob(basedir + "/*.py")
    modules = [os.path.basename(f) for f in all_py_files]

    for module in modules:
        if not module.startswith("_"):
            module = module.rstrip(".py")
            if prefix is None:
                module_name = module
            else:
                module_name = f"{prefix}.{module}"
            try:
                importlib.import_module(module_name)
            except Exception:
                logger.warning(f"import {module_name} erro", exc_info=True)
