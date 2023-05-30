import os
import glob
import importlib


def import_all_module(basedir, prefix):
    # realpath = os.path.realpath(__file__)
    # realdir = os.path.dirname(realpath)
    all_py_files = glob.glob(basedir+"/*.py")
    modules = [os.path.basename(f) for f in all_py_files]

    for module in modules:
        if not module.startswith("_"):
            module = module.strip(".py")
            module_name = f"{prefix}.{module}"
            importlib.import_module(module_name)