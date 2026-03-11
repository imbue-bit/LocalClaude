import os
import importlib

component_dir = os.path.dirname(__file__)
for filename in os.listdir(component_dir):
    if filename.endswith(".py") and filename not in ["__init__.py", "base.py"]:
        module_name = filename[:-3]
        importlib.import_module(f".{module_name}", package=__name__)

from .base import MUTATOR_REGISTRY