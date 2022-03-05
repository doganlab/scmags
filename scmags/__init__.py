from ._api import ScMags
from . import datasets

try:
    import importlib.metadata 
except ModuleNotFoundError:
    import importlib_metadata

__version__ = importlib.metadata.version('scmags')
