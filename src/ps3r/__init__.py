"""PS3R public interface. The DSO training backend is loaded only when needed."""

from .resources import export_experiment, export_patches, get_config, get_data_path

__version__ = "0.2.0"
__all__ = [
    "PS3RModel", "BinaryPathRouter", "TreeNode", "get_config", "get_data_path",
    "export_experiment", "export_patches", "__version__",
]


def __getattr__(name):
    if name in {"PS3RModel", "BinaryPathRouter", "TreeNode"}:
        from . import model
        value = getattr(model, name)
        globals()[name] = value
        return value
    raise AttributeError("module 'ps3r' has no attribute {!r}".format(name))
