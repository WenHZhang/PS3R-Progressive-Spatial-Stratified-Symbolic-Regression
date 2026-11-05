"""Access the separately installed, patched PyTorch DSO backend."""

import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys


class MissingDSOError(ImportError):
    """The required PyTorch DSO backend is unavailable or incompatible."""


_HELP = (
    "PS3R training requires the PyTorch DSO source checkout used by the "
    "experiments, plus its four supplied patches. Install that checkout with "
    "'python -m pip install -e /path/to/deep-symbolic-optimization-pytorch/dso "
    "--no-deps --no-build-isolation'. Source: "
    "https://github.com/dso-org/deep-symbolic-optimization-pytorch . "
    "Export patches with 'ps3r export-patches ./dso-patches'. "
    "See README_zh.md or docs/DSO_SETUP.md. Installing ps3r[dso-runtime] "
    "provides supporting dependencies, not the DSO source itself."
)


def require_dso():
    """Return the real DSO regressor; never substitute another algorithm."""
    try:
        importlib.import_module("torch")
        dso = importlib.import_module("dso")
        program = importlib.import_module("dso.program")
    except ImportError as exc:
        raise MissingDSOError(_HELP + " Original error: " + str(exc)) from exc
    regressor = getattr(dso, "DeepSymbolicRegressor", None)
    if regressor is None or not hasattr(program, "Program"):
        raise MissingDSOError(_HELP + " The installed dso lacks the required API.")
    return regressor


def clear_program_cache():
    """Clear the DSO cache when DSO has been loaded by a real solver."""
    module = sys.modules.get("dso.program")
    if module is not None and hasattr(module, "Program"):
        module.Program.clear_cache()


def cleanup_cuda():
    """Release CUDA memory when PyTorch is already in use."""
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def runtime_report(check_import=False):
    """Report dependency locations and patch hashes without changing files."""
    from . import __version__
    report = {"ps3r": __version__, "python": sys.version.split()[0],
              "dependencies": {}, "patches": {}, "training_ready": False}
    for name in ("numpy", "pandas", "scipy", "sklearn", "torch", "dso"):
        spec = importlib.util.find_spec(name)
        report["dependencies"][name] = None if spec is None else spec.origin
    origin = report["dependencies"]["dso"]
    if origin:
        root = Path(origin).parent
        patches = Path(__file__).parent / "patches"
        for source, target in PATCH_TARGETS.items():
            installed = root / target
            expected = hashlib.sha256((patches / source).read_bytes()).hexdigest()
            report["patches"][target] = (
                installed.is_file()
                and hashlib.sha256(installed.read_bytes()).hexdigest() == expected
            )
    if check_import:
        try:
            require_dso()
        except (ImportError, RuntimeError, OSError, AttributeError) as exc:
            report["error"] = str(exc)
        else:
            report["training_ready"] = all(report["patches"].values()) and bool(report["patches"])
            if not report["training_ready"]:
                report["error"] = "DSO imports, but the four experiment patches do not all match."
    return report


PATCH_TARGETS = {
    "core.py": "core.py",
    "program.py": "program.py",
    "train.py": "train.py",
    "sklearn.py": "task/regression/sklearn.py",
}
