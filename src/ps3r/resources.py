"""Read or export the unchanged experiment inputs shipped in the wheel."""

import json
from pathlib import Path
import shutil

EXPERIMENTS = ("bd", "socd", "simulation")


def _experiment_dir(experiment):
    if experiment not in EXPERIMENTS:
        raise ValueError("experiment must be one of {}".format(EXPERIMENTS))
    return Path(__file__).resolve().parent / "experiments" / experiment


def get_data_path(experiment, filename=None):
    """Return a bundled CSV/notebook path, or its experiment directory.

    Treat installed resources as read-only. To edit inputs, first use
    :func:`export_experiment` to copy the experiment to a writable directory.
    """
    root = _experiment_dir(experiment)
    if filename is None:
        return root
    if Path(filename).name != str(filename) or filename in ("", ".", ".."):
        raise ValueError("filename must be a single file name")
    path = root / filename
    if not path.is_file() or path.suffix not in {".csv", ".json", ".ipynb"}:
        raise FileNotFoundError("Bundled input not found: {}".format(filename))
    return path


def get_config(experiment="simulation"):
    """Return a fresh DSO configuration dictionary for an experiment."""
    return json.loads((_experiment_dir(experiment) / "config.json").read_text(encoding="utf-8"))


def _export(source, destination, extensions):
    destination = Path(destination).expanduser().resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError("Choose a new or empty destination: {}".format(destination))
    destination.mkdir(parents=True, exist_ok=True)
    for file in sorted(source.iterdir()):
        if file.is_file() and file.suffix in extensions and file.name != "__init__.py":
            shutil.copy2(file, destination / file.name)
    return destination


def export_experiment(experiment, destination):
    """Copy inputs and the runner to a new or empty writable directory."""
    return _export(_experiment_dir(experiment), destination, {".py", ".json", ".csv", ".ipynb"})


def export_patches(destination):
    """Copy the four original DSO patches and their destination map."""
    return _export(Path(__file__).parent / "patches", destination, {".py", ".md"})
