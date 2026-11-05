"""Command-line entry points for packaged PS3R experiments."""

import argparse
import json
import subprocess
import sys

from . import __version__
from .resources import EXPERIMENTS, export_experiment, export_patches
from .runtime import runtime_report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ps3r", description="PS3R package and experiment tools")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Inspect dependencies and DSO patches")
    doctor.add_argument("--require-runtime", action="store_true", help="Import DSO and fail if patches/runtime are missing")
    validate = commands.add_parser("validate", help="Check bundled inputs without training")
    validate.add_argument("experiment", choices=("all",) + EXPERIMENTS)
    export = commands.add_parser("export", help="Copy an experiment to a writable folder")
    export.add_argument("experiment", choices=EXPERIMENTS)
    export.add_argument("destination")
    patches = commands.add_parser("export-patches", help="Export the four DSO patches")
    patches.add_argument("destination")
    args = parser.parse_args(argv)
    if args.command == "doctor":
        report = runtime_report(check_import=args.require_runtime)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return int(args.require_runtime and not report["training_ready"])
    if args.command == "validate":
        names = EXPERIMENTS if args.experiment == "all" else (args.experiment,)
        for name in names:
            # Separate processes preserve each original runner's thread settings.
            result = subprocess.run([sys.executable, "-m", "ps3r.experiments.{}.run_{}".format(name, name), "--validate-data-only"])
            if result.returncode:
                return result.returncode
        return 0
    try:
        if args.command == "export":
            print(export_experiment(args.experiment, args.destination))
        else:
            print(export_patches(args.destination))
    except (OSError, ValueError) as exc:
        parser.exit(1, str(exc) + "\n")
    return 0
