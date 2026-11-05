"""Packaging regressions; synthetic solver tests do not reproduce DSO training."""

import ast
import hashlib
import importlib
import json
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest

import ps3r
from ps3r import BinaryPathRouter, PS3RModel, get_config, get_data_path
from ps3r.resources import export_experiment, export_patches
from ps3r.runtime import MissingDSOError


class AffineProgram:
    evaluate = {"success": False}

    def __init__(self, X, y):
        self.training_X = X.copy()
        self.coefficients = np.linalg.lstsq(np.column_stack([X, np.ones(len(X))]), y, rcond=None)[0]

    def execute(self, X):
        return np.column_stack([X, np.ones(len(X))]) @ self.coefficients

    def pretty(self):
        return "synthetic affine solver for packaging smoke test"


class AffineSolver:
    """Explicit custom solver for checking model wiring, not a DSO substitute."""

    def fit(self, X, y):
        self.program_ = AffineProgram(X, y)
        return self

    def predict(self, X):
        return self.program_.execute(X)


def test_public_api_and_config_independence():
    assert ps3r.__version__ == "0.2.0"
    assert PS3RModel.__module__ == "ps3r.model"
    model = PS3RModel(input_dim=9, solver_kwargs={"config": get_config()})
    assert model.input_dim == 9
    config = get_config()
    config["experiment"]["seed"] = -100
    assert get_config()["experiment"]["seed"] == 666


def test_binary_route_boundaries_and_probabilities():
    router = BinaryPathRouter([(0, 0.0, True), (1, 1.0, False)], 4)
    X = np.array([[0, 2], [0, 1], [1, 2], [-1, 2]])
    np.testing.assert_array_equal(router.predict(X), [1, 0, 0, 1])
    np.testing.assert_array_equal(router.predict_proba(X).sum(axis=1), np.ones(4))
    fuzzy = router.fuzzy_predict_proba(X, [1, 1])
    assert np.all((fuzzy >= 0) & (fuzzy <= 1))
    np.testing.assert_allclose(fuzzy.sum(axis=1), 1)


def test_group_holdout_fit_predict_and_pickle():
    rng = np.random.RandomState(7)
    X = rng.normal(size=(200, 3))
    y = X @ np.array([2.0, -3.0, 0.5]) + 1.7
    groups = np.repeat(np.arange(20), 10)
    models = [PS3RModel(input_dim=3, base_solver_class=AffineSolver, max_depth=0) for _ in range(2)]
    for model in models:
        assert model.fit(X, y, spatial_block_ids=groups) is model
        assert not set(model._fixed_root_train_blocks) & set(model._fixed_root_validation_blocks)
        np.testing.assert_array_equal(model.root.solver.program_.training_X, X[model._fixed_root_train_indices])
        np.testing.assert_allclose(model.predict(X), y, atol=1e-11)
        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_allclose(restored.predict(X), y, atol=1e-11)
    np.testing.assert_array_equal(models[0]._fixed_root_train_indices, models[1]._fixed_root_train_indices)
    assert len(models[0]._fixed_root_train_indices) == 160
    assert len(models[0]._fixed_root_validation_indices) == 40


def test_group_ids_required():
    model = PS3RModel(input_dim=2, base_solver_class=AffineSolver, max_depth=0)
    with pytest.raises(ValueError, match="spatial_block_ids is required"):
        model.fit(np.ones((20, 2)), np.ones(20))
    with pytest.raises(ValueError, match="at least two distinct"):
        model.fit(np.ones((20, 2)), np.ones(20), spatial_block_ids=np.zeros(20))


def test_default_solver_missing_backend_has_actionable_error(monkeypatch):
    runtime = importlib.import_module("ps3r.runtime")
    real_import = importlib.import_module

    def unavailable(name, *args, **kwargs):
        if name in {"torch", "dso", "dso.program"}:
            raise ModuleNotFoundError("blocked backend in packaging test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(runtime.importlib, "import_module", unavailable)
    model = PS3RModel(input_dim=2)
    with pytest.raises(MissingDSOError, match="PyTorch DSO"):
        model._create_solver(0)


def test_original_resource_hashes():
    root = Path(ps3r.__file__).parent
    provenance = json.loads((root / "PROVENANCE.json").read_text())
    checked = 0
    for entry in provenance["files"]:
        if not entry["packaged"].startswith("src/ps3r/") or entry["adapted"]:
            continue
        path = root / entry["packaged"][len("src/ps3r/"):]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        checked += 1
    assert checked == 14  # six CSVs, three configs, one notebook, four patches


@pytest.mark.parametrize("experiment", ["bd", "socd", "simulation"])
def test_exported_experiment_is_standalone_with_installed_package(tmp_path, experiment):
    destination = export_experiment(experiment, tmp_path / experiment)
    assert (destination / "config.json").read_bytes() == get_data_path(experiment, "config.json").read_bytes()
    result = subprocess.run([sys.executable, str(destination / ("run_" + experiment + ".py")), "--validate-data-only"], capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    with pytest.raises(FileExistsError):
        export_experiment(experiment, destination)


def test_resource_name_validation_and_patch_export(tmp_path):
    with pytest.raises(ValueError):
        get_data_path("invalid")
    with pytest.raises(ValueError):
        get_data_path("bd", "../config.json")
    with pytest.raises(FileNotFoundError):
        get_data_path("bd", "missing.csv")
    exported = export_patches(tmp_path / "patches")
    assert {p.name for p in exported.glob("*.py")} == {"core.py", "train.py", "program.py", "sklearn.py"}


def test_cli_validate_all_outside_source_tree(tmp_path):
    result = subprocess.run([sys.executable, "-m", "ps3r", "validate", "all"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not list(tmp_path.iterdir())


def test_numerical_model_methods_unchanged():
    project = Path(__file__).resolve().parents[1]
    original = ast.parse((project / "reference/model_ps3r.py").read_text())
    packaged = ast.parse(Path(importlib.import_module("ps3r.model").__file__).read_text())
    def methods(tree):
        return {(cls.name, fn.name): fn for cls in tree.body if isinstance(cls, ast.ClassDef)
                for fn in cls.body if isinstance(fn, ast.FunctionDef)}
    old, new = methods(original), methods(packaged)
    assert old.keys() == new.keys()
    adapted = {"_initialize_router", "_create_solver", "_cleanup_memory", "_train_solver"}
    checked = 0
    for key in old:
        if key[1] not in adapted:
            assert ast.dump(old[key]) == ast.dump(new[key]), key
            checked += 1
    assert checked >= 50
    # Apart from cache access, the training method is bytecode-equivalent at AST level.
    class CacheCall(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "Program" and node.func.attr == "clear_cache":
                node.func = ast.Name(id="_clear_dso_cache", ctx=ast.Load())
            return node
    key = ("PS3RModel", "_train_solver")
    assert ast.dump(CacheCall().visit(old[key])) == ast.dump(new[key])


@pytest.mark.parametrize("experiment", ["bd", "socd", "simulation"])
def test_experiment_computations_unchanged(experiment):
    project = Path(__file__).resolve().parents[1]
    filename = "run_" + experiment + ".py"
    original = ast.parse((project / "reference" / experiment / filename).read_text())
    packaged = ast.parse((get_data_path(experiment) / filename).read_text())
    class PackageImports(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            node.module = {"model_ps3r": "ps3r.model", "utils_dso": "ps3r.utils"}.get(node.module, node.module)
            return node
    original = PackageImports().visit(original)
    old = {fn.name: fn for fn in original.body if isinstance(fn, ast.FunctionDef) and fn.name not in {"main", "_parse_args"}}
    new = {fn.name: fn for fn in packaged.body if isinstance(fn, ast.FunctionDef) and fn.name not in {"main", "_parse_args"}}
    assert old.keys() == new.keys()
    for key in old:
        assert ast.dump(old[key]) == ast.dump(new[key]), (experiment, key)
