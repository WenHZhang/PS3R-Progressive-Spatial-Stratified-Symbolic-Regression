# ps3r

Progressive Spatial Stratified Symbolic Regression (PS3R) for interpretable modeling of variable relationships.
The package contains the shared model, three
experiment runners, all six input CSV files, configuration templates, the
simulation-generation notebook, and four original PyTorch DSO patches.

See [中文安装说明](README_zh.md) for instructions in Chinese.

## Installation and import

Use Python 3.8:

```bash
python -m pip install ps3r
```

For a downloaded wheel, install the supplied file instead:

```bash
python -m pip install ./ps3r-0.2.0-py3-none-any.whl
```

```python
import ps3r
from ps3r import PS3RModel, get_config, get_data_path

print(ps3r.__version__)
model = PS3RModel(input_dim=9, solver_kwargs={"config": get_config("simulation")})
csv_path = get_data_path("simulation", "simulation_data.csv")
```

Installation, model import/construction, configuration access, and input
validation do not require DSO or PyTorch. **Training with the default symbolic
solver requires the separately installed PyTorch DSO backend and its four
patches.** The original directory does not contain the complete backend, so
this wheel does not bundle it. `pip install 'ps3r[dso-runtime]'` installs its
supporting libraries, not the DSO implementation. See [DSO setup](docs/DSO_SETUP.md).

The original recorded training environment was Python 3.8, NumPy 1.19.5, and
Numba 0.53.1. The broader dependency ranges in this package support installation
and input validation on newer Python versions; they are not a guarantee of
identical training results across environments. Consult
[the original environment instructions](docs/ORIGINAL_README.md) for reproduction.

## Validate and run the supplied experiments

```bash
ps3r validate all
ps3r doctor
ps3r doctor --require-runtime
```

`validate` checks inputs and split definitions without fitting. The last
command also imports the backend and checks its four patch hashes; it exits
with an error if the backend is unavailable or the patches differ.

After configuring DSO, run any experiment from any working directory:

```bash
ps3r-bd --output-dir ./results/bd
ps3r-socd --output-dir ./results/socd --seeds 0 1 2 3 4
ps3r-simulation --output-dir ./results/simulation --seeds 0 1 2 3 4
```

Each runner supports `--help`, `--n-jobs`, and `--validate-data-only`. Default
outputs go to a timestamped directory in the current working directory.
Installed package data are read-only inputs, not a results directory.
Use `python -m ps3r` in place of `ps3r` if console scripts are not on your PATH.

| Experiment | Design | Included observations |
| --- | --- | --- |
| `bd` | Five predefined outer spatial folds; model seed 0 | 19,978 |
| `socd` | Fixed profile split; model seeds 0–4 | 16,068 train / 3,910 test |
| `simulation` | Fixed four-stratum split; model seeds 0–4 | 8,000 train / 2,000 test |

SOCD and simulation use fixed outer train/test splits; the five seeds are
sensitivity runs. All three models retain grouped internal validation and
the original routing, solver settings, random seeds, and refitting behavior.

## Train on your own data

Prepare a two-dimensional numeric feature matrix `X`, a one-dimensional target
`y`, and one spatial block/group identifier per training row. With DSO installed:

```python
from ps3r import PS3RModel, get_config

model = PS3RModel(
    input_dim=X_train.shape[1],
    solver_kwargs={"config": get_config("simulation")},
    continuous_feature_indices=list(range(X_train.shape[1])),
    random_state=0,
)
model.fit(X_train, y_train, spatial_block_ids=train_block_ids)
predictions = model.predict(X_test)
structure = model.get_model_structure()
```

Only training rows and their group identifiers enter `fit`. Supply at least
two distinct groups so the model can reserve whole groups for internal
validation. Configure symbolic and routing features for your own problem;
the example above does not select hyperparameters for a new dataset.

## Export editable resources

```bash
ps3r export simulation ./simulation-example
python ./simulation-example/run_simulation.py --validate-data-only
ps3r export-patches ./dso-patches
```

Exports require a new or empty destination and do not overwrite existing
experiments. Exported runners import the installed `ps3r` package.

## Build and development

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m build
python -m twine check dist/*.whl dist/*.tar.gz
```

`dist/` contains a platform-independent wheel and source distribution. The
source distribution also includes byte-identical reference model/runner files
and packaging tests. `ps3r/PROVENANCE.json` records original file hashes.
See [release notes](RELEASE_NOTES.md), [validation scope](docs/VALIDATION.md),
and [publishing instructions](docs/PUBLISHING.md).

PS3R is released under the [MIT license](LICENSE), as selected by the author.
The included DSO-derived patches retain their upstream BSD-3-Clause terms;
see [third-party notices](THIRD_PARTY_NOTICES.md).
