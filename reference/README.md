# PS3R code and data

This release contains the Progressive Spatial Stratified Symbolic Regression
(PS3R) experiments for soil bulk density (BD), the single-stratum soil organic
carbon density (SOCD) control, and the four-stratum simulation.

## Contents

```text
code&data/
  model_ps3r.py              shared PS3R implementation
  utils_dso.py               shared logging and feature utilities
  requirements.txt           dependency manifest
  dso_fix/                   required DSO patch files
  bd/
    run_bd.py                five-fold spatial-block experiment
    config.json              DSO configuration
    SOCS_V10_spatial_block_5fold_354571m.csv
  socd/
    run_socd.py              fixed profile-split experiment
    config.json              DSO configuration
    train_datasetv10_profile80_cf1.csv
    test_datasetv10_profile20_cf1.csv
    train_datasetv10_profile80_full.csv
    test_datasetv10_profile20_full.csv
  simulation/
    run_simulation.py        fixed-split simulation experiment
    config.json              DSO configuration
    generate_simulation_data.ipynb
    simulation_data.csv
```

All three entry scripts import `PS3RModel` from the single parent-level
`model_ps3r.py`.

## Experiment designs

| Folder | Entry script | Design |
| --- | --- | --- |
| `bd` | `run_bd.py` | Five outer spatial folds; model seed 0; 19,978 observations |
| `socd` | `run_socd.py` | Fixed profile 80/20 split; model seeds 0-4; 16,068 training and 3,910 test rows |
| `simulation` | `run_simulation.py` | Fixed 8,000/2,000 split; model seeds 0-4 |

The simulation CSV is included. To regenerate it, run
`generate_simulation_data.ipynb` from the `simulation` directory. The notebook
writes `simulation_data.csv` in that directory.

## Runtime environment

The recorded environment used Python 3.8, NumPy 1.19.5, and Numba 0.53.1.
The DSO environment has strict dependency constraints. Install packages in the
following order, matching the installation sequence used by the preceding
release. Do not replace these steps with a single `pip install -r` command.

```bash
conda create -n dso_env python=3.8 -c conda-forge
conda activate dso_env

conda install numpy=1.19.5 numba=0.53.1 -c conda-forge
conda install pytorch::pytorch torchvision torchaudio -c pytorch
conda install pandas scikit-learn sympy click deap pathos seaborn tqdm pyyaml -c conda-forge

pip install commentjson prettytable progress
conda install cython -c conda-forge
```

Install the PyTorch DSO source distribution that provides both
`dso.DeepSymbolicRegressor` and `dso.program.Program`:

```bash
cd <path-to-deep-symbolic-optimization-pytorch>
pip install -e . --no-deps
```

For notebook execution, install Jupyter in the same environment after DSO is
installed:

```bash
conda install ipython jupyter -c conda-forge
```

Exact versions of packages other than Python, NumPy, and Numba were not
recorded. `requirements.txt` is therefore a dependency manifest rather than a
complete environment lock.

## DSO patches

Apply the included patches to the installed DSO source checkout before model
training:

| Included file | Destination within the `dso` package |
| --- | --- |
| `dso_fix/core.py` | `core.py` |
| `dso_fix/program.py` | `program.py` |
| `dso_fix/train.py` | `train.py` |
| `dso_fix/sklearn.py` | `task/regression/sklearn.py` |

For an editable DSO installation, copy the four files into the corresponding
locations in the DSO source checkout. Reinstallation is not required after
replacement.

## Validate the release

Run these commands from `code&data`. They validate the included inputs and
split definitions without fitting a model, so DSO and PyTorch are not required:

```bash
python bd/run_bd.py --validate-data-only
python socd/run_socd.py --validate-data-only
python simulation/run_simulation.py --validate-data-only
```

After installing DSO and applying the patches, verify the model import:

```bash
python -c "from model_ps3r import PS3RModel; print('PS3R import OK')"
```

## Run the experiments

Run from `code&data`:

```bash
python bd/run_bd.py
python socd/run_socd.py
python simulation/run_simulation.py
```

Each script reads the data and `config.json` in its own folder and creates a
timestamped result directory there. Common options are:

```text
--output-dir PATH       write results to a selected directory
--n-jobs N              number of concurrent folds or seeds
--validate-data-only    validate inputs without training
```

The SOCD and simulation scripts also accept `--seeds`. Use the default values
`0 1 2 3 4` for the reported design. The BD script uses model seed 0 for each
of the five predefined spatial folds.
