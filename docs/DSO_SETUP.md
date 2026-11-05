# PyTorch DSO training backend

The supplied source directory contains four patch files, not the complete
`dso` package. Do not assume that the unrelated package name `dso` on a package
index supplies the required backend. The required checkout must expose
`dso.DeepSymbolicRegressor` and `dso.program.Program` and use PyTorch.

For exact historical environment instructions, follow `ORIGINAL_README.md`.
Python 3.8 / NumPy 1.19.5 / Numba 0.53.1 are the recorded versions; other
historical dependency versions were not recorded.

1. Obtain the PyTorch DSO source checkout identified by the author:

   ```bash
   git clone https://github.com/dso-org/deep-symbolic-optimization-pytorch.git
   cd deep-symbolic-optimization-pytorch
   git checkout 85f88faf3ff95636f0011764b30e13f7bb82b54c
   ```

   This is the upstream snapshot inspected for this package, not a verified
   record of the historical experiment commit.
2. In the intended training environment, install its dependencies in the
   recorded order. The `ps3r[dso-runtime]` extra lists supporting libraries
   but is not a frozen reproduction environment.
3. Install the source checkout:

   ```bash
   python -m pip install -e ./dso --no-deps --no-build-isolation
   ```

4. Export the unchanged patches:

   ```bash
   ps3r export-patches ./dso-patches
   ```

5. Back up the corresponding files in the DSO checkout, then copy the patches
   into the `dso/dso/` package directory within the checkout according to this table:

   | Exported patch | Destination relative to the `dso` package |
   | --- | --- |
   | `core.py` | `core.py` |
   | `program.py` | `program.py` |
   | `train.py` | `train.py` |
   | `sklearn.py` | `task/regression/sklearn.py` |

6. Restart any Python processes that previously imported DSO, then run:

   ```bash
   ps3r doctor --require-runtime
   ```

The command imports the backend and checks that all four file hashes match.
It does not prove that a complete training experiment succeeds. The package
never modifies a separately installed DSO checkout automatically.
