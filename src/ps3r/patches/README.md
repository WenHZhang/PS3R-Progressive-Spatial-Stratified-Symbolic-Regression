# Original DSO patches

These four files are unchanged from the AAAG final `code&data/dso_fix` directory.
Back up the existing files in the intended PyTorch DSO checkout before copying.

| File | Destination within the `dso` package |
| --- | --- |
| `core.py` | `core.py` |
| `program.py` | `program.py` |
| `train.py` | `train.py` |
| `sklearn.py` | `task/regression/sklearn.py` |

Restart Python after replacing files. Run `ps3r doctor --require-runtime` to
check imports and hashes. These patches do not constitute a complete backend.
