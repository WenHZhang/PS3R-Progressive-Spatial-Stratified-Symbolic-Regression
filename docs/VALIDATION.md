# Validation scope

Validation is performed on the prepared release in a macOS ARM64 environment
with Python 3.13.5. The accompanying `VALIDATION_RESULT.json` records the
observed checks for the delivered artifacts.

## Package checks

- Build a wheel and a source distribution using the declared build backend.
- Check both distributions using `twine check --strict`.
- Install the wheel with its declared dependencies in a new virtual environment.
- From outside the source directory, import `ps3r` and `PS3RModel`, construct
  the model, and run all three input-validation commands.
- Check the installed wheel's six CSVs, three configurations, simulation
  notebook, and four DSO patches against the original SHA-256 hashes.
- Build another wheel from the source distribution, install it separately,
  and repeat the import and input checks.

## Behavioral checks

The test suite checks routing boundaries, fresh configuration dictionaries,
safe exports, model serialization, deterministic grouped holdout, and
fit/predict wiring using an explicitly supplied synthetic affine solver.
This synthetic solver is only a test fixture and is not installed as an
alternative default model.

AST comparisons check that model numerical methods and experiment calculation
functions remain unchanged except for documented dependency imports/cache
access. Original model/runner files are retained in `reference/`.

All files in the specified original `final/code&data` directory are read-only
throughout packaging and are checked again against their initial hashes.

## Limits

The prepared package does not include a complete DSO backend. The upstream
repository and its inspected commit are documented, but full DSO training and
the original manuscript experiments are not rerun by these packaging checks.
Successful import, data validation, or synthetic-solver checks do not establish
training reproduction. Windows/Linux execution and the CI Python-version
matrix are only confirmed after the corresponding CI jobs run.

Remote publication must be verified separately using the actual GitHub release
and PyPI project/version pages. Building local artifacts is not publication.
