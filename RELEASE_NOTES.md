# 0.2.0

Packaged from the specified AAAG final `code&data` source, rather than the
older local 0.1.0 package or historical experiment implementations.

- Added the `ps3r` namespace and public `PS3RModel`, `BinaryPathRouter`, and
  `TreeNode` interfaces.
- Preserved model numerical methods, experiment seeds, configuration contents,
  six CSV inputs, the simulation notebook, and the four DSO patch files.
- Replaced project-directory path injection and top-level imports with package
  imports. DSO and CUDA cleanup are loaded only when the backend is used.
- Resolve the default DSO solver at first solver construction. Missing DSO
  raises an actionable error; no fallback training algorithm is selected.
- Moved experiment default result directories to the working directory.
  Runners now check DSO before creating training outputs.
- Library import no longer suppresses global warnings or configures the
  application's root logger. Console runners configure their own logging.
- Added bundled-resource access, safe exports, dependency diagnostics,
  command-line entry points, provenance, tests, and release workflows.

Training reproduction requires the original external PyTorch DSO backend.
This release does not claim a new execution of the manuscript experiments.
