"""Progressive Spatial Stratified Symbolic Regression (PS3R)."""

from __future__ import annotations


import gc
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text

PROJECT_ROOT = Path(__file__).resolve().parent

from .runtime import cleanup_cuda, clear_program_cache as _clear_dso_cache, require_dso
from .utils import get_logger

logger = get_logger(__name__)


PathCondition = Tuple[int, float, bool]
PATH_SELECTION_MODES = ("root_to_leaf", "all_subpaths")


class BinaryPathRouter:
    """Binary router defined by one root-to-leaf conjunction."""

    def __init__(
        self,
        conditions: Sequence[PathCondition],
        source_leaf_id: int,
    ):
        if not conditions:
            raise ValueError("A binary path router requires at least one condition.")
        self.conditions = tuple(
            (int(feature), float(threshold), bool(go_left))
            for feature, threshold, go_left in conditions
        )
        self.source_leaf_id = int(source_leaf_id)
        self.classes_ = np.array([0, 1], dtype=int)

    @property
    def path_depth(self) -> int:
        return len(self.conditions)

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        mask = np.ones(len(X), dtype=bool)
        for feature, threshold, go_left in self.conditions:
            if go_left:
                mask &= X[:, feature] <= threshold
            else:
                mask &= X[:, feature] > threshold
        return mask.astype(int)

    def predict_proba(self, X):
        labels = self.predict(X)
        return np.column_stack([1 - labels, labels]).astype(float)

    def fuzzy_predict_proba(self, X, feature_scales, alpha=10.0):
        """Return a smooth AND-membership and its complement."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        feature_scales = np.asarray(feature_scales, dtype=float)
        membership = np.ones(len(X), dtype=float)
        for feature, threshold, go_left in self.conditions:
            scale = max(float(feature_scales[feature]), 1e-8)
            signed_distance = (X[:, feature] - threshold) / scale
            if go_left:
                logits = -alpha * signed_distance
            else:
                logits = alpha * signed_distance
            logits = np.clip(logits, -60.0, 60.0)
            membership *= 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - membership, membership])

    def rule_text(self, feature_names=None) -> str:
        if feature_names is None:
            max_feature = max(condition[0] for condition in self.conditions)
            feature_names = [f"x{i + 1}" for i in range(max_feature + 1)]
        pieces = []
        for feature, threshold, go_left in self.conditions:
            operator = "<=" if go_left else ">"
            pieces.append(f"{feature_names[feature]} {operator} {threshold:.8g}")
        return " AND ".join(pieces)


class TreeNode:
    """Tree node used by :class:`PS3RModel`.

    Each node stores one symbolic solver and optional discriminator model that
    routes samples to child nodes. Leaf nodes keep ``is_leaf=True`` and have no
    effective discriminator-based split during inference.
    """

    def __init__(self, depth=0, parent=None, node_id=0):
        """Initialize a tree node container.

        Args:
            depth (int): Depth level of this node in the routing tree.
            parent (TreeNode | None): Parent node. ``None`` for root.
            node_id (int): Globally unique node identifier.

        Returns:
            None: The constructor initializes in-place attributes.

        Raises:
            TypeError: Propagated if callers pass unsupported argument types.
        """
        self.depth = depth
        self.parent = parent
        self.solver = None
        self.discriminator = None
        self.children = []
        self.is_leaf = True
        self.is_frozen = False
        self.sample_indices = None
        self.node_id = node_id
        self.performance = None
        self.split_label = None

    def __repr__(self):
        """Return a concise node summary."""
        status = "FROZEN" if self.is_frozen else "ACTIVE"
        leaf_status = "LEAF" if self.is_leaf else "INTERNAL"
        return f"TreeNode(id={self.node_id}, depth={self.depth}, {leaf_status}, {status})"


class PS3RModel:
    node_holdout_strategy = "fixed_root_80_20_inherited"
    root_validation_fraction = 0.20
    root_validation_seed = 42
    minimum_validation_samples = 10

    """PS3R model with recursive symbolic experts and data-driven routing.

    The class trains symbolic solvers at tree nodes and recursively grows a
    routing structure. During prediction, samples are either routed by local
    node discriminators or by an optional final global discriminator.
    """

    def _initialize_router(
        self,
        input_dim,
        base_solver_class=None,
        solver_kwargs=None,
        continuous_feature_indices=None,
        max_depth=10,
        min_samples_split=50,
        random_state=0,
        hard_pred=False,
    ):
        """Initialize the adaptive expert router.

        Args:
            input_dim (int): Number of full input features.
            base_solver_class (type): Symbolic regressor class used per node.
            solver_kwargs (dict | None): Keyword arguments for solver instances.
            continuous_feature_indices (list[int] | np.ndarray | None):
                Indices used by symbolic solvers. If ``None``, all features.
            max_depth (int): Maximum tree growth depth.
            min_samples_split (int): Minimum sample count required to split.

        Returns:
            None: The constructor initializes all object state.

        Raises:
            ValueError: If feature index processing fails due to invalid inputs.
            TypeError: Propagated for unsupported argument types.
        """
        self.input_dim = input_dim
        self.base_solver_class = base_solver_class
        self.solver_kwargs = solver_kwargs or {}
        if continuous_feature_indices is None:
            continuous_feature_indices = list(range(input_dim))
        self.continuous_feature_indices = np.array(continuous_feature_indices, dtype=int)
        self.symbolic_input_dim = int(self.continuous_feature_indices.size)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.global_discriminator = None
        self.node_counter = 0
        self.leaf_nodes = {}
        self._base_config_cache: Dict[str, Dict[str, Any]] = {}
        self.random_state = random_state
        self.hard_pred = hard_pred
        self.sbcv_split_records = []
        self.sbcv_growth_history = []
        self.sbcv_monitoring_data: Dict[str, Dict[str, np.ndarray]] = {}
        self.sbcv_leaf_refit_records = []
        self._sbcv_depth_iteration = 0
        self._sbcv_accepted_node_ids = []
        logger.info(
            f"Initialized PS3RModel with input_dim={input_dim}, "
            f"symbolic_input_dim={self.symbolic_input_dim}, max_depth={max_depth}"
        )

    @staticmethod
    def _coerce_route_labels(values):
        labels = np.asarray(values).ravel()
        if np.issubdtype(labels.dtype, np.floating):
            labels = np.rint(labels).astype(int)
        else:
            labels = labels.astype(int)
        return labels

    def _predict_route_labels(self, classifier, X_values):
        if len(X_values) == 0:
            return np.array([], dtype=int)
        return self._coerce_route_labels(classifier.predict(X_values))
    def _assert_fixed_holdout_state(self, node):
        """Require disjoint inherited sets that exactly cover one node."""
        node_indices = np.asarray(node.sample_indices, dtype=int)
        train_indices = np.asarray(node.inherited_train_indices, dtype=int)
        validation_indices = np.asarray(
            node.inherited_validation_indices,
            dtype=int,
        )
        for name, values in (
            ("node", node_indices),
            ("train", train_indices),
            ("validation", validation_indices),
        ):
            if np.unique(values).size != values.size:
                raise AssertionError(
                    f"Node {node.node_id}: duplicate {name} indices."
                )
        if np.intersect1d(train_indices, validation_indices).size:
            raise AssertionError(
                f"Node {node.node_id}: inherited train/validation overlap."
            )
        inherited_all = np.concatenate([train_indices, validation_indices])
        if not np.array_equal(np.sort(inherited_all), np.sort(node_indices)):
            raise AssertionError(
                f"Node {node.node_id}: inherited indices do not exactly "
                "cover node samples."
            )

        solver_training_indices = getattr(
            node,
            "solver_training_indices",
            None,
        )
        if solver_training_indices is not None:
            solver_training_indices = np.asarray(
                solver_training_indices,
                dtype=int,
            )
            if np.intersect1d(
                solver_training_indices,
                self._fixed_root_validation_indices,
            ).size:
                raise AssertionError(
                    f"Node {node.node_id}: parent solver has seen fixed "
                    "validation samples."
                )
        return train_indices, validation_indices

    def set_sbcv_monitoring_data(self, **datasets):
        """Register datasets whose predictions are captured during growth.

        Each keyword value must be ``(X, y)``. Arrays are retained by reference
        for the duration of fitting; the model does not modify them.
        """
        monitoring_data = {}
        for name, values in datasets.items():
            if not isinstance(values, (tuple, list)) or len(values) != 2:
                raise ValueError(
                    f"Monitoring dataset {name!r} must be supplied as (X, y)."
                )
            X_values, y_values = values
            monitoring_data[str(name)] = {
                "X": np.asarray(X_values),
                "y": np.asarray(y_values).ravel(),
            }
        self.sbcv_monitoring_data = monitoring_data
        return self

    @staticmethod
    def _sanitize_snapshot_predictions(values):
        values = np.asarray(values, dtype=float).ravel()
        return np.nan_to_num(values, nan=0.0, posinf=1e10, neginf=-1e10)

    def _record_growth_snapshot(
        self,
        event_type,
        activated_node=None,
        layer_attempted_node_ids=(),
        layer_accepted_node_ids=(),
    ):
        """Capture the current model immediately after a real training event."""
        predictions = {}
        for name, data in self.sbcv_monitoring_data.items():
            predictions[name] = self._sanitize_snapshot_predictions(
                self.predict(data["X"])
            ).copy()
        self.sbcv_growth_history.append(
            {
                "growth_step": len(self._sbcv_accepted_node_ids),
                "event_type": str(event_type),
                "depth_iteration": int(self._sbcv_depth_iteration),
                "activated_node_id": (
                    None if activated_node is None else int(activated_node.node_id)
                ),
                "activated_node_depth": (
                    None if activated_node is None else int(activated_node.depth)
                ),
                "accepted_node_ids": tuple(self._sbcv_accepted_node_ids),
                "layer_attempted_node_ids": tuple(
                    int(value) for value in layer_attempted_node_ids
                ),
                "layer_accepted_node_ids": tuple(
                    int(value) for value in layer_accepted_node_ids
                ),
                "layer_growth_occurred": bool(layer_accepted_node_ids),
                "n_effective_strata": int(len(self._get_all_leaves())),
                "predictions": predictions,
            }
        )

    def _cleanup_memory(self, clear_program_cache=False):
        """Release temporary training caches without changing retained solvers."""
        if clear_program_cache:
            _clear_dso_cache()
        gc.collect()
        cleanup_cuda()

    def _select_continuous_features(self, X):
        """Select features used by symbolic solvers.

        Args:
            X (array-like): Input feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            np.ndarray: Matrix restricted to ``continuous_feature_indices``.

        Raises:
            IndexError: If configured feature indices are out of bounds.
            ValueError: If input cannot be interpreted as a 2D array.
        """
        X_array = np.asarray(X)
        return X_array[:, self.continuous_feature_indices]

    def _resolve_config_path(self, config_path):
        """Resolve solver config file path against project root if needed.

        Args:
            config_path (str | Path): Relative or absolute path to JSON config.

        Returns:
            Path: Absolute resolved path.

        Raises:
            OSError: If path resolution fails at filesystem level.
        """
        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = PROJECT_ROOT / config_file
        return config_file.resolve()

    def _load_config_dict(self, config_path):
        """Load and cache a solver configuration dictionary from JSON.

        Args:
            config_path (str | Path): Path to JSON config.

        Returns:
            dict[str, Any]: Deep-copied configuration dictionary.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            json.JSONDecodeError: If the JSON content is invalid.
            OSError: If file reading fails.
        """
        resolved_path = self._resolve_config_path(config_path)
        cache_key = str(resolved_path)
        cached = self._base_config_cache.get(cache_key)
        if cached is not None:
            return copy.deepcopy(cached)
        text = resolved_path.read_text(encoding="utf-8")
        config_dict = json.loads(text)
        self._base_config_cache[cache_key] = config_dict
        return copy.deepcopy(config_dict)

    def _build_solver_kwargs(self, node_id):
        """Build solver kwargs for a specific node with deterministic seed wiring.

        Args:
            node_id (int): Current node identifier used as seed source.

        Returns:
            dict[str, Any]: Solver keyword arguments for node construction.

        Raises:
            FileNotFoundError: If referenced config file is missing.
            json.JSONDecodeError: If referenced config JSON is malformed.
            TypeError: If config structure is incompatible with expected schema.
        """
        solver_kwargs = copy.deepcopy(self.solver_kwargs)
        seed = int(node_id) + self.random_state
        config_value = solver_kwargs.get("config")
        if isinstance(config_value, str):
            config_dict = self._load_config_dict(config_value)
            experiment_cfg = copy.deepcopy(config_dict.get("experiment", {}))
            experiment_cfg["seed"] = seed
            config_dict["experiment"] = experiment_cfg
            solver_kwargs["config"] = config_dict
        elif isinstance(config_value, dict):
            config_dict = copy.deepcopy(config_value)
            experiment_cfg = copy.deepcopy(config_dict.get("experiment", {}))
            experiment_cfg["seed"] = seed
            config_dict["experiment"] = experiment_cfg
            solver_kwargs["config"] = config_dict
        elif "random_state" in solver_kwargs:
            solver_kwargs["random_state"] = seed
        elif "seed" in solver_kwargs:
            solver_kwargs["seed"] = seed
        return solver_kwargs

    def _create_solver(self, node_id):
        """Instantiate a node solver using node-specific kwargs.

        Args:
            node_id (int): Node identifier used for seed/config adaptation.

        Returns:
            Any: Instantiated solver object.

        Raises:
            Exception: Propagated from underlying solver constructor.
        """
        solver_class = self.base_solver_class or require_dso()
        solver = solver_class(**self._build_solver_kwargs(node_id))
        return solver

    def _get_next_node_id(self):
        """Allocate and return the next unique node id.

        Args:
            None

        Returns:
            int: Next increasing node identifier.

        Raises:
            None.
        """
        self.node_counter += 1
        return self.node_counter


    def _evaluate_solver(self, solver, X, y):
        """Compute mean squared error of a solver on given samples.

        Args:
            solver (Any): Solver instance exposing ``predict``.
            X (array-like): Input samples.
            y (array-like): Ground-truth targets.

        Returns:
            float: Mean squared error after numerical sanitization.

        Raises:
            Exception: Propagated from solver prediction or metric computation.
        """
        pred = solver.predict(self._select_continuous_features(X))
        pred = np.asarray(pred).ravel()
        y = np.asarray(y).ravel()
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e10, neginf=-1e10)
        return mean_squared_error(y, pred)

    def _get_leaves(self, node=None):
        """Return all active (not frozen) leaf nodes under a subtree.

        Args:
            node (TreeNode | None): Subtree root. Defaults to global root.

        Returns:
            list[TreeNode]: Active leaf nodes.

        Raises:
            AttributeError: If tree structure is not initialized correctly.
        """
        if node is None:
            node = self.root
        leaves = []
        stack = [node]
        while stack:
            current = stack.pop()
            if current.is_leaf and not current.is_frozen:
                leaves.append(current)
            else:
                stack.extend(current.children)
        return leaves

    def _get_all_leaves(self, node=None):
        """Return all leaf nodes under a subtree, including frozen leaves.

        Args:
            node (TreeNode | None): Subtree root. Defaults to global root.

        Returns:
            list[TreeNode]: All leaf nodes in DFS traversal domain.

        Raises:
            AttributeError: If tree structure is not initialized correctly.
        """
        if node is None:
            node = self.root
        leaves = []
        stack = [node]
        while stack:
            current = stack.pop()
            if current.is_leaf:
                leaves.append(current)
            else:
                stack.extend(current.children)
        return leaves

    def _find_best_solver_for_samples(self, solvers, X, y):
        """Assign each sample to the solver with minimum per-sample error.

        Args:
            solvers (list[Any]): Candidate child solvers.
            X (array-like): Candidate input samples.
            y (array-like): Corresponding targets.

        Returns:
            np.ndarray: Integer solver index per sample.

        Raises:
            ValueError: If ``solvers`` is empty.
            Exception: Propagated from solver evaluation.
        """
        if not solvers:
            raise ValueError("At least one solver is required.")

        X_array = np.asarray(X)
        y_array = np.asarray(y, dtype=float).ravel()
        if len(X_array) != y_array.size:
            raise ValueError(
                f"Sample count mismatch: X has {len(X_array)} rows, "
                f"but y has {y_array.size} values."
            )

        X_continuous = self._select_continuous_features(X_array)
        squared_errors = []
        for solver_index, solver in enumerate(solvers):
            predictions = np.asarray(
                solver.predict(X_continuous),
                dtype=float,
            ).ravel()
            if predictions.size != y_array.size:
                raise ValueError(
                    f"Solver {solver_index} returned {predictions.size} "
                    f"predictions for {y_array.size} samples."
                )

            predictions = np.nan_to_num(
                predictions,
                nan=0.0,
                posinf=1e10,
                neginf=-1e10,
            )
            squared_errors.append(np.square(predictions - y_array))

        error_matrix = np.column_stack(squared_errors)
        return np.argmin(error_matrix, axis=1).astype(int, copy=False)


    def _train_discriminator(self, X, labels, max_depth=2, force_classifier=False):
        """Train a discriminator model that predicts child assignment labels.

        Args:
            X (array-like): Input features for discriminator fitting.
            labels (array-like): Routing labels.
            max_depth (int): Maximum discriminator tree depth.
            force_classifier (bool): Force classifier even for multi-label targets.

        Returns:
            DecisionTreeClassifier | DecisionTreeRegressor | None:
                Trained discriminator, or ``None`` when label set is degenerate.

        Raises:
            ValueError: Propagated if scikit-learn rejects fitting inputs.
            Exception: Other training-time errors from scikit-learn.
        """
        X = np.asarray(X)
        labels = np.asarray(labels)
        if len(np.unique(labels)) < 2:
            return None
        if force_classifier:
            tree_model = DecisionTreeClassifier(
                criterion="gini",
                max_depth=max_depth,
                min_samples_leaf=10,
                random_state=42+self.random_state,
            )
        elif len(np.unique(labels)) > 2:
            tree_model = DecisionTreeRegressor(
                criterion="squared_error",
                max_depth=max_depth,
                min_samples_leaf=10,
                random_state=42+self.random_state,
            )
        else:
            tree_model = DecisionTreeClassifier(
                criterion="gini",
                max_depth=max_depth,
                min_samples_leaf=10,
                random_state=42+self.random_state,
            )
        discriminator = tree_model
        discriminator.fit(X, labels)
        return discriminator

    def _validate_split(
        self,
        parent_node,
        child_solvers,
        discriminator,
        X,
        y,
        before_predictions,
    ):
        """Validate an installed candidate split with whole-tree soft output.

        ``before_predictions`` is captured immediately before the candidate
        split is attached.  At entry, the candidate split is already attached
        temporarily, so the second prediction uses exactly the same soft
        membership-weighted inference as final model evaluation.  The fixed
        root-level validation holdout is used in both cases.
        """
        if discriminator is None:
            logger.info(
                f"Node {parent_node.node_id} split validation failed because "
                "the router is unavailable (all samples may share one label)."
            )
            return False

        target_indices = np.asarray(
            self._fixed_root_validation_indices,
            dtype=int,
        )
        if len(target_indices) < self.minimum_validation_samples:
            logger.info(
                f"Node {parent_node.node_id} split validation failed because "
                f"the fixed whole-tree holdout is too small ({len(target_indices)} < "
                f"{self.minimum_validation_samples})"
            )
            return False

        X_target = X[target_indices]
        y_target = np.asarray(y[target_indices], dtype=float).ravel()
        before_pred = np.asarray(
            before_predictions,
            dtype=float,
        ).ravel()
        after_pred = np.asarray(
            self._predict_with_tree_routing_prob_weighted(X_target),
            dtype=float,
        ).ravel()

        if before_pred.size != y_target.size:
            logger.info(
                f"    >>> Split Rejected for Node {parent_node.node_id}: "
                f"before-split whole-tree prediction length "
                f"{before_pred.size} != "
                f"target length {y_target.size}"
            )
            return False
        if after_pred.size != y_target.size:
            logger.info(
                f"    >>> Split Rejected for Node {parent_node.node_id}: "
                f"after-split whole-tree prediction length {after_pred.size} "
                f"!= "
                f"target length {y_target.size}"
            )
            return False

        if not (
            np.all(np.isfinite(y_target))
            and np.all(np.isfinite(before_pred))
            and np.all(np.isfinite(after_pred))
        ):
            logger.info(
                f"    >>> Split Rejected for Node {parent_node.node_id}: "
                "whole-tree validation targets or predictions contain NaN/Inf"
            )
            return False

        before_loss_each = np.square(before_pred - y_target)
        after_loss_each = np.square(after_pred - y_target)
        if not (
            np.all(np.isfinite(before_loss_each))
            and np.all(np.isfinite(after_loss_each))
        ):
            logger.info(
                f"    >>> Split Rejected for Node {parent_node.node_id}: "
                "whole-tree squared errors contain NaN/Inf"
            )
            return False

        try:
            _, p_value = wilcoxon(
                before_loss_each,
                after_loss_each,
                alternative="greater",
                zero_method="zsplit",
            )
        except ValueError as exc:
            logger.info(
                f"    >>> Split Rejected for Node {parent_node.node_id}: "
                f"whole-tree Wilcoxon test failed ({exc})"
            )
            return False

        before_rmse = float(np.sqrt(np.mean(before_loss_each)))
        after_rmse = float(np.sqrt(np.mean(after_loss_each)))
        rmse_improvement = before_rmse - after_rmse
        self._last_split_validation = {
            "scope": "fixed_root_validation_whole_tree",
            "inference": "soft_membership_weighted",
            "n_validation_samples": int(len(target_indices)),
            "before_rmse": before_rmse,
            "after_rmse": after_rmse,
            "rmse_improvement": rmse_improvement,
            "wilcoxon_p_value": float(p_value),
        }

        logger.info(
            f"    Whole-tree Soft Split Validation "
            f"(Node {parent_node.node_id}):"
        )
        logger.info(
            f"      Before-split RMSE: {before_rmse:.6f}, "
            f"After-split RMSE: {after_rmse:.6f}"
        )
        logger.info(
            f"      RMSE Improvement: {rmse_improvement:.6f} "
            "(Required: > 0)"
        )
        logger.info(f"      Wilcoxon: p={p_value:.6f} (Required: <= 0.1)")

        p_value_passed = bool(np.isfinite(p_value) and p_value <= 0.1)
        rmse_passed = bool(after_rmse < before_rmse)
        if p_value_passed and rmse_passed:
            logger.info(
                f"    >>> Split Accepted for Node {parent_node.node_id}"
            )
            return True

        reasons = []
        if not p_value_passed:
            reasons.append(f"p-value({p_value:.4f}) > 0.1 or non-finite")
        if not rmse_passed:
            reasons.append(
                f"after-split RMSE({after_rmse:.6f}) >= "
                f"before-split RMSE({before_rmse:.6f})"
            )
        logger.info(f"    >>> Split Rejected: {', '.join(reasons)}")
        return False

    def _execute_split(self, parent_node, child_solvers, discriminator, best_indices, child_node_ids):
        """Materialize a validated split by creating and attaching child nodes.

        Args:
            parent_node (TreeNode): Parent node to convert into an internal node.
            child_solvers (list[Any]): Trained child solvers.
            discriminator (Any): Trained router for child dispatching.
            best_indices (np.ndarray): Child labels over parent samples.
            child_node_ids (list[int]): Global node ids for child slots.

        Returns:
            None: Tree is modified in place.

        Raises:
            IndexError: If label/index mapping is inconsistent.
            Exception: Propagated from child performance evaluation.
        """
        parent_node.is_leaf = False
        parent_node.discriminator = discriminator
        unique_labels = np.unique(best_indices)

        for label in unique_labels:
            mask = best_indices == label
            child_indices = parent_node.sample_indices[mask]
            child_node = TreeNode(
                depth=parent_node.depth + 1,
                parent=parent_node,
                node_id=child_node_ids[int(label)],
            )
            child_node.solver = child_solvers[label]
            child_node.sample_indices = child_indices
            child_node.split_label = int(label)
            child_node.performance = self._evaluate_solver(
                child_solvers[label],
                self.X_full[child_indices],
                self.y_full[child_indices],
            )
            parent_node.children.append(child_node)
        logger.info(f"    Node {parent_node.node_id} split into {len(unique_labels)} children.")



    @staticmethod
    def _solver_formula(solver):
        try:
            return str(solver.program_.pretty())
        except Exception as exc:
            return f"Formula unavailable: {exc}"

    def _refit_final_leaf_solvers(self, X, y):
        """Refit frozen final leaves on all outer-training samples.

        Tree topology, split rules and classifiers are not changed. Samples
        are assigned to the final strata by the frozen hard routing rules, and
        only the leaf expressions are trained again. Thus the fixed validation
        subset participates in this final estimation stage only after it has
        finished serving its structure-selection role.
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        if len(X) != len(y):
            raise ValueError(
                f"Final leaf refit length mismatch: X={len(X)}, y={len(y)}"
            )

        leaves = self._get_all_leaves()
        routed_leaf_ids = np.asarray(self.predict_leaf_ids(X), dtype=int)
        known_leaf_ids = {int(leaf.node_id) for leaf in leaves}
        observed_leaf_ids = set(np.unique(routed_leaf_ids).tolist())
        if not observed_leaf_ids.issubset(known_leaf_ids):
            raise RuntimeError(
                "Final hard routing produced unknown leaf ids: "
                f"{sorted(observed_leaf_ids.difference(known_leaf_ids))}"
            )

        assigned_total = 0
        records = []
        logger.info(
            ">>> Tree structure fixed. Re-fitting each final leaf expression "
            "with all outer-training samples routed to that stratum..."
        )
        for leaf in sorted(leaves, key=lambda value: int(value.node_id)):
            full_indices = np.flatnonzero(
                routed_leaf_ids == int(leaf.node_id)
            ).astype(int, copy=False)
            if full_indices.size < 10:
                raise RuntimeError(
                    f"Final leaf Node {leaf.node_id} has only "
                    f"{full_indices.size} outer-training samples; refit "
                    "requires at least 10."
                )

            old_solver = leaf.solver
            old_formula = self._solver_formula(old_solver)
            refit_solver = self._create_solver(leaf.node_id)
            logger.info(
                f"  > Final leaf Node {leaf.node_id}: refit with "
                f"{full_indices.size} outer-training samples."
            )
            try:
                self._train_solver(
                    refit_solver,
                    X[full_indices],
                    y[full_indices],
                )
            except Exception as exc:
                leaf.solver = old_solver
                raise RuntimeError(
                    f"Final full-sample refit failed for leaf Node "
                    f"{leaf.node_id}; the fold is not considered complete."
                ) from exc

            leaf.solver = refit_solver
            leaf.sample_indices = full_indices.copy()
            leaf.solver_training_indices = full_indices.copy()
            leaf.performance = self._evaluate_solver(
                refit_solver,
                X[full_indices],
                y[full_indices],
            )
            leaf.is_frozen = True
            n_fixed_validation = int(
                np.intersect1d(
                    full_indices,
                    self._fixed_root_validation_indices,
                ).size
            )
            records.append(
                {
                    "node_id": int(leaf.node_id),
                    "n_outer_training_samples": int(full_indices.size),
                    "n_fixed_training_samples": int(
                        full_indices.size - n_fixed_validation
                    ),
                    "n_fixed_validation_samples": n_fixed_validation,
                    "routing": "frozen_tree_hard_stratum_assignment",
                    "old_formula": old_formula,
                    "refit_formula": self._solver_formula(refit_solver),
                    "refit_mse_on_leaf_outer_training_samples": float(
                        leaf.performance
                    ),
                }
            )
            assigned_total += int(full_indices.size)

        if assigned_total != len(X):
            raise RuntimeError(
                f"Final leaf refit assigned {assigned_total}/{len(X)} "
                "outer-training samples."
            )
        self.sbcv_leaf_refit_records = records
        logger.info(
            f">>> Final leaf refit complete: {len(records)} leaves, "
            f"{assigned_total} outer-training samples used exactly once."
        )



    def _predict_with_tree_routing(self, X):
        """Predict by deterministic routing along local discriminator outputs.

        Args:
            X (array-like): Input samples.

        Returns:
            np.ndarray: Predicted values for all samples.

        Raises:
            Exception: Propagated from discriminator or solver inference.
        """
        predictions = np.zeros(len(X), dtype=float)
        handled_mask = np.zeros(len(X), dtype=bool)
        stack = [(self.root, np.arange(len(X), dtype=int))]
        while stack:
            node, sample_indices = stack.pop()
            if sample_indices.size == 0:
                continue
            if node.is_leaf or node.discriminator is None or not node.children:
                node_pred = node.solver.predict(self._select_continuous_features(X[sample_indices]))
                predictions[sample_indices] = np.asarray(node_pred).ravel()
                handled_mask[sample_indices] = True
                continue
            child_assignments = np.asarray(node.discriminator.predict(X[sample_indices]))
            if np.issubdtype(child_assignments.dtype, np.floating):
                child_assignments = np.rint(child_assignments).astype(int)
            else:
                child_assignments = child_assignments.astype(int)
            child_map = {}
            for idx, child in enumerate(node.children):
                if child.split_label is None:
                    child_map[idx] = child
                else:
                    child_map[int(child.split_label)] = child
            for label in np.unique(child_assignments):
                label_mask = child_assignments == label
                routed_indices = sample_indices[label_mask]
                if routed_indices.size == 0:
                    continue
                child_node = child_map.get(int(label))
                if child_node is None:
                    fallback_pred = node.solver.predict(self._select_continuous_features(X[routed_indices]))
                    predictions[routed_indices] = np.asarray(fallback_pred).ravel()
                    handled_mask[routed_indices] = True
                else:
                    stack.append((child_node, routed_indices))
        if np.any(~handled_mask):
            fallback_pred = self.root.solver.predict(self._select_continuous_features(X[~handled_mask]))
            predictions[~handled_mask] = np.asarray(fallback_pred).ravel()
        return predictions

    def predict_leaf_ids(self, X):
        """Route samples and return terminal leaf node ids.

        Args:
            X (array-like): One sample ``(n_features,)`` or many
                ``(n_samples, n_features)``.

        Returns:
            np.ndarray: Integer node ids for terminal routed leaves.

        Raises:
            Exception: Propagated from discriminator inference.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        leaf_ids = np.full(len(X), -1, dtype=int)
        handled_mask = np.zeros(len(X), dtype=bool)
        stack = [(self.root, np.arange(len(X), dtype=int))]

        while stack:
            node, sample_indices = stack.pop()
            if sample_indices.size == 0:
                continue

            if node.is_leaf or node.discriminator is None or not node.children:
                leaf_ids[sample_indices] = int(node.node_id)
                handled_mask[sample_indices] = True
                continue

            child_assignments = np.asarray(node.discriminator.predict(X[sample_indices]))
            if np.issubdtype(child_assignments.dtype, np.floating):
                child_assignments = np.rint(child_assignments).astype(int)
            else:
                child_assignments = child_assignments.astype(int)

            child_map = {}
            for idx, child in enumerate(node.children):
                if child.split_label is None:
                    child_map[idx] = child
                else:
                    child_map[int(child.split_label)] = child

            for label in np.unique(child_assignments):
                label_mask = child_assignments == label
                routed_indices = sample_indices[label_mask]
                if routed_indices.size == 0:
                    continue

                child_node = child_map.get(int(label))
                if child_node is None:
                    leaf_ids[routed_indices] = int(node.node_id)
                    handled_mask[routed_indices] = True
                else:
                    stack.append((child_node, routed_indices))

        if np.any(~handled_mask):
            leaf_ids[~handled_mask] = int(self.root.node_id)
        return leaf_ids

    def predict(self, X):
        """Predict output values for input samples.

        Args:
            X (array-like): Input samples with shape ``(n_samples, n_features)``.

        Returns:
            np.ndarray: Predicted values.

        Raises:
            Exception: Propagated from discriminator or solver inference.
        """
        if self.hard_pred:
            return self._predict_with_tree_routing(X)
        else:
            return self._predict_with_tree_routing_prob_weighted(X)

    def evaluate_performance(self, X, y, sy=None):
        """Evaluate model performance with MSE and optional inverse transform.

        Args:
            X (array-like): Evaluation features.
            y (array-like): Evaluation targets.
            sy (Any | None): Optional scaler-like object exposing
                ``inverse_transform``.

        Returns:
            tuple[float, np.ndarray]:
                ``(mse, predictions)`` in scaled or original target space.

        Raises:
            Exception: Propagated from prediction, inverse transform, or metrics.
        """
        predictions = self.predict(X)
        if sy is None:
            predictions = np.nan_to_num(predictions, nan=0.0, posinf=1e10, neginf=-1e10)
            loss = mean_squared_error(y, predictions)
            return loss, predictions
        y_origin = sy.inverse_transform(y.reshape(-1, 1))
        pred_origin = sy.inverse_transform(predictions.reshape(-1, 1))
        pred_origin = np.nan_to_num(pred_origin, nan=0.0, posinf=1e10, neginf=-1e10)
        loss = mean_squared_error(y_origin, pred_origin)
        return loss, pred_origin.ravel()

    node_holdout_strategy = (
        "fixed_root_group_80_20_spatial_blocks_inherited"
    )

    def __init__(
        self,
        input_dim,
        base_solver_class=None,
        solver_kwargs=None,
        continuous_feature_indices=None,
        max_depth=10,
        min_samples_split=50,
        random_state=0,
        hard_pred=False,
        proposal_tree_max_depth=3,
        proposal_min_samples_leaf=10,
        min_path_fraction=0.05,
        min_path_q=0.0,
        enable_residual_restratification=True,
        relabel_margin=0.0,
        classifier_min_samples_leaf=10,
        min_classifier_balanced_accuracy=None,
        refit_final_leaves=False,
        fuzzy_membership_alpha=10.0,
        validation_hard_routing=False,
        path_selection_mode="root_to_leaf",
        path_stability_top_k=5,
    ):
        base_kwargs = dict(
            input_dim=input_dim,
            solver_kwargs=solver_kwargs,
            continuous_feature_indices=continuous_feature_indices,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            hard_pred=hard_pred,
        )
        if base_solver_class is not None:
            base_kwargs["base_solver_class"] = base_solver_class
        self._initialize_router(**base_kwargs)

        if int(proposal_tree_max_depth) < 1:
            raise ValueError("proposal_tree_max_depth must be at least 1.")
        if int(proposal_min_samples_leaf) < 1:
            raise ValueError("proposal_min_samples_leaf must be at least 1.")
        if not 0.0 <= float(min_path_fraction) < 0.5:
            raise ValueError("min_path_fraction must lie in [0, 0.5).")
        if not 0.0 <= float(min_path_q) <= 1.0:
            raise ValueError("min_path_q must lie in [0, 1].")
        if path_selection_mode not in PATH_SELECTION_MODES:
            raise ValueError(
                "path_selection_mode must be one of "
                f"{PATH_SELECTION_MODES}, got {path_selection_mode!r}."
            )
        if float(relabel_margin) < 0.0:
            raise ValueError("relabel_margin must be non-negative.")
        if int(path_stability_top_k) < 1:
            raise ValueError("path_stability_top_k must be at least 1.")
        if min_classifier_balanced_accuracy is not None and not (
            0.0 <= float(min_classifier_balanced_accuracy) <= 1.0
        ):
            raise ValueError(
                "min_classifier_balanced_accuracy must be None or lie in [0, 1]."
            )

        self.proposal_tree_max_depth = int(proposal_tree_max_depth)
        self.proposal_min_samples_leaf = int(proposal_min_samples_leaf)
        self.min_path_fraction = float(min_path_fraction)
        self.min_path_q = float(min_path_q)
        self.path_selection_mode = str(path_selection_mode)
        self.enable_residual_restratification = bool(
            enable_residual_restratification
        )
        self.relabel_margin = float(relabel_margin)
        self.classifier_min_samples_leaf = int(classifier_min_samples_leaf)
        self.min_classifier_balanced_accuracy = (
            None
            if min_classifier_balanced_accuracy is None
            else float(min_classifier_balanced_accuracy)
        )
        self.refit_final_leaves = bool(refit_final_leaves)
        self.fuzzy_membership_alpha = float(fuzzy_membership_alpha)
        self.validation_hard_routing = bool(validation_hard_routing)
        self.path_stability_top_k = int(path_stability_top_k)
        self.dso_early_stop_records: List[Dict[str, Any]] = []

    def _train_solver(self, solver, X_train, y_train):
        """Train one DSO solver and retain its success-based stop metadata.

        The implementation deliberately clears the fitted solver's bulky
        training state.  Capture the DSO ``success`` flag and trainer counters
        immediately after fitting, then keep only the program and a compact
        metadata dictionary.
        """
        logger.info(f"  > Start training solver. Train samples: {len(X_train)}")
        _clear_dso_cache()
        self._cleanup_memory()
        try:
            solver.fit(self._select_continuous_features(X_train), y_train)
            program = solver.program_
            evaluation = getattr(program, "evaluate", {})
            if not isinstance(evaluation, dict):
                evaluation = {}
            trainer = getattr(solver, "trainer", None)
            early_stopping_enabled = bool(
                getattr(trainer, "early_stopping", False)
            )
            success = bool(evaluation.get("success", False))
            metadata = {
                "early_stopping_enabled": early_stopping_enabled,
                "success": success,
                "early_stop_triggered": bool(
                    early_stopping_enabled and success
                ),
                "nmse_test": evaluation.get("nmse_test"),
                "nmse_test_noiseless": evaluation.get(
                    "nmse_test_noiseless"
                ),
                "nevals": getattr(trainer, "nevals", None),
                "n_samples_budget": getattr(trainer, "n_samples", None),
                "iteration": getattr(trainer, "iteration", None),
            }
            logger.info(
                f"  > Solver training finished. Equation: {program.pretty()}"
            )
            if metadata["early_stop_triggered"]:
                logger.info(
                    "  > DSO success-based early stopping triggered "
                    f"(nevals={metadata['nevals']}, "
                    f"budget={metadata['n_samples_budget']})."
                )

            solver.__dict__.clear()
            solver.program_ = program
            solver.dso_training_metadata_ = metadata
        finally:
            self._cleanup_memory(clear_program_cache=True)
        return solver

    @staticmethod
    def _solver_triggered_early_stop(solver):
        metadata = getattr(solver, "dso_training_metadata_", {})
        return bool(metadata.get("early_stop_triggered", False))

    def _get_split_tree_depth(self, node):
        """Return the residual proposal-tree depth budget for every node."""
        return self.proposal_tree_max_depth

    @staticmethod
    def _geodetector_q_components(values, category_labels):
        """Calculate q exactly as ``1 - sum(N_h sigma_h^2)/(N sigma^2)``.

        For a path candidate, category labels
        are 1 for samples satisfying the path and 0 for its complement.
        """
        values = np.asarray(values, dtype=float).ravel()
        category_labels = np.asarray(category_labels).ravel()
        if values.size != category_labels.size:
            raise ValueError(
                "q-statistic length mismatch: "
                f"values={values.size}, labels={category_labels.size}."
            )
        if values.size == 0:
            raise ValueError("q statistic requires at least one sample.")
        if not np.all(np.isfinite(values)):
            raise ValueError("q statistic values contain NaN or infinity.")
        if not np.all(np.isfinite(category_labels.astype(float))):
            raise ValueError("q statistic category labels are non-finite.")

        unique_labels = np.unique(category_labels)
        if not set(unique_labels.tolist()).issubset({0, 1}):
            raise ValueError(
                "Binary path q statistic requires category labels 0 and/or 1."
            )

        n_total = int(values.size)
        total_variance = float(np.var(values, ddof=0))
        sst = float(n_total * total_variance)
        ssw = 0.0
        stratum_components = []
        for label in unique_labels:
            stratum_values = values[category_labels == label]
            n_h = int(stratum_values.size)
            variance_h = float(np.var(stratum_values, ddof=0))
            contribution = float(n_h * variance_h)
            ssw += contribution
            stratum_components.append(
                {
                    "label": int(label),
                    "n": n_h,
                    "variance": variance_h,
                    "n_times_variance": contribution,
                }
            )

        if sst <= np.finfo(float).eps:
            q_value = 0.0
        else:
            q_value = 1.0 - ssw / sst
            q_value = float(np.clip(q_value, 0.0, 1.0))
        return {
            "q": q_value,
            "n": n_total,
            "total_variance": total_variance,
            "sst": sst,
            "ssw": float(ssw),
            "strata": tuple(stratum_components),
        }

    @classmethod
    def calculate_geodetector_q(cls, values, category_labels):
        """Public scalar geographical-detector q helper."""
        return float(cls._geodetector_q_components(values, category_labels)["q"])

    @staticmethod
    def _extract_leaf_paths(tree_model) -> List[Dict[str, Any]]:
        tree = tree_model.tree_
        paths: List[Dict[str, Any]] = []

        def visit(tree_node_id, conditions):
            left_id = int(tree.children_left[tree_node_id])
            right_id = int(tree.children_right[tree_node_id])
            if left_id == right_id:
                paths.append(
                    {
                        "leaf_id": int(tree_node_id),
                        "conditions": tuple(conditions),
                        "depth": int(len(conditions)),
                    }
                )
                return
            feature = int(tree.feature[tree_node_id])
            threshold = float(tree.threshold[tree_node_id])
            visit(left_id, conditions + [(feature, threshold, True)])
            visit(right_id, conditions + [(feature, threshold, False)])

        visit(0, [])
        return paths

    @classmethod
    def _extract_candidate_paths(
        cls,
        tree_model,
        path_selection_mode="root_to_leaf",
    ) -> List[Dict[str, Any]]:
        """Extract complete paths or complete paths plus internal subpaths.

        ``root_to_leaf`` preserves the original behavior. ``all_subpaths``
        additionally emits every contiguous, non-empty slice of the conditions
        along each root-to-leaf path. Each slice is evaluated as a standalone
        conjunction on all samples. Shared segments are de-duplicated.
        """
        if path_selection_mode not in PATH_SELECTION_MODES:
            raise ValueError(
                "path_selection_mode must be one of "
                f"{PATH_SELECTION_MODES}, got {path_selection_mode!r}."
            )

        leaf_paths = cls._extract_leaf_paths(tree_model)
        if path_selection_mode == "root_to_leaf":
            return [
                {
                    **path,
                    "path_kind": "root_to_leaf",
                    "path_start_depth": 1,
                    "path_end_depth": int(path["depth"]),
                    "source_leaf_ids": (int(path["leaf_id"]),),
                }
                for path in leaf_paths
            ]

        candidates_by_segment: Dict[Any, Dict[str, Any]] = {}
        for leaf_path in leaf_paths:
            full_conditions = tuple(leaf_path["conditions"])
            full_depth = len(full_conditions)
            for start in range(full_depth):
                for stop in range(start + 1, full_depth + 1):
                    conditions = tuple(full_conditions[start:stop])
                    start_depth = start + 1
                    end_depth = stop
                    segment_key = (conditions, start_depth, end_depth)
                    source_leaf_id = int(leaf_path["leaf_id"])
                    if segment_key in candidates_by_segment:
                        candidate = candidates_by_segment[segment_key]
                        candidate["source_leaf_ids"] = tuple(
                            sorted(
                                set(candidate["source_leaf_ids"])
                                | {source_leaf_id}
                            )
                        )
                        candidate["leaf_id"] = min(candidate["source_leaf_ids"])
                        continue

                    is_complete = start == 0 and stop == full_depth
                    candidates_by_segment[segment_key] = {
                        "leaf_id": source_leaf_id,
                        "source_leaf_ids": (source_leaf_id,),
                        "conditions": conditions,
                        "depth": int(len(conditions)),
                        "path_kind": (
                            "root_to_leaf" if is_complete else "internal_subpath"
                        ),
                        "path_start_depth": int(start_depth),
                        "path_end_depth": int(end_depth),
                    }

        return sorted(
            candidates_by_segment.values(),
            key=lambda path: (
                path["leaf_id"],
                path["path_start_depth"],
                path["path_end_depth"],
            ),
        )

    def _eligible_q_paths(self, proposal_tree, X_train, residuals):
        X_train = np.asarray(X_train)
        residuals = np.asarray(residuals, dtype=float).ravel()
        required_count = max(
            self.proposal_min_samples_leaf,
            int(np.ceil(self.min_path_fraction * len(X_train))),
        )
        candidates = []
        for path in self._extract_candidate_paths(
            proposal_tree,
            path_selection_mode=self.path_selection_mode,
        ):
            if path["depth"] < 1:
                continue
            path_router = BinaryPathRouter(
                path["conditions"],
                source_leaf_id=path["leaf_id"],
            )
            category_labels = path_router.predict(X_train)
            n_path = int(np.sum(category_labels == 1))
            n_complement = int(np.sum(category_labels == 0))
            if min(n_path, n_complement) < required_count:
                continue
            q_components = self._geodetector_q_components(
                residuals,
                category_labels,
            )
            if q_components["q"] + 1e-15 < self.min_path_q:
                continue
            candidates.append(
                {
                    **path,
                    "router": path_router,
                    "category_labels": category_labels.copy(),
                    "n_path": n_path,
                    "n_complement": n_complement,
                    "minimum_required_count": int(required_count),
                    "q_components": q_components,
                    "q": float(q_components["q"]),
                }
            )
        candidates.sort(
            key=lambda candidate: (
                -candidate["q"],
                candidate["depth"],
                candidate["path_start_depth"],
                candidate["leaf_id"],
            )
        )
        return candidates

    def _train_residual_tree(self, X_train, residuals, max_depth):
        tree_model = DecisionTreeRegressor(
            criterion="squared_error",
            max_depth=int(max_depth),
            min_samples_leaf=self.proposal_min_samples_leaf,
            random_state=42 + self.random_state,
        )
        tree_model.fit(np.asarray(X_train), np.asarray(residuals).ravel())
        return tree_model

    def _child_squared_error_matrix(self, solvers, X, y):
        if len(solvers) != 2:
            raise ValueError("Residual-q-path splitting requires exactly two solvers.")
        X = np.asarray(X)
        y = np.asarray(y, dtype=float).ravel()
        errors = []
        X_continuous = self._select_continuous_features(X)
        for solver in solvers:
            predictions = np.asarray(
                solver.predict(X_continuous),
                dtype=float,
            ).ravel()
            predictions = np.nan_to_num(
                predictions,
                nan=0.0,
                posinf=1e10,
                neginf=-1e10,
            )
            errors.append(np.square(predictions - y))
        return np.column_stack(errors)

    def _child_preference_labels(
        self,
        solvers,
        X,
        y,
        initial_labels=None,
    ):
        error_matrix = self._child_squared_error_matrix(solvers, X, y)
        labels = np.argmin(error_matrix, axis=1).astype(int)
        denominator = np.sum(error_matrix, axis=1) + np.finfo(float).eps
        relative_margin = np.abs(error_matrix[:, 0] - error_matrix[:, 1]) / denominator
        if initial_labels is not None and self.relabel_margin > 0.0:
            initial_labels = self._coerce_route_labels(initial_labels)
            ambiguous = relative_margin < self.relabel_margin
            labels[ambiguous] = initial_labels[ambiguous]
        return labels, relative_margin

    def _train_restratification_classifier(
        self,
        X,
        labels,
        max_depth,
        sample_weight=None,
    ):
        labels = self._coerce_route_labels(labels)
        if len(np.unique(labels)) < 2:
            return None
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float).ravel()
            if sample_weight.size != labels.size:
                raise ValueError(
                    "Classification sample-weight length mismatch: "
                    f"weights={sample_weight.size}, labels={labels.size}."
                )
            if not np.all(np.isfinite(sample_weight)):
                raise ValueError("Classification sample weights must be finite.")
            if np.any(sample_weight < 0.0):
                raise ValueError(
                    "Classification sample weights must be non-negative."
                )
            if float(np.sum(sample_weight)) <= np.finfo(float).eps:
                return None
        classifier = DecisionTreeClassifier(
            criterion="gini",
            max_depth=max(1, int(max_depth)),
            min_samples_leaf=self.classifier_min_samples_leaf,
            class_weight="balanced",
            random_state=42 + self.random_state,
        )
        classifier.fit(
            np.asarray(X),
            labels,
            sample_weight=sample_weight,
        )
        return classifier

    @staticmethod
    def _weight_summary(weights):
        weights = np.asarray(weights, dtype=float).ravel()
        if weights.size == 0:
            return {
                "n": 0,
                "n_positive": 0,
                "minimum": np.nan,
                "median": np.nan,
                "mean": np.nan,
                "maximum": np.nan,
            }
        return {
            "n": int(weights.size),
            "n_positive": int(np.sum(weights > 0.0)),
            "minimum": float(np.min(weights)),
            "median": float(np.median(weights)),
            "mean": float(np.mean(weights)),
            "maximum": float(np.max(weights)),
        }

    def _binary_path_coverage(self, labels, sample_indices):
        """Summarize sample and spatial-block coverage for a binary path."""
        labels = self._coerce_route_labels(labels)
        sample_indices = np.asarray(sample_indices, dtype=int).ravel()
        if labels.size != sample_indices.size:
            raise ValueError(
                "Path coverage length mismatch: "
                f"labels={labels.size}, indices={sample_indices.size}."
            )
        blocks = np.asarray(self._spatial_block_ids)[sample_indices]
        label_0_blocks = np.unique(blocks[labels == 0])
        label_1_blocks = np.unique(blocks[labels == 1])
        return {
            "n_samples": int(labels.size),
            "n_blocks": int(np.unique(blocks).size),
            "label_0_n_samples": int(np.sum(labels == 0)),
            "label_1_n_samples": int(np.sum(labels == 1)),
            "label_0_n_blocks": int(label_0_blocks.size),
            "label_1_n_blocks": int(label_1_blocks.size),
            "both_labels_n_blocks": int(
                np.intersect1d(label_0_blocks, label_1_blocks).size
            ),
        }

    def _finish_split_attempt(
        self,
        attempt_record,
        path_records,
        outcome,
        reason,
        selected_router_kind=None,
    ):
        """Persist one accepted or rejected split attempt and its top paths."""
        attempt_record["outcome"] = str(outcome)
        attempt_record["rejection_reason"] = (
            "" if outcome == "accepted" else str(reason)
        )
        attempt_record["selected_router_kind"] = (
            "" if selected_router_kind is None else str(selected_router_kind)
        )
        self.sbcv_split_attempt_records.append(attempt_record)
        for path_record in path_records:
            path_record["attempt_outcome"] = str(outcome)
            path_record["attempt_rejection_reason"] = (
                "" if outcome == "accepted" else str(reason)
            )
            path_record["selected_router_kind"] = (
                "" if selected_router_kind is None else str(selected_router_kind)
            )
            self.sbcv_path_candidate_records.append(path_record)

    @staticmethod
    def _safe_balanced_accuracy(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=int).ravel()
        y_pred = np.asarray(y_pred, dtype=int).ravel()
        if y_true.size == 0 or np.unique(y_true).size < 2:
            return np.nan
        return float(balanced_accuracy_score(y_true, y_pred))

    def _feature_scales_for_node(self, node):
        scale_indices = np.asarray(
            getattr(node, "inherited_train_indices", node.sample_indices),
            dtype=int,
        )
        values = np.asarray(self.X_full)[scale_indices]
        return np.std(values, axis=0) + 1e-8

    def _fuzzy_tree_probabilities(self, classifier, X, feature_scales):
        """Evaluate every split in a multi-depth classification tree softly."""
        X = np.asarray(X)
        n_samples = len(X)
        tree = classifier.tree_
        classes = np.asarray(classifier.classes_, dtype=int)
        output = {int(label): np.zeros(n_samples, dtype=float) for label in classes}
        stack = [(0, np.ones(n_samples, dtype=float))]
        while stack:
            tree_node_id, weights = stack.pop()
            left_id = int(tree.children_left[tree_node_id])
            right_id = int(tree.children_right[tree_node_id])
            if left_id == right_id:
                class_values = np.asarray(tree.value[tree_node_id]).reshape(-1)
                value_sum = float(np.sum(class_values))
                if value_sum <= 0.0:
                    class_probabilities = np.full(len(classes), 1.0 / len(classes))
                else:
                    class_probabilities = class_values / value_sum
                for class_index, label in enumerate(classes):
                    output[int(label)] += weights * class_probabilities[class_index]
                continue

            feature = int(tree.feature[tree_node_id])
            threshold = float(tree.threshold[tree_node_id])
            scale = max(float(feature_scales[feature]), 1e-8)
            distance = (X[:, feature] - threshold) / scale
            logits = np.clip(
                self.fuzzy_membership_alpha * distance,
                -60.0,
                60.0,
            )
            right_weight = 1.0 / (1.0 + np.exp(-logits))
            left_weight = 1.0 - right_weight
            stack.append((left_id, weights * left_weight))
            stack.append((right_id, weights * right_weight))

        total = np.zeros(n_samples, dtype=float)
        for probabilities in output.values():
            total += probabilities
        valid = total > np.finfo(float).eps
        for label in output:
            output[label][valid] /= total[valid]
        return output

    def _router_probability_by_label(self, router, X, node):
        X = np.asarray(X)
        feature_scales = self._feature_scales_for_node(node)
        if isinstance(router, BinaryPathRouter):
            probabilities = router.fuzzy_predict_proba(
                X,
                feature_scales,
                alpha=self.fuzzy_membership_alpha,
            )
            return {0: probabilities[:, 0], 1: probabilities[:, 1]}
        if hasattr(router, "tree_") and hasattr(router, "classes_"):
            return self._fuzzy_tree_probabilities(router, X, feature_scales)
        labels = self._predict_route_labels(router, X)
        return {
            int(label): (labels == int(label)).astype(float)
            for label in np.unique(labels)
        }

    def _predict_with_tree_routing_prob_weighted(self, X):
        """Predict with fuzzy traversal that supports multi-depth routers."""
        X = np.asarray(X)
        n_samples = len(X)
        predictions = np.zeros(n_samples, dtype=float)
        total_weight = np.zeros(n_samples, dtype=float)
        stack = [
            (
                self.root,
                np.arange(n_samples, dtype=int),
                np.ones(n_samples, dtype=float),
            )
        ]

        while stack:
            node, sample_indices, weights = stack.pop()
            if sample_indices.size == 0:
                continue
            active = np.flatnonzero(weights > 1e-18)
            sample_indices = sample_indices[active]
            weights = weights[active]
            if sample_indices.size == 0:
                continue

            if node.is_leaf or node.discriminator is None or not node.children:
                node_predictions = np.asarray(
                    node.solver.predict(
                        self._select_continuous_features(X[sample_indices])
                    ),
                    dtype=float,
                ).ravel()
                node_predictions = np.nan_to_num(
                    node_predictions,
                    nan=0.0,
                    posinf=1e10,
                    neginf=-1e10,
                )
                predictions[sample_indices] += node_predictions * weights
                total_weight[sample_indices] += weights
                continue

            child_map = {
                int(child.split_label if child.split_label is not None else index): child
                for index, child in enumerate(node.children)
            }
            label_probabilities = self._router_probability_by_label(
                node.discriminator,
                X[sample_indices],
                node,
            )
            covered_probability = np.zeros(sample_indices.size, dtype=float)
            for label, local_probabilities in label_probabilities.items():
                child = child_map.get(int(label))
                if child is None:
                    continue
                local_probabilities = np.asarray(local_probabilities, dtype=float)
                covered_probability += local_probabilities
                stack.append(
                    (
                        child,
                        sample_indices,
                        weights * local_probabilities,
                    )
                )

            missing_probability = np.clip(1.0 - covered_probability, 0.0, 1.0)
            if np.any(missing_probability > 1e-18):
                fallback = np.asarray(
                    node.solver.predict(
                        self._select_continuous_features(X[sample_indices])
                    ),
                    dtype=float,
                ).ravel()
                fallback = np.nan_to_num(
                    fallback,
                    nan=0.0,
                    posinf=1e10,
                    neginf=-1e10,
                )
                fallback_weights = weights * missing_probability
                predictions[sample_indices] += fallback * fallback_weights
                total_weight[sample_indices] += fallback_weights

        valid = total_weight > 1e-12
        predictions[valid] /= total_weight[valid]
        if np.any(~valid):
            fallback = self.root.solver.predict(
                self._select_continuous_features(X[~valid])
            )
            predictions[~valid] = np.asarray(fallback, dtype=float).ravel()
        return predictions

    def _predict_for_split_validation(self, X):
        """Predict a candidate tree using the configured validation routing."""
        if self.validation_hard_routing:
            return self._predict_with_tree_routing(X)
        return self._predict_with_tree_routing_prob_weighted(X)

    def _score_whole_tree_predictions(self, y_target, before, after):
        y_target = np.asarray(y_target, dtype=float).ravel()
        before = np.asarray(before, dtype=float).ravel()
        after = np.asarray(after, dtype=float).ravel()
        if not (
            y_target.size == before.size == after.size
            and np.all(np.isfinite(y_target))
            and np.all(np.isfinite(before))
            and np.all(np.isfinite(after))
        ):
            return {
                "n_validation_samples": int(y_target.size),
                "before_rmse": np.nan,
                "after_rmse": np.nan,
                "rmse_improvement": np.nan,
                "wilcoxon_p_value": np.nan,
                "passed": False,
            }
        before_loss = np.square(before - y_target)
        after_loss = np.square(after - y_target)
        try:
            _, p_value = wilcoxon(
                before_loss,
                after_loss,
                alternative="greater",
                zero_method="zsplit",
            )
        except ValueError:
            p_value = np.nan
        before_rmse = float(np.sqrt(np.mean(before_loss)))
        after_rmse = float(np.sqrt(np.mean(after_loss)))
        passed = bool(
            np.isfinite(p_value)
            and p_value <= 0.1
            and after_rmse < before_rmse
        )
        return {
            "scope": "fixed_root_validation_whole_tree",
            "inference": (
                "multi_depth_hard_routing"
                if self.validation_hard_routing
                else "multi_depth_fuzzy_membership_weighted"
            ),
            "n_validation_samples": int(y_target.size),
            "before_rmse": before_rmse,
            "after_rmse": after_rmse,
            "rmse_improvement": float(before_rmse - after_rmse),
            "wilcoxon_p_value": float(p_value),
            "passed": passed,
        }

    def _temporary_router_predictions(
        self,
        node,
        child_solvers,
        router,
        child_node_ids,
        X_target,
    ):
        all_labels = self._predict_route_labels(
            router,
            np.asarray(self.X_full)[node.sample_indices],
        )
        if set(np.unique(all_labels).tolist()) != {0, 1}:
            return None

        old_children = node.children
        old_discriminator = node.discriminator
        old_is_leaf = node.is_leaf
        try:
            node.children = []
            node.discriminator = None
            node.is_leaf = True
            self._execute_split(
                node,
                child_solvers,
                router,
                all_labels,
                child_node_ids,
            )
            return np.asarray(
                self._predict_for_split_validation(X_target),
                dtype=float,
            ).ravel()
        finally:
            node.children = old_children
            node.discriminator = old_discriminator
            node.is_leaf = old_is_leaf

    def _try_split_node(self, node, X, y):
        """Try one residual-q-path binary split at a current PS3R leaf."""
        attempt_id = len(self.sbcv_split_attempt_records) + 1
        attempt_record: Dict[str, Any] = {
            "attempt_id": int(attempt_id),
            "node_id": int(node.node_id),
            "node_depth": int(node.depth),
            "depth_iteration": int(self._sbcv_depth_iteration),
            "node_n_samples": int(len(node.sample_indices)),
            "path_selection_mode": self.path_selection_mode,
            "eligible_q_path_count": 0,
            "recorded_top_path_count": 0,
        }
        path_stability_records: List[Dict[str, Any]] = []
        logger.info(
            f"Attempting residual-q-path split for Node {node.node_id} "
            f"(Samples: {len(node.sample_indices)})..."
        )
        if len(node.sample_indices) < self.min_samples_split:
            node.is_frozen = True
            self._finish_split_attempt(
                attempt_record,
                path_stability_records,
                outcome="rejected",
                reason="node_sample_count_below_min_samples_split",
            )
            return False

        train_idx, val_idx = self._assert_fixed_holdout_state(node)
        attempt_record.update(
            {
                "proposal_train_n_samples": int(len(train_idx)),
                "proposal_validation_n_samples": int(len(val_idx)),
                "proposal_train_n_blocks": int(
                    np.unique(np.asarray(self._spatial_block_ids)[train_idx]).size
                ),
                "proposal_validation_n_blocks": int(
                    np.unique(np.asarray(self._spatial_block_ids)[val_idx]).size
                ),
            }
        )
        if (
            len(val_idx) < self.minimum_validation_samples
            or len(train_idx) < 2 * self.proposal_min_samples_leaf
        ):
            node.is_frozen = True
            self._finish_split_attempt(
                attempt_record,
                path_stability_records,
                outcome="rejected",
                reason="insufficient_inherited_training_or_validation_samples",
            )
            return False

        split_tree_depth = self._get_split_tree_depth(node)
        parent_predictions = np.asarray(
            node.solver.predict(
                self._select_continuous_features(np.asarray(X)[train_idx])
            ),
            dtype=float,
        ).ravel()
        parent_predictions = np.nan_to_num(
            parent_predictions,
            nan=0.0,
            posinf=1e10,
            neginf=-1e10,
        )
        parent_residuals = np.asarray(y, dtype=float).ravel()[train_idx] - parent_predictions

        proposal_tree = self._train_residual_tree(
            np.asarray(X)[train_idx],
            parent_residuals,
            max_depth=split_tree_depth,
        )
        leaf_ids_train = proposal_tree.apply(np.asarray(X)[train_idx])
        q_candidates = self._eligible_q_paths(
            proposal_tree,
            np.asarray(X)[train_idx],
            parent_residuals,
        )
        attempt_record["eligible_q_path_count"] = int(len(q_candidates))
        if not q_candidates:
            logger.info(
                f"Node {node.node_id}: no path satisfies q and sample constraints."
            )
            node.is_frozen = True
            self._finish_split_attempt(
                attempt_record,
                path_stability_records,
                outcome="rejected",
                reason="no_path_satisfied_q_and_sample_constraints",
            )
            return False

        validation_parent_predictions = np.asarray(
            node.solver.predict(
                self._select_continuous_features(np.asarray(X)[val_idx])
            ),
            dtype=float,
        ).ravel()
        validation_parent_predictions = np.nan_to_num(
            validation_parent_predictions,
            nan=0.0,
            posinf=1e10,
            neginf=-1e10,
        )
        validation_parent_residuals = (
            np.asarray(y, dtype=float).ravel()[val_idx]
            - validation_parent_predictions
        )
        selected_path_validation_q_components = None
        for rank, candidate in enumerate(
            q_candidates[: self.path_stability_top_k],
            start=1,
        ):
            candidate_validation_labels = self._predict_route_labels(
                candidate["router"],
                np.asarray(X)[val_idx],
            )
            candidate_validation_q = self._geodetector_q_components(
                validation_parent_residuals,
                candidate_validation_labels,
            )
            train_coverage = self._binary_path_coverage(
                candidate["category_labels"],
                train_idx,
            )
            validation_coverage = self._binary_path_coverage(
                candidate_validation_labels,
                val_idx,
            )
            train_q = float(candidate["q"])
            validation_q = float(candidate_validation_q["q"])
            path_record: Dict[str, Any] = {
                "attempt_id": int(attempt_id),
                "node_id": int(node.node_id),
                "node_depth": int(node.depth),
                "depth_iteration": int(self._sbcv_depth_iteration),
                "train_q_rank": int(rank),
                "is_selected_train_q_path": bool(rank == 1),
                "leaf_id": int(candidate["leaf_id"]),
                "source_leaf_ids": tuple(
                    int(value) for value in candidate["source_leaf_ids"]
                ),
                "path_depth": int(candidate["depth"]),
                "path_kind": str(candidate["path_kind"]),
                "path_start_depth": int(candidate["path_start_depth"]),
                "path_end_depth": int(candidate["path_end_depth"]),
                "path_rule": candidate["router"].rule_text(),
                "train_q": train_q,
                "validation_q": validation_q,
                "validation_minus_train_q": validation_q - train_q,
                "validation_to_train_q_ratio": (
                    validation_q / train_q
                    if train_q > np.finfo(float).eps
                    else np.nan
                ),
            }
            path_record.update(
                {
                    f"train_{key}": value
                    for key, value in train_coverage.items()
                }
            )
            path_record.update(
                {
                    f"validation_{key}": value
                    for key, value in validation_coverage.items()
                }
            )
            path_stability_records.append(path_record)
            if rank == 1:
                selected_path_validation_q_components = candidate_validation_q

        attempt_record["recorded_top_path_count"] = int(
            len(path_stability_records)
        )

        selected_path = q_candidates[0]
        path_router = selected_path["router"]
        initial_train_labels = selected_path["category_labels"]
        initial_val_labels = self._predict_route_labels(
            path_router,
            np.asarray(X)[val_idx],
        )
        selected_train_coverage = self._binary_path_coverage(
            initial_train_labels,
            train_idx,
        )
        selected_validation_coverage = self._binary_path_coverage(
            initial_val_labels,
            val_idx,
        )
        attempt_record.update(
            {
                "selected_path_rule": path_router.rule_text(),
                "selected_path_kind": str(selected_path["path_kind"]),
                "selected_path_depth": int(selected_path["depth"]),
                "selected_path_train_q": float(selected_path["q"]),
                "selected_path_validation_q": float(
                    selected_path_validation_q_components["q"]
                ),
                "selected_path_validation_to_train_q_ratio": (
                    float(selected_path_validation_q_components["q"])
                    / float(selected_path["q"])
                    if float(selected_path["q"]) > np.finfo(float).eps
                    else np.nan
                ),
            }
        )
        attempt_record.update(
            {
                f"selected_path_train_{key}": value
                for key, value in selected_train_coverage.items()
            }
        )
        attempt_record.update(
            {
                f"selected_path_validation_{key}": value
                for key, value in selected_validation_coverage.items()
            }
        )
        logger.info(
            f"  Node {node.node_id}: selected {selected_path['path_kind']} "
            f"from source leaf/leaves {selected_path['source_leaf_ids']} "
            f"with {selected_path['depth']} condition(s), "
            f"q={selected_path['q']:.6f}, "
            f"counts=(0:{selected_path['n_complement']}, "
            f"1:{selected_path['n_path']})."
        )

        child_solvers = []
        child_node_ids = []
        child_training_indices = []
        for label in (0, 1):
            child_train_indices = train_idx[initial_train_labels == label]
            child_node_id = self._get_next_node_id()
            child_solver = self._create_solver(child_node_id)
            logger.info(
                f"    Training q-path child {label} with "
                f"{len(child_train_indices)} samples..."
            )
            self._train_solver(
                child_solver,
                np.asarray(X)[child_train_indices],
                np.asarray(y)[child_train_indices],
            )
            child_solvers.append(child_solver)
            child_node_ids.append(child_node_id)
            child_training_indices.append(child_train_indices.copy())

        early_stop_labels = [
            label
            for label, solver in enumerate(child_solvers)
            if self._solver_triggered_early_stop(solver)
        ]
        if early_stop_labels:
            if len(early_stop_labels) == 1:
                winner_idx = int(early_stop_labels[0])
                selection_reason = "only_early_stopped_child"
            else:
                parent_train_X = np.asarray(X)[train_idx]
                parent_train_y = np.asarray(y)[train_idx]
                candidate_errors = {
                    int(label): float(
                        self._evaluate_solver(
                            child_solvers[label],
                            parent_train_X,
                            parent_train_y,
                        )
                    )
                    for label in early_stop_labels
                }
                winner_idx = min(
                    early_stop_labels,
                    key=lambda label: (candidate_errors[int(label)], int(label)),
                )
                winner_idx = int(winner_idx)
                selection_reason = "lowest_parent_training_mse_among_early_stopped"

            winner_solver = child_solvers[winner_idx]
            winner_metadata = dict(
                getattr(winner_solver, "dso_training_metadata_", {})
            )
            logger.info(
                f"Node {node.node_id}: child solver {winner_idx} triggered "
                "DSO success-based early stopping. Replacing the parent "
                "function and cancelling the split without validation."
            )
            node.solver = winner_solver
            node.solver_training_indices = child_training_indices[
                winner_idx
            ].copy()
            node.is_frozen = True
            node.is_leaf = True
            self.node_counter -= len(child_node_ids)
            attempt_record.update(
                {
                    "early_stop_intercepted": True,
                    "early_stop_child_labels": tuple(
                        int(label) for label in early_stop_labels
                    ),
                    "early_stop_winner_label": int(winner_idx),
                    "early_stop_selection_reason": selection_reason,
                    "parent_solver_replaced": True,
                    "replacement_solver_training_n_samples": int(
                        len(child_training_indices[winner_idx])
                    ),
                    "single_class_intercepted": False,
                    "single_class_winner_label": None,
                }
            )
            self.dso_early_stop_records.append(
                {
                    "event": "child_promoted_to_parent",
                    "node_id": int(node.node_id),
                    "node_depth": int(node.depth),
                    "child_label": int(winner_idx),
                    "child_node_id": int(child_node_ids[winner_idx]),
                    "child_training_n_samples": int(
                        len(child_training_indices[winner_idx])
                    ),
                    "selection_reason": selection_reason,
                    **winner_metadata,
                }
            )
            self._finish_split_attempt(
                attempt_record,
                path_stability_records,
                outcome="rejected",
                reason=(
                    "child_dso_early_stopping_"
                    "parent_solver_replaced_without_validation"
                ),
            )
            del child_solvers
            self._cleanup_memory(clear_program_cache=True)
            return False

        attempt_record.update(
            {
                "early_stop_intercepted": False,
                "early_stop_child_labels": tuple(),
                "early_stop_winner_label": None,
                "early_stop_selection_reason": "",
            }
        )

        preference_train_labels, train_margins = self._child_preference_labels(
            child_solvers,
            np.asarray(X)[train_idx],
            np.asarray(y)[train_idx],
            initial_labels=initial_train_labels,
        )
        if len(np.unique(preference_train_labels)) == 1:
            winner_idx = int(preference_train_labels[0])
            winner_solver = child_solvers[winner_idx]
            logger.info(
                f"Node {node.node_id}: all inherited training samples prefer "
                f"child solver {winner_idx}. Replacing the parent function "
                "and cancelling the split."
            )
            node.solver = winner_solver
            node.solver_training_indices = child_training_indices[
                winner_idx
            ].copy()
            node.is_frozen = True
            node.is_leaf = True
            self.node_counter -= len(child_node_ids)
            attempt_record.update(
                {
                    "single_class_intercepted": True,
                    "single_class_winner_label": int(winner_idx),
                    "parent_solver_replaced": True,
                    "replacement_solver_training_n_samples": int(
                        len(child_training_indices[winner_idx])
                    ),
                }
            )
            self._finish_split_attempt(
                attempt_record,
                path_stability_records,
                outcome="rejected",
                reason=(
                    "all_training_samples_preferred_one_child_"
                    "parent_solver_replaced"
                ),
            )
            del child_solvers
            self._cleanup_memory(clear_program_cache=True)
            return False

        attempt_record.update(
            {
                "single_class_intercepted": False,
                "single_class_winner_label": None,
                "parent_solver_replaced": False,
                "replacement_solver_training_n_samples": 0,
            }
        )
        preference_val_labels, validation_margins = self._child_preference_labels(
            child_solvers,
            np.asarray(X)[val_idx],
            np.asarray(y)[val_idx],
            initial_labels=initial_val_labels,
        )

        classification_sample_weight = None
        classification_weight_summary = self._weight_summary(
            classification_sample_weight
            if classification_sample_weight is not None
            else np.ones(len(preference_train_labels), dtype=float)
        )
        attempt_record.update(
            {
                "classification_sample_weight_enabled": bool(
                    classification_sample_weight is not None
                ),
                "classification_sample_weight_formula": (
                    "abs(E0-E1)/(E0+E1+eps)"
                    if classification_sample_weight is not None
                    else "uniform_weight_1"
                ),
                **{
                    f"classification_sample_weight_{key}": value
                    for key, value in classification_weight_summary.items()
                },
            }
        )

        classification_tree = None
        if self.enable_residual_restratification:
            classification_tree = self._train_restratification_classifier(
                np.asarray(X)[train_idx],
                preference_train_labels,
                max_depth=selected_path["depth"],
                sample_weight=classification_sample_weight,
            )

        fixed_validation_X = np.asarray(X)[self._fixed_root_validation_indices]
        fixed_validation_y = np.asarray(y, dtype=float).ravel()[
            self._fixed_root_validation_indices
        ]
        whole_tree_before = np.asarray(
            self._predict_for_split_validation(fixed_validation_X),
            dtype=float,
        ).ravel()

        router_candidates = [("q_path", path_router)]
        classifier_train_balanced_accuracy = np.nan
        classifier_validation_balanced_accuracy = np.nan
        classifier_reliability_passed = True
        if classification_tree is not None:
            classifier_train_prediction = self._predict_route_labels(
                classification_tree,
                np.asarray(X)[train_idx],
            )
            classifier_validation_prediction = self._predict_route_labels(
                classification_tree,
                np.asarray(X)[val_idx],
            )
            classifier_train_balanced_accuracy = self._safe_balanced_accuracy(
                preference_train_labels,
                classifier_train_prediction,
            )
            classifier_validation_balanced_accuracy = self._safe_balanced_accuracy(
                preference_val_labels,
                classifier_validation_prediction,
            )
            if self.min_classifier_balanced_accuracy is not None:
                classifier_reliability_passed = bool(
                    np.isfinite(classifier_validation_balanced_accuracy)
                    and classifier_validation_balanced_accuracy
                    >= self.min_classifier_balanced_accuracy
                )
            if classifier_reliability_passed:
                router_candidates.append(
                    ("residual_restratification_classifier", classification_tree)
                )

        candidate_validations = []
        for router_kind, router in router_candidates:
            after_predictions = self._temporary_router_predictions(
                node,
                child_solvers,
                router,
                child_node_ids,
                fixed_validation_X,
            )
            if after_predictions is None:
                continue
            metrics = self._score_whole_tree_predictions(
                fixed_validation_y,
                whole_tree_before,
                after_predictions,
            )
            metrics["router_kind"] = router_kind
            metrics["router"] = router
            candidate_validations.append(metrics)
            logger.info(
                f"    Router {router_kind}: before RMSE="
                f"{metrics['before_rmse']:.6f}, after RMSE="
                f"{metrics['after_rmse']:.6f}, p="
                f"{metrics['wilcoxon_p_value']:.6g}, "
                f"passed={metrics['passed']}."
            )

        for router_kind in (
            "q_path",
            "residual_restratification_classifier",
        ):
            metrics = next(
                (
                    value
                    for value in candidate_validations
                    if value["router_kind"] == router_kind
                ),
                None,
            )
            prefix = (
                "q_path_validation"
                if router_kind == "q_path"
                else "classification_tree_validation"
            )
            attempt_record[f"{prefix}_available"] = bool(metrics is not None)
            if metrics is not None:
                attempt_record.update(
                    {
                        f"{prefix}_n_samples": int(
                            metrics["n_validation_samples"]
                        ),
                        f"{prefix}_before_rmse": float(metrics["before_rmse"]),
                        f"{prefix}_after_rmse": float(metrics["after_rmse"]),
                        f"{prefix}_rmse_improvement": float(
                            metrics["rmse_improvement"]
                        ),
                        f"{prefix}_wilcoxon_p_value": float(
                            metrics["wilcoxon_p_value"]
                        ),
                        f"{prefix}_passed": bool(metrics["passed"]),
                    }
                )
        attempt_record.update(
            {
                "classification_tree_available": bool(
                    classification_tree is not None
                ),
                "classification_train_balanced_accuracy": float(
                    classifier_train_balanced_accuracy
                ),
                "classification_validation_balanced_accuracy": float(
                    classifier_validation_balanced_accuracy
                ),
                "classification_reliability_passed": bool(
                    classifier_reliability_passed
                ),
            }
        )

        passing_candidates = [
            metrics for metrics in candidate_validations if metrics["passed"]
        ]
        if not passing_candidates:
            logger.info(
                f"Node {node.node_id}: neither q-path nor eligible "
                "re-stratification routing improved independent validation."
            )
            self.node_counter -= len(child_node_ids)
            node.is_frozen = True
            self._finish_split_attempt(
                attempt_record,
                path_stability_records,
                outcome="rejected",
                reason="no_router_passed_fixed_spatial_validation",
            )
            del child_solvers
            self._cleanup_memory(clear_program_cache=True)
            return False

        passing_candidates.sort(
            key=lambda metrics: (
                metrics["after_rmse"],
                0 if metrics["router_kind"] == "q_path" else 1,
            )
        )
        selected_validation = passing_candidates[0]
        selected_router = selected_validation.pop("router")
        selected_router_kind = selected_validation["router_kind"]
        for metrics in candidate_validations:
            metrics.pop("router", None)

        all_routed_labels = self._predict_route_labels(
            selected_router,
            np.asarray(X)[node.sample_indices],
        )
        train_routed_labels = self._predict_route_labels(
            selected_router,
            np.asarray(X)[train_idx],
        )
        validation_routed_labels = self._predict_route_labels(
            selected_router,
            np.asarray(X)[val_idx],
        )
        self._execute_split(
            node,
            child_solvers,
            selected_router,
            all_routed_labels,
            child_node_ids,
        )

        child_inherited_train_indices = [np.array([], dtype=int) for _ in range(2)]
        child_inherited_validation_indices = [
            np.array([], dtype=int) for _ in range(2)
        ]
        for child in node.children:
            label = int(child.split_label)
            child.inherited_train_indices = train_idx[
                train_routed_labels == label
            ].copy()
            child.inherited_validation_indices = val_idx[
                validation_routed_labels == label
            ].copy()
            child.solver_training_indices = child_training_indices[label].copy()
            child.fixed_validation_inherited = True
            child.parent_solver_refit_count = 0
            self._assert_fixed_holdout_state(child)
            child_inherited_train_indices[label] = (
                child.inherited_train_indices.copy()
            )
            child_inherited_validation_indices[label] = (
                child.inherited_validation_indices.copy()
            )

        selected_path_rule = path_router.rule_text()
        validation_q_components = selected_path_validation_q_components
        self._last_split_validation = dict(selected_validation)
        split_record: Dict[str, Any] = {
            "node_id": int(node.node_id),
            "node_depth": int(node.depth),
            "depth_iteration": int(self._sbcv_depth_iteration),
            "split_order": len(self.sbcv_split_records) + 1,
            "node_sample_indices": np.asarray(node.sample_indices).copy(),
            "proposal_train_indices": train_idx.copy(),
            "proposal_validation_indices": val_idx.copy(),
            "proposal_tree": proposal_tree,
            "proposal_tree_kind": type(proposal_tree).__name__,
            "proposal_target": "signed_parent_dso_residual",
            "path_selection_mode": self.path_selection_mode,
            "proposal_parent_residuals_train": parent_residuals.copy(),
            "proposal_leaf_ids_train": np.asarray(leaf_ids_train).copy(),
            "q_path_candidates": tuple(
                {
                    "leaf_id": int(candidate["leaf_id"]),
                    "source_leaf_ids": tuple(
                        int(value) for value in candidate["source_leaf_ids"]
                    ),
                    "depth": int(candidate["depth"]),
                    "path_kind": candidate["path_kind"],
                    "path_start_depth": int(candidate["path_start_depth"]),
                    "path_end_depth": int(candidate["path_end_depth"]),
                    "q": float(candidate["q"]),
                    "n_path": int(candidate["n_path"]),
                    "n_complement": int(candidate["n_complement"]),
                    "rule": candidate["router"].rule_text(),
                    "q_components": candidate["q_components"],
                }
                for candidate in q_candidates
            ),
            "selected_q_path_leaf_id": int(selected_path["leaf_id"]),
            "selected_q_path_source_leaf_ids": tuple(
                int(value) for value in selected_path["source_leaf_ids"]
            ),
            "selected_q_path_depth": int(selected_path["depth"]),
            "selected_q_path_kind": selected_path["path_kind"],
            "selected_q_path_start_depth": int(
                selected_path["path_start_depth"]
            ),
            "selected_q_path_end_depth": int(selected_path["path_end_depth"]),
            "selected_q_path_rule": selected_path_rule,
            "selected_q_path_train_q": float(selected_path["q"]),
            "selected_q_path_train_q_components": selected_path["q_components"],
            "selected_q_path_validation_q": float(validation_q_components["q"]),
            "selected_q_path_validation_q_components": validation_q_components,
            "selected_q_path_train_labels": initial_train_labels.copy(),
            "selected_q_path_validation_labels": initial_val_labels.copy(),
            "candidate_child_solvers": tuple(child_solvers),
            "candidate_child_node_ids": tuple(int(value) for value in child_node_ids),
            "candidate_child_training_indices": tuple(
                values.copy() for values in child_training_indices
            ),
            "classification_tree": (
                classification_tree if classification_tree is not None else path_router
            ),
            "residual_restratification_classifier": classification_tree,
            "classification_train_true_labels": preference_train_labels.copy(),
            "classification_validation_true_labels": preference_val_labels.copy(),
            "classification_train_relative_margin": train_margins.copy(),
            "classification_validation_relative_margin": validation_margins.copy(),
            "classification_sample_weight_enabled": bool(
                classification_sample_weight is not None
            ),
            "classification_sample_weight_formula": (
                "abs(E0-E1)/(E0+E1+eps)"
                if classification_sample_weight is not None
                else "uniform_weight_1"
            ),
            "classification_train_sample_weight": (
                np.asarray(classification_sample_weight, dtype=float).copy()
                if classification_sample_weight is not None
                else np.ones(len(preference_train_labels), dtype=float)
            ),
            "classification_sample_weight_summary": dict(
                classification_weight_summary
            ),
            "classification_train_balanced_accuracy": (
                classifier_train_balanced_accuracy
            ),
            "classification_validation_balanced_accuracy": (
                classifier_validation_balanced_accuracy
            ),
            "classification_reliability_passed": bool(
                classifier_reliability_passed
            ),
            "selected_router": selected_router,
            "selected_router_kind": selected_router_kind,
            "router_candidate_validations": tuple(
                dict(metrics) for metrics in candidate_validations
            ),
            "node_holdout_strategy": self.node_holdout_strategy,
            "fixed_root_training_blocks": tuple(
                np.asarray(self._fixed_root_train_blocks).tolist()
            ),
            "fixed_root_validation_blocks": tuple(
                np.asarray(self._fixed_root_validation_blocks).tolist()
            ),
            "n_fixed_root_training_blocks": int(
                len(self._fixed_root_train_blocks)
            ),
            "n_fixed_root_validation_blocks": int(
                len(self._fixed_root_validation_blocks)
            ),
            "node_inherited_train_indices": train_idx.copy(),
            "node_inherited_validation_indices": val_idx.copy(),
            "parent_solver_training_indices": np.asarray(
                node.solver_training_indices,
                dtype=int,
            ).copy(),
            "parent_solver_refit_at_current_node": False,
            "fixed_validation_inherited": True,
            "split_acceptance_scope": "fixed_root_validation_whole_tree",
            "split_acceptance_inference": (
                "multi_depth_hard_routing"
                if self.validation_hard_routing
                else "multi_depth_fuzzy_membership_weighted"
            ),
            "whole_tree_validation": dict(selected_validation),
            "child_inherited_train_indices": tuple(
                child_inherited_train_indices
            ),
            "child_inherited_validation_indices": tuple(
                child_inherited_validation_indices
            ),
        }
        node.sbcv_split_record = split_record
        self.sbcv_split_records.append(split_record)
        self._finish_split_attempt(
            attempt_record,
            path_stability_records,
            outcome="accepted",
            reason="",
            selected_router_kind=selected_router_kind,
        )
        self._sbcv_accepted_node_ids.append(int(node.node_id))
        self._record_growth_snapshot("accepted_split", activated_node=node)
        self._cleanup_memory(clear_program_cache=True)
        return True

    def _split_root_by_spatial_blocks(self, sample_indices, spatial_block_ids):
        """Return one reproducible split whose block sets are disjoint."""
        sample_indices = np.asarray(sample_indices, dtype=int).ravel()
        block_ids = np.asarray(spatial_block_ids)
        if block_ids.ndim != 1:
            raise ValueError(
                "spatial_block_ids must be a one-dimensional array with one "
                "block identifier per sample."
            )
        if block_ids.size != sample_indices.size:
            raise ValueError(
                "spatial_block_ids length mismatch: "
                f"samples={sample_indices.size}, block_ids={block_ids.size}."
            )
        if block_ids.size == 0:
            raise ValueError("Spatial-block validation requires samples.")

        if np.issubdtype(block_ids.dtype, np.number):
            if not np.all(np.isfinite(block_ids)):
                raise ValueError("spatial_block_ids contains NaN or infinity.")
        else:
            missing = np.fromiter(
                (
                    value is None
                    or (
                        isinstance(value, (float, np.floating))
                        and np.isnan(value)
                    )
                    for value in block_ids
                ),
                dtype=bool,
                count=block_ids.size,
            )
            if np.any(missing):
                raise ValueError("spatial_block_ids contains missing values.")

        unique_blocks = np.unique(block_ids)
        if unique_blocks.size < 2:
            raise ValueError(
                "Spatial-block validation requires at least two distinct "
                "blocks."
            )

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=self.root_validation_fraction,
            random_state=self.root_validation_seed,
        )
        train_positions, validation_positions = next(
            splitter.split(sample_indices, groups=block_ids)
        )
        train_indices = sample_indices[np.asarray(train_positions, dtype=int)]
        validation_indices = sample_indices[
            np.asarray(validation_positions, dtype=int)
        ]
        train_blocks = np.unique(block_ids[train_positions])
        validation_blocks = np.unique(block_ids[validation_positions])

        if train_indices.size == 0 or validation_indices.size == 0:
            raise RuntimeError(
                "Spatial-block split produced an empty training or validation "
                "subset."
            )
        if np.intersect1d(train_blocks, validation_blocks).size:
            raise RuntimeError(
                "Spatial-block split leakage: at least one block occurs in "
                "both training and validation."
            )
        assigned = np.concatenate([train_indices, validation_indices])
        if (
            assigned.size != sample_indices.size
            or np.unique(assigned).size != sample_indices.size
            or not np.array_equal(np.sort(assigned), np.sort(sample_indices))
        ):
            raise RuntimeError(
                "Spatial-block split did not assign every sample exactly once."
            )
        return train_indices, validation_indices, train_blocks, validation_blocks

    def fit(self, X, y, spatial_block_ids=None):
        """Fit PS3R using a fixed spatial-block validation holdout.

        ``spatial_block_ids`` must correspond to the outer-training rows in
        ``X`` and ``y``.  No outer-test block may be passed here.
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        if spatial_block_ids is None:
            raise ValueError(
                "spatial_block_ids is required; this model does not fall "
                "back to a random row-wise internal validation split."
            )
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional feature matrix.")
        if len(X) != len(y):
            raise ValueError(
                f"X/y length mismatch: X={len(X)}, y={len(y)}."
            )
        self.sbcv_split_records = []
        self.sbcv_split_attempt_records = []
        self.sbcv_path_candidate_records = []
        self.sbcv_growth_history = []
        self.sbcv_leaf_refit_records = []
        self.dso_early_stop_records = []
        self._sbcv_accepted_node_ids = []
        self._sbcv_depth_iteration = 0

        self.X_full = X
        self.y_full = y
        self.root = TreeNode(node_id=self._get_next_node_id())
        self.root.sample_indices = np.arange(len(X), dtype=int)
        (
            root_train_indices,
            root_validation_indices,
            root_train_blocks,
            root_validation_blocks,
        ) = self._split_root_by_spatial_blocks(
            self.root.sample_indices,
            spatial_block_ids,
        )
        self._spatial_block_ids = np.asarray(spatial_block_ids).copy()
        self._fixed_root_train_indices = np.asarray(root_train_indices, dtype=int)
        self._fixed_root_validation_indices = np.asarray(
            root_validation_indices,
            dtype=int,
        )
        self._fixed_root_train_blocks = np.asarray(root_train_blocks).copy()
        self._fixed_root_validation_blocks = np.asarray(
            root_validation_blocks
        ).copy()
        self.root.inherited_train_indices = self._fixed_root_train_indices.copy()
        self.root.inherited_validation_indices = (
            self._fixed_root_validation_indices.copy()
        )
        self.root.solver_training_indices = self._fixed_root_train_indices.copy()
        self.root.fixed_validation_inherited = True
        self.root.parent_solver_refit_count = 0
        self._assert_fixed_holdout_state(self.root)

        self.root.solver = self._create_solver(self.root.node_id)
        logger.info(
            ">>> Fitting root DSO on a fixed spatial-block split: "
            f"training blocks={len(self._fixed_root_train_blocks)}, "
            f"validation blocks={len(self._fixed_root_validation_blocks)}, "
            f"training rows={len(self._fixed_root_train_indices)}, "
            f"validation rows={len(self._fixed_root_validation_indices)}. "
            "Residual q-path search will use training residuals only."
        )
        self._train_solver(
            self.root.solver,
            X[self._fixed_root_train_indices],
            y[self._fixed_root_train_indices],
        )
        if self._solver_triggered_early_stop(self.root.solver):
            root_metadata = dict(
                getattr(self.root.solver, "dso_training_metadata_", {})
            )
            self.root.is_frozen = True
            self.dso_early_stop_records.append(
                {
                    "event": "root_frozen",
                    "node_id": int(self.root.node_id),
                    "node_depth": int(self.root.depth),
                    "solver_training_n_samples": int(
                        len(self._fixed_root_train_indices)
                    ),
                    **root_metadata,
                }
            )
            logger.info(
                "Root DSO triggered success-based early stopping; freezing "
                "the root and skipping all stratification without validation."
            )
        self.root.performance = self._evaluate_solver(self.root.solver, X, y)
        self._record_growth_snapshot("root_fitted")

        current_depth = 0
        while current_depth < self.max_depth:
            self._sbcv_depth_iteration = current_depth + 1
            leaves = self._get_leaves()
            if not leaves:
                break
            growth_occurred = False
            attempted_node_ids = []
            accepted_node_ids = []
            for leaf in leaves:
                if leaf.is_frozen:
                    continue
                attempted_node_ids.append(int(leaf.node_id))
                if self._try_split_node(leaf, X, y):
                    growth_occurred = True
                    accepted_node_ids.append(int(leaf.node_id))
            self._record_growth_snapshot(
                "layer_complete",
                layer_attempted_node_ids=attempted_node_ids,
                layer_accepted_node_ids=accepted_node_ids,
            )
            if not growth_occurred:
                break
            self._cleanup_memory(clear_program_cache=True)
            current_depth += 1

        if self.refit_final_leaves:
            self._refit_final_leaf_solvers(X, y)
            final_event = "final_model_with_leaf_refit"
            refit_description = "enabled"
        else:
            self.sbcv_leaf_refit_records = []
            final_event = "final_model_without_leaf_refit"
            refit_description = "disabled"
        logger.info(
            f"Training complete. Growth depth={current_depth}, "
            f"leaves={len(self._get_all_leaves())}, final leaf refit="
            f"{refit_description}."
        )
        self._record_growth_snapshot(final_event)
        return self

    def get_model_structure(self, feature_names: List[str] = None) -> Dict[str, Any]:
        """Export full multi-depth rules, including a selected q-path router."""
        if feature_names is None:
            feature_names = [f"x{i + 1}" for i in range(self.input_dim)]
        structure_info = {
            "tree_structure": [],
            "formulas": {},
            "raw_formulas": {},
        }

        def traverse(node, indent=0):
            node_info = (
                f"{'  ' * indent}Node {node.node_id} "
                f"(Depth {node.depth}, Samples {len(node.sample_indices)})"
            )
            if node.is_leaf:
                node_info += " [LEAF]"
            elif isinstance(node.discriminator, BinaryPathRouter):
                node_info += (
                    "\n"
                    f"{'  ' * (indent + 1)}IF "
                    f"{node.discriminator.rule_text(feature_names)} "
                    "THEN child 1 ELSE child 0"
                )
            elif node.discriminator is not None:
                try:
                    rules = export_text(
                        node.discriminator,
                        feature_names=feature_names,
                    )
                    rendered = "\n".join(
                        f"{'  ' * (indent + 1)}| {line}"
                        for line in rules.strip().split("\n")
                    )
                    node_info += f"\n{rendered}"
                except Exception as exc:
                    node_info += f" [Split condition unavailable: {exc}]"
            structure_info["tree_structure"].append(node_info)
            for child in node.children:
                traverse(child, indent + 1)

        traverse(self.root)
        for leaf in self._get_all_leaves():
            key = f"Node_{leaf.node_id}"
            try:
                formula = leaf.solver.program_.pretty()
                structure_info["raw_formulas"][key] = formula
                structure_info["formulas"][key] = formula
            except Exception as exc:
                structure_info["formulas"][key] = (
                    f"Error extracting formula: {exc}"
                )
        return structure_info


__all__ = [
    "PS3RModel",
    "BinaryPathRouter",
    "PATH_SELECTION_MODES",
    "TreeNode",
]
