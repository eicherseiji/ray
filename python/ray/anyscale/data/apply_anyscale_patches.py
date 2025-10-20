from dataclasses import fields

import ray
from ray._private.arrow_utils import get_pyarrow_version
from ray._private.ray_constants import env_bool

ANYSCALE_ENABLE_AGGREGATION_BASED_PREPROCESSORS = env_bool(
    "ANYSCALE_ENABLE_AGGREGATION_BASED_PREPROCESSORS", True
)


ANYSCALE_MAP_TASK_MEMORY_CONFIGURATION_ENABLED = env_bool(
    "ANYSCALE_MAP_TASK_MEMORY_CONFIGURATION_ENABLED", False
)


def _patch_class_with_mixin(original_cls, mixin_cls):
    for name, method in mixin_cls.__dict__.items():
        if not name.startswith("__"):
            setattr(original_cls, name, method)


def _patch_class_with_dataclass_mixin(original_cls, dataclass_mixin_cls):
    # Patch fields
    # Create an instance of the dataclass in order to get default values.
    mixin_instance = dataclass_mixin_cls()
    for field in fields(dataclass_mixin_cls):
        setattr(original_cls, field.name, getattr(mixin_instance, field.name))
    # Patch properties and methods
    for name, method in dataclass_mixin_cls.__dict__.items():
        if isinstance(method, property) or callable(method):
            # Skip special methods and private methods starting with underscore
            if not name.startswith("_"):
                setattr(original_cls, name, method)


def _patch_default_execution_callbacks():
    from ...data._internal.execution.execution_callback import add_execution_callback

    from ._internal.execution.callbacks.insert_issue_detectors import (
        IssueDetectionExecutionCallback,
    )

    add_execution_callback(
        IssueDetectionExecutionCallback(), ray.data.context.DataContext.get_current()
    )


def _patch_aggregations():
    from .aggregate_vectorized import (
        MIN_PYARROW_VERSION_VECTORIZED_AGGREGATIONS,
    )

    # NOTE: For Arrow versions >= 14.0 (supporting type promotions) we override
    #       standard aggregations to use vectorized versions
    if get_pyarrow_version() >= MIN_PYARROW_VERSION_VECTORIZED_AGGREGATIONS:
        from . import aggregate_vectorized
        from ...data import aggregate

        aggregate.Count = aggregate_vectorized.CountVectorized
        aggregate.Sum = aggregate_vectorized.SumVectorized
        aggregate.Min = aggregate_vectorized.MinVectorized
        aggregate.Max = aggregate_vectorized.MaxVectorized
        aggregate.AbsMax = aggregate_vectorized.AbsMaxVectorized
        aggregate.Quantile = aggregate_vectorized.QuantileVectorized
        aggregate.Unique = aggregate_vectorized.UniqueVectorized


def _patch_map_transformations():
    """Patches ``MapTransformer`` implementation"""
    from ray.anyscale.data._internal.execution.operators.map_transformer import (
        OptimizedMapTransformer,
        OptimizedBlockMapTransformFn,
        OptimizedBatchMapTransformFn,
        OptimizedRowMapTransformFn,
    )

    from ray.data._internal.execution.operators import map_transformer

    map_transformer.MapTransformer = OptimizedMapTransformer
    map_transformer.BlockMapTransformFn = OptimizedBlockMapTransformFn
    map_transformer.BatchMapTransformFn = OptimizedBatchMapTransformFn
    map_transformer.RowMapTransformFn = OptimizedRowMapTransformFn


def _patch_preprocessors():
    if ANYSCALE_ENABLE_AGGREGATION_BASED_PREPROCESSORS:

        from ...data import preprocessors

        from .preprocessors import (
            Categorizer,
            Chain,
            LabelEncoder,
            MultiHotEncoder,
            OneHotEncoder,
            OrdinalEncoder,
            SimpleImputer,
            StandardScaler,
        )

        preprocessors.Chain = Chain
        preprocessors.SimpleImputer = SimpleImputer
        preprocessors.StandardScaler = StandardScaler
        preprocessors.OrdinalEncoder = OrdinalEncoder
        preprocessors.OneHotEncoder = OneHotEncoder
        preprocessors.MultiHotEncoder = MultiHotEncoder
        preprocessors.LabelEncoder = LabelEncoder
        preprocessors.Categorizer = Categorizer


def _patch_arrow_ops():
    """Patch arrow operations with optimized implementations."""
    try:
        from ._internal.arrow_ops.transform_pyarrow import (
            hash_partition_optimized,
        )
        from ...data._internal.arrow_ops import transform_pyarrow

        # Replace the hash_partition function with the optimized version
        transform_pyarrow.hash_partition = hash_partition_optimized
    except Exception:
        pass


def _patch_observability_metrics():
    """
    This function patches the OpRuntimeMetrics class to add custom metrics on the
    RayTurbo side.

    In particular, it adds counter metrics to track the number of detector issues.
    For rendering RayTurbo dashboard, these counters are indexed by timestamp so are
    performant to query across multiple datasets.

    We also persist the details of each issue as exported events. These details are not
    indexed by timestamp and are not performant to query across multiple datasets. We
    will only query these details at the operator level in RayTurbo dashboard.
    """
    from ...data._internal.execution.interfaces.op_runtime_metrics import (
        MetricsGroup,
        OpRuntimeMetrics,
        metric_property,
    )

    OpRuntimeMetrics._issue_detector_hanging = 0
    OpRuntimeMetrics._issue_detector_high_memory = 0

    @metric_property(
        description="Indicates if the operator is hanging.",
        metrics_group=MetricsGroup.MISC,
        internal_only=True,
    )
    def issue_detector_hanging(self) -> int:
        return self._issue_detector_hanging

    @metric_property(
        description="Indicates if the operator is using high memory.",
        metrics_group=MetricsGroup.MISC,
        internal_only=True,
    )
    def issue_detector_high_memory(self) -> int:
        return self._issue_detector_high_memory

    OpRuntimeMetrics.issue_detector_hanging = issue_detector_hanging
    OpRuntimeMetrics.issue_detector_high_memory = issue_detector_high_memory


def _add_optimization_rules():
    from ...data._internal.logical.optimizers import (
        get_logical_ruleset,
        get_physical_ruleset,
    )

    from ._internal.logical.rules.combine_repartitions import (
        CombineRepartitions,
    )
    from ._internal.logical.rules.inherit_target_batch_size import (
        FuseMapWithRepartitionRule,
    )
    from ._internal.logical.rules.combine_downloads import (
        CombineDownloads,
    )

    from ._internal.logical.rules import (
        PredicatePushdown,
        ProjectionPushdown,
        PushdownCountFiles,
    )

    # Logical optimization rules

    logical_ruleset = get_logical_ruleset()
    logical_ruleset.add(PredicatePushdown)
    logical_ruleset.add(PushdownCountFiles)
    logical_ruleset.add(ProjectionPushdown)
    logical_ruleset.add(CombineRepartitions)
    logical_ruleset.add(FuseMapWithRepartitionRule)
    logical_ruleset.add(CombineDownloads)

    from ._internal.logical.rules.configure_map_task_memory import (
        ConfigureMapTaskMemoryWithProfiling,
    )

    # Physical optimization rules

    from ...data._internal.logical.rules.configure_map_task_memory import (
        ConfigureMapTaskMemoryUsingOutputSize,
    )

    physical_ruleset = get_physical_ruleset()
    if ANYSCALE_MAP_TASK_MEMORY_CONFIGURATION_ENABLED:
        physical_ruleset.remove(ConfigureMapTaskMemoryUsingOutputSize)
        physical_ruleset.add(ConfigureMapTaskMemoryWithProfiling)


def _patch_hash_shuffle_operator():
    """Patch hash shuffle operator with RayTurbo-specific dependency checking."""
    from ray.data._internal.execution.operators.hash_shuffle import (
        HashShufflingOperatorBase,
    )
    from ._internal.util.dependencies import check_numba_for_hash_partitioning

    # Store the original __init__ method
    original_init = HashShufflingOperatorBase.__init__

    def patched_init(self, *args, **kwargs):
        # Call the original __init__
        original_init(self, *args, **kwargs)

        # Add RayTurbo-specific numba check
        check_numba_for_hash_partitioning()

    # Replace the __init__ method
    HashShufflingOperatorBase.__init__ = patched_init


def apply_anyscale_patches():
    """Apply Anyscale-specific patches for Ray Data.

    NOTE: Ordering of operations is important and reordering of these operations
          might have an effect.

    """

    from ._internal.logging import configure_anyscale_logging

    configure_anyscale_logging()

    # Patches ``MapTransformer`` and ``MapTransformFn``s
    _patch_map_transformations()

    # Patch observability metrics
    _patch_observability_metrics()

    # Patch Arrow operations with optimized implementations
    _patch_arrow_ops()

    # Patch default aggregation implementations with more performant
    # vectorized versions
    _patch_aggregations()

    # Patch preprocessors
    _patch_preprocessors()

    # Add RayTurbo-specific dependency checking for hash shuffle
    _patch_hash_shuffle_operator()

    # Add optimization rules
    _add_optimization_rules()

    from .api.context_mixin import DataContextMixin
    from .api.dataset_mixin import DatasetMixin
    from .checkpoint.iterator_mixin import DataIteratorMixin

    # Patch ...data.Dataset
    _patch_class_with_mixin(ray.data.dataset.Dataset, DatasetMixin)
    _patch_class_with_dataclass_mixin(ray.data.context.DataContext, DataContextMixin)
    _patch_class_with_mixin(ray.data.iterator.DataIterator, DataIteratorMixin)

    # Patch default execution callbacks
    _patch_default_execution_callbacks()
