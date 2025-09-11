from dataclasses import fields

import ray
import ray.data.preprocessors as preproc_module
from ray._private.arrow_utils import get_pyarrow_version
from ray._private.ray_constants import env_bool
from ray.anyscale.data._internal.execution.callbacks.insert_issue_detectors import (
    IssueDetectionExecutionCallback,
)
from ray.anyscale.data._internal.logging import configure_anyscale_logging
from ray.anyscale.data._internal.logical.rules import (
    FuseRepartitionOutputBlocks,
    PredicatePushdown,
    ProjectionPushdown,
    PushdownCountFiles,
    RedundantMapTransformPruning,
)
from ray.anyscale.data._internal.logical.rules.combine_repartitions import (
    CombineRepartitions,
)
from ray.anyscale.data._internal.logical.rules.configure_map_task_memory import (
    ConfigureMapTaskMemoryWithProfiling,
)
from ray.anyscale.data._internal.logical.rules.map_fusion import (
    BatchesToBatchesMapTransformTuning,
    BatchesToRowsMapTransformPrunning,
)
from ray.anyscale.data.api.context_mixin import DataContextMixin
from ray.anyscale.data.api.dataset_mixin import DatasetMixin
from ray.anyscale.data.checkpoint.iterator_mixin import DataIteratorMixin
from ray.anyscale.data.preprocessors import (
    Categorizer,
    Chain,
    LabelEncoder,
    MultiHotEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    SimpleImputer,
    StandardScaler,
)
from ray.data._internal.execution.execution_callback import add_execution_callback
from ray.data._internal.execution.interfaces.op_runtime_metrics import (
    MetricsGroup,
    OpRuntimeMetrics,
    metric_property,
)
from ray.data._internal.logical.optimizers import (
    get_logical_ruleset,
    get_physical_ruleset,
)
from ray.data._internal.logical.rules.configure_map_task_memory import (
    ConfigureMapTaskMemoryUsingOutputSize,
)

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
    # Patch properties
    for name, method in dataclass_mixin_cls.__dict__.items():
        if isinstance(method, property):
            setattr(original_cls, name, method)


def _patch_default_execution_callbacks():
    add_execution_callback(
        IssueDetectionExecutionCallback(), ray.data.DataContext.get_current()
    )


def _patch_aggregations():
    from ray.anyscale.data.aggregate_vectorized import (
        MIN_PYARROW_VERSION_VECTORIZED_AGGREGATIONS,
    )

    # NOTE: For Arrow versions >= 14.0 (supporting type promotions) we override
    #       standard aggregations to use vectorized versions
    if get_pyarrow_version() >= MIN_PYARROW_VERSION_VECTORIZED_AGGREGATIONS:
        from ray.anyscale.data import aggregate_vectorized
        from ray.data import aggregate

        aggregate.Count = aggregate_vectorized.CountVectorized
        aggregate.Sum = aggregate_vectorized.SumVectorized
        aggregate.Min = aggregate_vectorized.MinVectorized
        aggregate.Max = aggregate_vectorized.MaxVectorized
        aggregate.AbsMax = aggregate_vectorized.AbsMaxVectorized
        aggregate.Quantile = aggregate_vectorized.QuantileVectorized
        aggregate.Unique = aggregate_vectorized.UniqueVectorized


def _patch_arrow_ops():
    """Patch arrow operations with optimized implementations."""
    try:
        from ray.anyscale.data._internal.arrow_ops.transform_pyarrow import (
            hash_partition_optimized,
        )
        from ray.data._internal.arrow_ops import transform_pyarrow

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


def apply_anyscale_patches():
    """Apply Anyscale-specific patches for Ray Data."""

    # Patch ray.data.Dataset
    _patch_class_with_mixin(ray.data.Dataset, DatasetMixin)
    _patch_class_with_dataclass_mixin(ray.data.DataContext, DataContextMixin)
    _patch_class_with_mixin(ray.data.DataIterator, DataIteratorMixin)

    if ANYSCALE_ENABLE_AGGREGATION_BASED_PREPROCESSORS:
        preproc_module.Chain = Chain
        preproc_module.SimpleImputer = SimpleImputer
        preproc_module.StandardScaler = StandardScaler
        preproc_module.OrdinalEncoder = OrdinalEncoder
        preproc_module.OneHotEncoder = OneHotEncoder
        preproc_module.MultiHotEncoder = MultiHotEncoder
        preproc_module.LabelEncoder = LabelEncoder
        preproc_module.Categorizer = Categorizer

    # Patch default aggregation implementations with more performant
    # vectorized versions
    _patch_aggregations()

    # Patch arrow operations with optimized implementations
    _patch_arrow_ops()

    # Patch observability metrics
    _patch_observability_metrics()

    _patch_default_execution_callbacks()

    logical_ruleset = get_logical_ruleset()
    logical_ruleset.add(PredicatePushdown)
    logical_ruleset.add(PushdownCountFiles)
    logical_ruleset.add(ProjectionPushdown)
    logical_ruleset.add(CombineRepartitions)

    physical_ruleset = get_physical_ruleset()
    physical_ruleset.add(RedundantMapTransformPruning)
    physical_ruleset.add(FuseRepartitionOutputBlocks)
    physical_ruleset.add(BatchesToRowsMapTransformPrunning)
    physical_ruleset.add(BatchesToBatchesMapTransformTuning)
    if ANYSCALE_MAP_TASK_MEMORY_CONFIGURATION_ENABLED:
        physical_ruleset.remove(ConfigureMapTaskMemoryUsingOutputSize)
        physical_ruleset.add(ConfigureMapTaskMemoryWithProfiling)

    configure_anyscale_logging()
