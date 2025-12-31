from ray.anyscale.data._internal.util.cached_ray_internals import (
    get_local_ongoing_lineage_reconstruction_tasks,
)
from ray.data._internal.execution.interfaces import ExecutionResources
from ray.data._internal.execution.interfaces.physical_operator import (
    ReportsExtraResourceUsage,
)
from ray.data._internal.execution.operators.task_pool_map_operator import (
    TaskPoolMapOperator as OSSTaskPoolMapOperator,
)


class TaskPoolMapOperator(OSSTaskPoolMapOperator, ReportsExtraResourceUsage):
    def extra_resource_usage(self) -> ExecutionResources:
        """Returns resources occupied by lineage reconstruction tasks."""
        return self.incremental_resource_usage().scale(
            self._num_lineage_reconstruction_tasks()
        )

    def _num_lineage_reconstruction_tasks(self) -> int:
        # This method assumes the base class launches tasks with the
        # `PhysicalOperator._OPERATOR_ID_LABEL_KEY` label.
        task_infos = get_local_ongoing_lineage_reconstruction_tasks()
        return sum(
            num_tasks
            for task_info, num_tasks in task_infos
            if task_info.labels.get(self._OPERATOR_ID_LABEL_KEY) == self.id
        )
