from typing import Tuple

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
    def min_max_resource_requirements(
        self,
    ) -> Tuple[ExecutionResources, ExecutionResources]:
        # The base implementation only implements the min resource requirements.
        min_resource_usage, _ = super().min_max_resource_requirements()
        return min_resource_usage, self._max_resource_usage()

    def _max_resource_usage(self) -> ExecutionResources:
        num_cpus_per_task = self._ray_remote_args.get("num_cpus", 0)
        num_gpus_per_task = self._ray_remote_args.get("num_gpus", 0)
        memory_per_task = self._ray_remote_args.get("memory", 0)
        object_store_memory_per_task = (
            self._metrics.obj_store_mem_max_pending_output_per_task or 0
        )

        if self._inputs_complete:
            # If the operator has already received all input data, we know it won't
            # launch more tasks. So, we only need to reserve resources for the tasks
            # that are currently running.
            resources = ExecutionResources(
                cpu=num_cpus_per_task * self.num_active_tasks(),
                gpu=num_gpus_per_task * self.num_active_tasks(),
                memory=memory_per_task * self.num_active_tasks(),
                object_store_memory=object_store_memory_per_task
                * self.num_active_tasks(),
            )
        elif self._concurrency is not None:
            resources = ExecutionResources(
                cpu=num_cpus_per_task * self._concurrency,
                gpu=num_gpus_per_task * self._concurrency,
                memory=memory_per_task * self._concurrency,
                object_store_memory=object_store_memory_per_task * self._concurrency,
            )
        else:
            resources = ExecutionResources.for_limits()

        return resources

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
