import abc
import math
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from .supports_cluster_autoscaling import SupportsClusterAutoscaling
from ray.data._internal.execution.interfaces import ExecutionResources
from ray.data._internal.execution.resource_manager import ResourceManager

if TYPE_CHECKING:
    from ray.data._internal.execution.resource_manager import ResourceManager


class ProductivityCalculator(abc.ABC):
    """Base class for determining how "productive" an operator is."""

    @abc.abstractmethod
    def get_productivity(self, op: SupportsClusterAutoscaling) -> Optional[float]:
        """Calculate the productivity of an operator.

        Productivity can be computed in different ways. The only requirement is
        that the bottleneck operator is the one with the lowest productivity.

        Args:
            op: The operator to calculate the productivity of.

        Returns:
            The productivity of the operator, or `None` if productivity isn't defined.
        """
        ...


class NormalizedThroughputCalculator(ProductivityCalculator):
    """Calculate the productivity of an operator using a normalized throughput score.

    Each operator launches tasks that produce outputs at a steady rate and consume
    inputs from the previous operator. To compare different operators which consume and
    produce different numbers of blocks, this class converts each operator's rate to
    represent "sink operator outputs per second".

    This class estimates an operator’s overall throughput by first computing the optimal
    number of tasks and then multiplying by the normalized rate per task. The
    calculation assumes a flow model and solves a simple optimization problem to
    maximize throughput.

    This implementation makes the following assumptions:
    - Tasks produce blocks at a constant average rate.
    - Tasks produce and consume a constant number of blocks.
    - There are many tasks in flight, so the system behaves like a fluid pipeline
      (averages dominate, quantization effects are negligible).
    """

    def __init__(
        self, ops: List[SupportsClusterAutoscaling], resource_manager: "ResourceManager"
    ):
        assert all(isinstance(op, SupportsClusterAutoscaling) for op in ops)

        self._ops = ops
        self._resource_manager = resource_manager

    def get_productivity(self, op: SupportsClusterAutoscaling) -> Optional[float]:
        if op.completed():
            return None

        if op.metrics.num_output_blocks_per_task_s is None:
            return None

        global_limits = self._resource_manager.get_global_limits()
        if global_limits is None:
            return None

        # Calculate the optimal processor resource allocation assuming a simple flow
        # model.
        allocation = self._get_optimal_processor_allocation(op)

        # Calculate the maximum output rate, assuming the operator uses all of its
        # allocation.
        max_num_tasks = self._get_max_num_concurrent_tasks(op, allocation)

        num_outputs_per_s = max_num_tasks * op.metrics.num_output_blocks_per_task_s

        # Normalize the rate to represent the number of sink operator outputs per
        # second. This is necessary to account for batching and tasks that produce
        # multiple outputs.
        return num_outputs_per_s * self._get_normalization_factor(op)

    def _get_optimal_processor_allocation(
        self, op: SupportsClusterAutoscaling
    ) -> ExecutionResources:
        """Estimate an optimal processor allocation assuming a flow model.

        This method solves the optimization problem:

            max min (rate_i * num_tasks_i)
            subject to
                sum(num_tasks_i * num_cpus_i) <= total_num_cpus
                sum(num_tasks_i * num_gpus_i) <= total_num_gpus
                num_tasks_i >= 0
            where i = 1,...,num_ops

        In other words, this method maximizes the throughput of the bottleneck, subject
        to resource constraints. It doesn't consider backpressure.

        We use a distinct allocation for autoscaling decisions because the actual
        allocation might be noisy (e.g., resource usage oscillating between operators).
        """
        # TODO(@balaji): Refactor this method to avoid repeated computation, even though
        # it's not a bottleneck.
        assert op.metrics.num_output_blocks_per_task_s is not None

        valid_ops = [op for op in self._ops if op.metrics.num_output_blocks_per_task_s]
        op_index = valid_ops.index(op)

        global_limits = self._resource_manager.get_global_limits()
        rate_per_op = np.array(
            [
                op.metrics.num_output_blocks_per_task_s
                * self._get_normalization_factor(op)
                for op in valid_ops
            ]
        )
        assert np.isfinite(rate_per_op).all() and np.all(rate_per_op > 0), (
            op,
            rate_per_op,
        )
        num_cpus_per_op = np.array(
            [op.per_task_resource_allocation().cpu for op in valid_ops]
        )
        num_gpus_per_op = np.array(
            [op.per_task_resource_allocation().gpu for op in valid_ops]
        )

        # These represent the number of CPUs and GPUs (respectively) to produce
        # one block per second across the pipeline.
        num_cpus_per_unit_rate: float = np.sum(num_cpus_per_op / rate_per_op)
        num_gpus_per_unit_rate: float = np.sum(num_gpus_per_op / rate_per_op)

        # Maximum feasible throughput based on each resource.
        max_cpu_rate: float = (
            np.inf
            if num_cpus_per_unit_rate == 0
            else global_limits.cpu / num_cpus_per_unit_rate
        )
        max_gpu_rate: float = (
            np.inf
            if num_gpus_per_unit_rate == 0
            else global_limits.gpu / num_gpus_per_unit_rate
        )

        # Overall max throughput (bottleneck).
        optimal_rate: float = min(max_cpu_rate, max_gpu_rate)

        # If all operators don't use any logical resources, or if there aren't enough
        # resources to run the pipeline, then split the processor resources equally.
        # These edge cases are unlikely.
        if math.isinf(optimal_rate) or optimal_rate == 0:
            global_processor_limits = global_limits.copy(
                memory=0, object_store_memory=0
            )
            return global_processor_limits.scale(1 / len(valid_ops))

        optimal_num_tasks_per_op = optimal_rate / rate_per_op
        assert np.isfinite(optimal_num_tasks_per_op).all(), (
            op,
            optimal_num_tasks_per_op,
        )

        # Calculate what percent of resources each operator would use assuming each
        # operator uses only their optimal number of tasks.
        cpu_fraction_per_op = (optimal_num_tasks_per_op * num_cpus_per_op) / np.sum(
            optimal_num_tasks_per_op * num_cpus_per_op
        )
        gpu_fraction_per_op = (optimal_num_tasks_per_op * num_gpus_per_op) / np.sum(
            optimal_num_tasks_per_op * num_gpus_per_op
        )

        return ExecutionResources(
            cpu=global_limits.cpu * cpu_fraction_per_op[op_index],
            gpu=global_limits.gpu * gpu_fraction_per_op[op_index],
        )

    def _get_max_num_concurrent_tasks(
        self, op: SupportsClusterAutoscaling, allocation: ExecutionResources
    ) -> float:
        per_task_resources = op.per_task_resource_allocation()

        # Given the total allocation, calculate the maximum number of concurrently
        # running tasks. This logic doesn't consider object store memory.
        if per_task_resources.cpu:
            max_cpu_num_tasks = allocation.cpu / per_task_resources.cpu
        else:
            max_cpu_num_tasks = float("inf")

        if per_task_resources.gpu:
            max_gpu_num_tasks = allocation.gpu / per_task_resources.gpu
        else:
            max_gpu_num_tasks = float("inf")

        max_num_tasks = min(max_cpu_num_tasks, max_gpu_num_tasks)

        # Some operators enforce a maximum number of concurrent tasks. For example, if
        # the user specifies `concurrency` in `map_batches`.
        if op.max_task_concurrency() is not None:
            max_num_tasks = min(max_num_tasks, op.max_task_concurrency())

        return max_num_tasks

    def _get_normalization_factor(self, op: SupportsClusterAutoscaling) -> float:
        """Calculate the normalization factor for an operator.

        To compare different operators, which might consume and produce different counts
        of blocks, this method calculates a normalization factor. This is used to
        convert every operator's rate into a single value that represents the number of
        sink operator outputs per second.

        Example:

            Consider a pipeline: A -> B -> C (sink)

            - Operator A: produces 2 outputs per 1 input (ratio = 2.0)
            - Operator B: produces 3 outputs per 2 inputs (ratio = 1.5)
            - Operator C: produces 1 output per 1 input (ratio = 1.0)

            For operator A: normalization_factor = 2.0 * 1.5 * 1.0 = 3.0
            For operator B: normalization_factor = 1.5 * 1.0 = 1.5
            For operator C: normalization_factor = 1.0

            This means:
            - If A produces 10 blocks/sec, it contributes 10 * 3.0 = 30 sink outputs/sec
            - If B produces 20 blocks/sec, it contributes 20 * 1.5 = 30 sink outputs/sec
            - If C produces 30 blocks/sec, it contributes 30 * 1.0 = 30 sink outputs/sec

            All operators now have comparable productivity metrics in terms of
            final sink outputs per second.

        Args:
            op: The operator to calculate the normalization factor for.

        Returns:
            The normalization factor.
        """
        if not op.output_dependencies:
            return 1

        factor = 1
        while op.output_dependencies:
            assert len(op.output_dependencies) == 1, (op, len(op.output_dependencies))
            op = op.output_dependencies[0]

            if (
                op.metrics.average_num_outputs_per_task is None
                or op.metrics.average_num_inputs_per_task is None
            ):
                continue

            factor *= (
                op.metrics.average_num_outputs_per_task
                / op.metrics.average_num_inputs_per_task
            )

        return factor
