import abc
import math
from logging import getLogger
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from .supports_cluster_autoscaling import SupportsClusterAutoscaling
from ray.data._internal.execution.interfaces import ExecutionResources
from ray.data._internal.execution.resource_manager import ResourceManager
from ray.util.metrics import Gauge

if TYPE_CHECKING:
    from ray.data._internal.execution.resource_manager import ResourceManager


logger = getLogger(__name__)
PRODUCTIVITY_RELATIVE_TOLERANCE = 0.01  # 1%


class BottleneckDetector(abc.ABC):
    """Base class for detecting the bottleneck operator in a pipeline."""

    @abc.abstractmethod
    def get_bottleneck(
        self, ops: List[SupportsClusterAutoscaling]
    ) -> Optional[SupportsClusterAutoscaling]:
        """Identify the bottleneck operator in the pipeline.

        The bottleneck operator is the one that limits overall pipeline throughput.

        Args:
            ops: The operators to analyze.

        Returns:
            The bottleneck operator, or `None` if no bottleneck can be identified.
        """
        ...


class NormalizedThroughputBottleneckDetector(BottleneckDetector):
    """Calculate the productivity of an operator using a normalized throughput score.

    Each operator launches tasks that produce outputs at a steady rate and consume
    inputs from the previous operator. To compare different operators which consume and
    produce different numbers of blocks, this class converts each operator's rate to
    represent "sink operator outputs per second".

    This class estimates an operator's overall throughput by first computing the optimal
    number of tasks and then multiplying by the normalized rate per task. The
    calculation assumes a flow model and solves a simple optimization problem to
    maximize throughput.

    This implementation makes the following assumptions:
    - Tasks produce blocks at a constant average rate.
    - Tasks produce and consume a constant number of blocks.
    - There are many tasks in flight, so the system behaves like a fluid pipeline
      (averages dominate, quantization effects are negligible).
    """

    def __init__(self, resource_manager: "ResourceManager", requester_id: str):
        self._resource_manager = resource_manager
        self._requester_id = requester_id
        self._productivity_gauge = Gauge(
            "data_productivity",
            "Productivity per operator",
            tag_keys=("requester", "operator"),
        )

    def get_bottleneck(
        self, ops: List[SupportsClusterAutoscaling]
    ) -> Optional[SupportsClusterAutoscaling]:
        assert all(isinstance(op, SupportsClusterAutoscaling) for op in ops)

        # Calculate the throughput-balanced processor resource allocation assuming a
        # simple flow model.
        allocations = self._get_throughput_balanced_processor_allocation(ops)
        productivities = {
            op: self._compute_max_normalized_output_rate(op, allocation)
            for op, allocation in allocations.items()
        }

        assert all(op in productivities for op in ops), (
            f"Must return producitivites for all specified operators, but the "
            f"following operators are missing: {set(ops) - set(productivities)}."
        )

        # Filter out operators with undefined productivity.
        defined_productivities = {
            op: productivity
            for op, productivity in productivities.items()
            if productivity is not None
        }

        # Emit Prometheus metrics for all operators with defined productivity.
        for op, score in defined_productivities.items():
            self._productivity_gauge.set(
                score,
                tags={"requester": self._requester_id, "operator": repr(op)},
            )

        # Return the bottleneck operator (the one with lowest productivity).
        if not defined_productivities:
            return None

        # If all productivity scores are approximately equal, then there is no
        # meaningful bottleneck. This happens on CPU-only clusters where the
        # throughput-balanced allocation distributes CPUs proportionally to each
        # operator's needs, resulting in nearly identical productivity scores.
        # In this case, we return the operator with the least object store memory rather
        # than picking an arbitrary operator based on floating-point noise. We use
        # object store memory as a heuristic because it's consistent with how we
        # select which operators to run in the streaming executor.
        scores = list(defined_productivities.values())
        if len(scores) > 1:
            if np.allclose(scores, max(scores), rtol=PRODUCTIVITY_RELATIVE_TOLERANCE):
                logger.debug(
                    "Productivity scores approximately equal (within tolerance). "
                    "No clear bottleneck, returning the operator with the least "
                    "object store memory."
                )

                def key(op: SupportsClusterAutoscaling) -> float:
                    return self._resource_manager.get_op_usage(op).object_store_memory

                return min(defined_productivities, key=key)

        return min(defined_productivities, key=defined_productivities.get)

    def _compute_max_normalized_output_rate(
        self, op: SupportsClusterAutoscaling, allocation: Optional[ExecutionResources]
    ):
        if op.has_completed():
            return None

        if op.metrics.num_output_blocks_per_task_s is None or allocation is None:
            return None

        # Calculate the maximum output rate, assuming the operator uses all of its
        # allocation.
        max_num_tasks = self._get_max_num_concurrent_tasks(op, allocation)
        num_outputs_per_s = max_num_tasks * op.metrics.num_output_blocks_per_task_s

        # Normalize the rate to represent the number of sink operator outputs per
        # second. This is necessary to account for batching and tasks that produce
        # multiple outputs.
        return num_outputs_per_s * self._get_normalization_factor(op)

    def _get_throughput_balanced_processor_allocation(
        self, ops: List[SupportsClusterAutoscaling]
    ) -> Dict[SupportsClusterAutoscaling, Optional[ExecutionResources]]:
        """Calculate a throughput-balanced processor allocation for an operator.

        This method solves an optimization problem to find the resource allocation that
        maximizes pipeline throughput by balancing resources across operators. The
        returned allocation represents this operator's share of the total cluster
        resources, proportional to what it would need in the optimal solution.

        The optimization problem:

            max min (rate_i * num_tasks_i)
            subject to
                sum(num_tasks_i * num_cpus_i) <= total_num_cpus
                sum(num_tasks_i * num_gpus_i) <= total_num_gpus
                num_tasks_i >= 0
            where i = 1,...,num_ops

        In other words, this method maximizes the throughput of the bottleneck operator,
        subject to CPU and GPU constraints.

        We use this calculated allocation for productivity scoring rather than the actual
        allocation because the actual allocation might be noisy (e.g., resource usage
        oscillating between operators) and lead to poor autoscaling decisions.

        Known limitation: this implementation doesn't consider backpressure. If an
        operator is backpressured, this class can overestimate the operator's
        productivity. This might not be an issue in practice, because if an operator
        gets backpressured, it probably isn't the bottleneck anyway.
        """
        valid_ops = [op for op in ops if op.metrics.num_output_blocks_per_task_s]

        if not valid_ops:
            return dict.fromkeys(ops)

        global_limits = self._resource_manager.get_global_limits()
        assert global_limits is not None, "`get_global_limits` should never return None"

        # How many normalized outputs does each operator produce per task per second?
        task_rate_per_op = np.array(
            [
                op.metrics.num_output_blocks_per_task_s
                * self._get_normalization_factor(op)
                for op in valid_ops
            ]
        )
        assert np.isfinite(task_rate_per_op).all() and np.all(task_rate_per_op > 0), (
            task_rate_per_op,
        )

        # How many CPUs and GPUs does each operator need per task?
        task_num_cpus_per_op = np.array(
            [op.per_task_resource_allocation().cpu for op in valid_ops]
        )
        task_num_gpus_per_op = np.array(
            [op.per_task_resource_allocation().gpu for op in valid_ops]
        )

        # How many CPUs and GPUs does the pipeline need to produce one block per second
        # across the pipeline?
        num_cpus_per_unit_rate: float = np.sum(task_num_cpus_per_op / task_rate_per_op)
        num_gpus_per_unit_rate: float = np.sum(task_num_gpus_per_op / task_rate_per_op)

        # Given the number of CPUs and GPUs in the cluster, what's the max throughput
        # assuming you only need one type of resource?
        max_throughput_limited_by_cpu: float = (
            np.inf
            if num_cpus_per_unit_rate == 0
            else global_limits.cpu / num_cpus_per_unit_rate
        )
        max_throughput_limited_by_gpu: float = (
            np.inf
            if num_gpus_per_unit_rate == 0
            else global_limits.gpu / num_gpus_per_unit_rate
        )

        # The bottleneck resource determines the optimal throughput.
        optimal_rate: float = min(
            max_throughput_limited_by_cpu, max_throughput_limited_by_gpu
        )

        # If all operators don't use any logical resources, or if there aren't enough
        # resources to run the pipeline, then don't return any allocations. These edge
        # cases are unlikely.
        if math.isinf(optimal_rate) or optimal_rate == 0:
            return dict.fromkeys(ops)

        optimal_num_tasks_per_op = optimal_rate / task_rate_per_op
        assert np.isfinite(optimal_num_tasks_per_op).all(), (optimal_num_tasks_per_op,)

        # Calculate what percent of resources each operator would use assuming each
        # operator uses only their optimal number of tasks.
        cpu_fraction_per_op = self._compute_processor_fractions(
            optimal_num_tasks_per_op, task_num_cpus_per_op
        )
        gpu_fraction_per_op = self._compute_processor_fractions(
            optimal_num_tasks_per_op, task_num_gpus_per_op
        )

        # Allocate resources based on the fractions computed above.
        allocations = {
            op: ExecutionResources(
                cpu=global_limits.cpu * cpu_fraction_per_op[op_index],
                gpu=global_limits.gpu * gpu_fraction_per_op[op_index],
            )
            for op_index, op in enumerate(valid_ops)
        }
        allocations.update({op: None for op in ops if op not in valid_ops})

        return allocations

    def _compute_processor_fractions(
        self,
        num_tasks_per_op: NDArray[np.floating],
        task_num_processors_per_op: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Calculate what fraction of processors each operator uses.

        Given the number of tasks and processors per task for each operator, this
        computes what fraction of the total processors each operator would consume.
        The returned fractions sum to 1.0 (or all zeros if no processors are used).

        Args:
            num_tasks_per_op: Number of tasks for each operator.
            task_num_processors_per_op: Processors (CPUs or GPUs) per task for each operator.

        Returns:
            Fraction of total processors used by each operator.
        """
        num_processors_per_op = num_tasks_per_op * task_num_processors_per_op
        total_num_processors = np.sum(num_processors_per_op)
        if total_num_processors == 0:
            return np.zeros_like(task_num_processors_per_op)

        processor_fraction_per_op = num_processors_per_op / total_num_processors

        assert np.isclose(
            np.sum(processor_fraction_per_op), 1.0
        ), processor_fraction_per_op
        return processor_fraction_per_op

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
        if op.get_max_concurrency_limit() is not None:
            max_num_tasks = min(max_num_tasks, op.get_max_concurrency_limit())

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

        # NOTE: This will recompute values if you call this method with operators in the
        # same path. The logic is much simpler this way, and the number of operators is
        # small, so we accept the extra work instead of doing a single-pass version.
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
