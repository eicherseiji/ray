"""
Resource Allocator V2: Water-filling based allocation strategy

This implementation is inspired by the simple water-filling allocator from
_resource_allocator_benchmark.py. It aims to equalize throughput across operators
by allocating resources proportionally to their productivity coefficients.

Key concepts:
- Productivity coefficients are measured with respect to sink mebibytes (MiB/s):
  alpha = CPU-seconds/MiB, beta = GPU-seconds/MiB, gamma ~ seconds of buffering per MiB rate.
- Water-filling allocation ensures all operators can achieve the same target throughput (MiB/s).
- Resources are allocated based on operator productivity and current workload.
"""
import logging
import math
import sys
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from ray.data import ExecutionResources
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.execution.resource_manager import (
    OpResourceAllocator,
    _is_shuffle_op,
)
from ray.data._internal.util import GiB, MiB

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ray.data._internal.execution.resource_manager import ResourceManager


class ThroughputBasedResourceAllocator(OpResourceAllocator):

    _DEFAULT_EWMA_FACTOR = 0.9
    _ALLOCATION_REFRESH_THRESHOLD_S = 0.5

    """
    Water-filling resource allocator that equalizes throughput across operators.

    This allocator implements a water-filling strategy where:
    1. Each operator has productivity coefficients measured per sink mebibyte
       (alpha=CPU-s/MiB, beta=GPU-s/MiB).
    2. Target throughput is calculated as: min(total_cpu / sum(alpha), total_gpu / sum(beta)).
    3. Each operator gets: cpu_alloc = target_rate * alpha, gpu_alloc = target_rate * beta.
    4. Object store allocation is based on expected data flow and queue depths.
    """

    def __init__(self, resource_manager: "ResourceManager"):
        super().__init__(resource_manager)

        # Per-operator resource allocations
        self._op_budgets: Dict[PhysicalOperator, ExecutionResources] = {}
        self._op_allocations: Dict[PhysicalOperator, ExecutionResources] = {}

        self._target_rate = 0.0

        # Per-operator productivity coefficients (CPU-s/MiB, GPU-s/MiB, HEAP-s/MiB)
        self._op_productivity: Dict[PhysicalOperator, Tuple[float, float, float]] = {}

        # Exponential moving average factor for productivity updates
        self._ema_factor = self._DEFAULT_EWMA_FACTOR

        self._last_allocated_at = 0

    def can_submit_new_task(self, op: PhysicalOperator) -> bool:
        if op not in self._op_budgets:
            return True

        # Do not throttle, until at least 1 task is submitted
        if op.metrics.num_tasks_submitted == 0:
            return True

        task_resource_req = op.incremental_resource_usage()

        # Check if operator has enough resources to submit one new task
        has_sufficient_budget = task_resource_req.satisfies_limit(self._op_budgets[op])

        # Handle the case when operator is allocated less than its incremental usage,
        # and can't therefore utilize its budget
        if not has_sufficient_budget:
            # To unblock the operator to schedule at least 1 task following conditions
            # have to be met:
            #
            #   - Operator is allocated less than its incremental usage
            #   - Operator still has non-zero budget (ie it hasn't utilized it yet)
            #
            has_sufficient_allocation = task_resource_req.satisfies_limit(
                self._op_allocations[op]
            )
            non_zero_budget = (
                self._op_budgets[op].cpu > 0 or self._op_allocations[op].gpu > 0
            )

            has_sufficient_budget = not has_sufficient_allocation and non_zero_budget

        return (
            has_sufficient_budget
            and
            # Avoid scheduling if there's no more Object Store budget (for task outputs)
            self._op_budgets[op].object_store_memory > 0
        )

    def max_task_output_bytes_to_read(self, op: PhysicalOperator) -> Optional[int]:
        if op not in self._op_budgets:
            return None

        # Do not throttle, when output queue is empty
        if self._topology[op].output_queue.num_blocks == 0:
            return None

        remaining_budget = round(self._op_budgets[op].object_store_memory)

        # Allow unblocking if downstream is starved
        if (
            remaining_budget == 0
            and self._should_unblock_streaming_output_backpressure(op)
        ):
            remaining_budget = 1

        return remaining_budget

    def update_budgets(
        self,
        *,
        limits: ExecutionResources,
    ):
        """Update resource budgets using water-filling allocation strategy."""

        eligible_ops = self._get_eligible_ops()

        # NOTE: Refresh allocations no more frequently than configured
        if (
            set(eligible_ops) != set(self._op_allocations.keys())
            or time.perf_counter() - self._last_allocated_at
            > self._ALLOCATION_REFRESH_THRESHOLD_S
        ):
            # Compute expansion ratios for operators:
            #   - Bytes: output bytes per task / input bytes per task
            op_byte_expansion_ratios = {
                op: _estimate_byte_expansion_ratio(op) for op in eligible_ops
            }

            # Update operator productivity coefficients
            self._update_productivity_coefficients(
                eligible_ops, op_byte_expansion_ratios
            )

            # Perform water-filling allocation
            (
                allocatable_limits,
                target_rate,
                op_allocations,
            ) = self._allocate_water_filling(
                eligible_ops,
                self._op_productivity,
                op_byte_expansion_ratios,
                limits,
            )

            self._target_rate = target_rate
            self._op_allocations = op_allocations
            self._last_allocated_at = time.perf_counter()

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("=== Water-Filling Allocator: Allocation Complete ===")
                logger.debug(f"Total limits: {limits}")
                logger.debug(f"Allocatable limits: {allocatable_limits}")
                logger.debug(f"Target rate: {self._target_rate:.2f} MiB/s")

            for op in eligible_ops:
                rows_expansion_ratio = (
                    op.metrics.average_rows_outputs_per_task or 0
                ) / (op.metrics.average_rows_inputs_per_task or 1)
                bytes_expansion_ratio = op_byte_expansion_ratios.get(op)

                alpha, beta, gamma_heap = self._op_productivity[op]
                allocation = self._op_allocations.get(op, ExecutionResources.zero())
                budget = self._op_budgets.get(op, ExecutionResources.zero())

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"  {op}:")
                    logger.debug(
                        f"    Row expansion ratio: {rows_expansion_ratio=:.2e} ({op.metrics.average_rows_outputs_per_task} / {op.metrics.average_rows_inputs_per_task})"
                    )
                    logger.debug(
                        f"    Bytes expansion ratio: {bytes_expansion_ratio=:.2e} ({op.metrics.average_bytes_outputs_per_task} / {op.metrics.average_bytes_inputs_per_task})"
                    )
                    logger.debug(
                        f"    Productivity: {alpha=:.2e} {beta=:.2e} {gamma_heap=:.2e}"
                    )
                    logger.debug(
                        f"    Allocation: CPU={allocation.cpu:.1f} GPU={allocation.gpu:.1f} "
                        f"Mem={allocation.memory / GiB:.1f}GB ObjStore={allocation.object_store_memory / GiB:.1f}GB"
                    )
                    logger.debug(
                        f"    Budget: CPU={budget.cpu:.1f} GPU={budget.gpu:.1f} "
                        f"Mem={budget.memory / GiB:.1f}GB OS={budget.object_store_memory / GiB:.1f}GB"
                    )

        # Clear budgets to remove stale entries for ops that are no longer eligible
        self._op_budgets = {}

        # Budget = Allocation - Current usage
        for op in eligible_ops:
            current_usage = self._resource_manager.get_op_usage(op)
            allocation = self._op_allocations[op]

            self._op_budgets[op] = allocation.subtract(current_usage).max(
                ExecutionResources.zero()
            )

    def get_budget(self, op: PhysicalOperator) -> Optional[ExecutionResources]:
        return self._op_budgets.get(op)

    def get_output_budget(self, op: PhysicalOperator) -> Optional[int]:
        budget = self._op_budgets.get(op)
        return budget.object_store_memory if budget is not None else None

    def get_allocation(self, op: PhysicalOperator) -> Optional[ExecutionResources]:
        return self._op_allocations.get(op)

    def _allocate_water_filling(
        self,
        eligible_ops: List[PhysicalOperator],
        op_productivity_rates: Dict[PhysicalOperator, Tuple[float, float, float]],
        op_byte_expansion_ratios: Dict[PhysicalOperator, float],
        limits: ExecutionResources,
    ) -> Tuple[ExecutionResources, float, Dict[PhysicalOperator, ExecutionResources]]:
        """
        Perform water-filling allocation to equalize throughput across operators.
        Target throughput is expressed in MiB/second.
        """

        op_allocations: Dict[PhysicalOperator, ExecutionResources] = {}

        if not eligible_ops:
            return limits, 0, op_allocations

        # We're explicitly exempting shuffle operator at the end of the stage from the pool
        # of operators considered for resource allocation:
        #
        #   - No operations downstream from the shuffle op could start executing until
        #     shuffle completes (group of concurrently executing operators preceding
        #     shuffle is called a "stage")
        #   - Shuffle op (for the most part) is just materializing the data from upstream
        #   - Shuffle op won't produce outputs, until all upstream operators are complete
        #
        # Shuffle ops essentially serve as sinks and hence do not require corresponding
        # resource allocation and hence are exempted from it.
        if _is_shuffle_op(eligible_ops[-1]):
            shuffle_op = eligible_ops[-1]
            eligible_ops = eligible_ops[:-1]

            op_allocations = {
                shuffle_op: ExecutionResources(
                    cpu=float("+inf"),
                    gpu=0,
                    memory=float("+inf"),
                    object_store_memory=sys.maxsize,
                )
            }
        else:
            eligible_ops = eligible_ops

        if not eligible_ops:
            return limits, 0, op_allocations

        # NOTE: Until all allocatable operators start executing and producing outputs (ie
        #       when their alpha or beta become > 0) we limit total amount of resources
        #       that could be allocated to prevent upstream operators from hogging all
        #       of the cluster's resources, starving downstream ones when they start
        #       executing.
        #
        # To achieve that baseline reservation ratios are devised, determining max
        # amount of resources that could be *reserved* for a given operator before
        # they could be allocated based on the productivity rates.
        #
        # These ratios are picked such that:
        #   - Sum of ratios == 1: ie when all operators become running, all resources
        #     will be allocated based on productivity rates and default reservations
        #     become 0
        #   - Ratios are exponentially decreasing: ie earlier operator is provisioned with
        #     exponentially more resources than downstream one
        baseline_op_reservation_ratios = _get_baseline_allocatable_ratios(eligible_ops)

        assert len(baseline_op_reservation_ratios) == len(
            eligible_ops
        ), f"ratios={baseline_op_reservation_ratios}, ops={eligible_ops}"

        running_ops_indices = [
            i
            for i, op in enumerate(eligible_ops)
            # Check that either alpha or beta productivity factors are positive,
            # entailing that the operator has produced some outputs (ie is running)
            if (
                # Alpha
                op_productivity_rates[op][0] > 0
                or
                # Beta
                op_productivity_rates[op][1] > 0
            )
        ]

        last_running_op_idx = max(running_ops_indices) + 1 if running_ops_indices else 0

        # We split all operators into *running* (ie that have already produced
        # some outputs) and pending to determine what portion of cluster resources
        # would be allocatable via water-filling algorithm:
        #
        #   - We find latest operator that's already running in the linearized DAG
        #   - We sum up base allocation ratios up to that operator to determine
        #     total allocatable portion of the cluster resources
        #
        running_ops = eligible_ops[:last_running_op_idx]
        pending_ops = eligible_ops[len(running_ops) :]

        assert (
            running_ops + pending_ops == eligible_ops
        ), f"{len(running_ops)=}, {len(pending_ops)=}, {len(eligible_ops)=}"

        # Total allocatable fraction is determined as sum of baseline reservation
        # ratios of all operators with productivity rates > 0 (ie operators that started
        # executing and produce outputs)
        total_allocatable_fraction = sum(
            [ratio for op, ratio in zip(running_ops, baseline_op_reservation_ratios)]
        )

        allocatable_resource_limits = limits.scale(total_allocatable_fraction)
        assert (
            allocatable_resource_limits.is_non_negative()
        ), f"{limits=}, {total_allocatable_fraction=}"

        # Calculate target throughput (MiB/second) for the given resource limits using
        # water-filling allocation strategy
        target_throughput = self._calculate_target_throughput(
            running_ops,
            op_productivity_rates,
            allocatable_resource_limits,
        )

        running_ops_object_store_allocations = self._allocate_object_store(
            running_ops,
            allocatable_resource_limits.object_store_memory,
            op_byte_expansion_ratios,
        )

        # Allocate resources to each of the running operators based on target rate
        for idx, op in enumerate(eligible_ops):
            # Extract corresponding CPU/GPU/memory productivity rates
            alpha, beta, gamma_heap = op_productivity_rates[op]

            # If operator is running, its resources are allocated based on the
            # its estimated productivity rates
            if alpha > 0 or beta > 0:
                # Calculate resource allocation based on target rate (MiB/s) and productivity (s/MiB)
                #
                # NOTE: Provided that target rate could be infinite, we're adding a
                #       safety check and zero in corresponding allocation whenever
                #       productivity rate is 0
                cpu_alloc = target_throughput * alpha if alpha != 0 else 0
                gpu_alloc = target_throughput * beta if beta != 0 else 0

                object_store_alloc = running_ops_object_store_allocations[op]

                # TODO support memory allocation as well (this will require a
                #      fallback/default value in cases when this value isn't provided).
                #      Given that we use num_cpu=1 by default we can fallback to
                #      default of ``memory - object-store / num_cpus``
                memory_alloc = float("+inf")
            else:
                # Each of the operators not participating in productivity-based
                # allocation still receives "reservation" based on the baseline
                # ratio (decreasing exponentially) to be able to launch its first
                # set of tasks

                # TODO avoid allocating gpus to non-gpu ops (and vice versa)
                op_reserved_resources = limits.scale(
                    baseline_op_reservation_ratios[idx]
                )

                cpu_alloc = op_reserved_resources.cpu
                gpu_alloc = op_reserved_resources.gpu

                object_store_alloc = op_reserved_resources.object_store_memory

                memory_alloc = float("+inf")

            op_allocations[op] = ExecutionResources(
                cpu=cpu_alloc,
                gpu=gpu_alloc,
                memory=memory_alloc,
                object_store_memory=object_store_alloc,
            )

        return allocatable_resource_limits, target_throughput, op_allocations

    def _calculate_target_throughput(
        self,
        allocatable_ops: List[PhysicalOperator],
        op_productivity_rates: Dict[PhysicalOperator, Tuple[float, float, float]],
        limits: ExecutionResources,
    ) -> float:
        """
        Calculate target throughput using water-filling principle in MiB/second.
        Target rate (MiB/s) = min(total_cpu / sum(alpha), total_gpu / sum(beta)).
        """
        if not allocatable_ops:
            return 0.0

        # Sum productivity coefficients across all operators, following the
        # constraints:
        #
        #   resource-alloc_i = target-rate * resource-productivity-factor_i
        #   sum(resource-alloc_i for all eligible operators i) = total-resource
        #
        # Which yields:
        #
        #   target_rate * sum(resource-productivity-factor_i) = total-resource
        #
        # And hence:
        #
        #   target_rate = total-resource / sum(resource-productivity-factor_i)
        #
        # Where
        #   - resource-alloc_i: Allocation of resource R to an i-th operator
        #   - target-rate: Target rate (MiB / s) for the whole pipeline
        #   - resource-productivity-factor_i: Productivity of the i-th operator
        #       determined as: `resource * seconds / MiB`
        #
        cumulative_alpha = sum(op_productivity_rates[op][0] for op in allocatable_ops)
        cumulative_beta = sum(op_productivity_rates[op][1] for op in allocatable_ops)

        # Calculate target rate (throughput) based on available CPUs
        cpu_limited_rate = (
            limits.cpu / cumulative_alpha if cumulative_alpha > 0 else float("inf")
        )
        # Cap it based on available GPUs
        if limits.gpu and limits.gpu > 0 and cumulative_beta > 0:
            gpu_limited_rate = limits.gpu / cumulative_beta
        else:
            gpu_limited_rate = float("inf")

        target_rate = min(cpu_limited_rate, gpu_limited_rate)

        # Apply operator-specific rate caps
        max_rate_caps = [
            self._get_operator_max_rate_cap(op, op_productivity_rates[op])
            for op in allocatable_ops
        ]

        final_target_rate = min(target_rate, *max_rate_caps)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("=== Water-Filling Allocator: Computing Target Rate ===")
            logger.debug(
                f"Computed target throughput: {final_target_rate:.2f} MiB/s (CPU={cpu_limited_rate:.2f} MiB/s / GPU={gpu_limited_rate:.2f} MiB/s)"
            )
            logger.debug(f"Target rate (before capping): {target_rate=:.2f} MiB/s")
            for op, cap in zip(allocatable_ops, max_rate_caps):
                logger.debug(f"  {op}: {cap:.5f}")

        return final_target_rate

    @classmethod
    def _estimate_temporal_productivity(
        cls,
        op: PhysicalOperator,
        yield_normalization_factor: float,
    ) -> float:
        # Calculate normalized (at the sink) output bytes yield
        output_bytes_per_task = op.metrics.average_bytes_outputs_per_task or 0
        normalized_output_bytes = output_bytes_per_task * yield_normalization_factor
        # Convert to MiB to avoid FP precision issues
        normalized_output_mibs = normalized_output_bytes / MiB

        # Calculate temporal productivity using normalized MiB
        avg_task_dur_s = (
            op.metrics.average_task_completion_excl_backpressure_time_s or 0
        )

        # Temporal productivity is measured in seconds / MiB, representing
        # how many seconds it takes current operator to produce 1 MiB of output
        # (normalized to pipeline's sink mebibytes)
        return (
            avg_task_dur_s / normalized_output_mibs if normalized_output_mibs > 0 else 0
        )

    @classmethod
    def _estimate_productivity(
        cls, op: PhysicalOperator, yield_normalization_factor: float
    ) -> Tuple[float, float, float]:
        """Estimate per task CPU, GPU, and heap memory productivity with respect to MiB:

            Compute (allocation) * Average task duration (seconds) /
                Average # of *normalized* output MiB produced (per task)

        Normalization of MiB produced per task is achieved through application
        of ``yield_normalization_factor`` (dimensionless), which accounts for operators' bytes
        expansion ratios across downstream operators.

        Args:
            op: The operator to estimate productivity for
            yield_normalization_factor: Yield normalization factor (product of downstream
            bytes expansion ratios)

        Returns a tuple of CPU-s/MiB, GPU-s/MiB, and heap-memory-s/MiB productivity rates.
        """
        # NOTE: We're not estimating shuffle operator productivity as it's not
        #       producing any output until the whole shuffle operation completes
        if _is_shuffle_op(op):
            return 0, 0, 0

        per_task_resource_allocation = op.per_task_resource_allocation()

        temporal_productivity = cls._estimate_temporal_productivity(
            op, yield_normalization_factor
        )

        return (
            temporal_productivity * per_task_resource_allocation.cpu,
            temporal_productivity * per_task_resource_allocation.gpu,
            temporal_productivity * per_task_resource_allocation.memory,
        )

    def _update_productivity_coefficients(
        self,
        eligible_ops: List[PhysicalOperator],
        op_byte_expansion_ratios: Dict[PhysicalOperator, float],
    ):
        """Update productivity coefficients using exponential moving average."""

        # Normalization factor is dimensionless (accounts for bytes expansion across operators)
        output_bytes_normalization_factor = 1.0

        # We compute productivities in reverse order to simplify calculation
        # of the yield normalization factor (dimensionless)
        for idx, op in enumerate(reversed(eligible_ops)):
            # Fetch current productivity estimates
            cur_alpha, cur_beta, cur_gamma_heap = self._estimate_productivity(
                op, output_bytes_normalization_factor
            )

            # Update normalization factor with the current op bytes expansion ratio
            bytes_expansion_ratio = op_byte_expansion_ratios[op]

            # In pipeline A -> B -> C, normalized MiB of the operator A is defined as:
            #
            #   output_mib_A_norm = output_mib_A * bytes_expansion_ratio_B * bytes_expansion_ratio_C
            #   bytes_expansion_ratio_B = output_bytes_B / input_bytes_B (dimensionless)
            #   bytes_expansion_ratio_C = output_bytes_C / input_bytes_C (dimensionless)
            #
            # In other words we express how many MiB output by operator A translate
            # into MiB at the pipeline sink.
            output_bytes_normalization_factor *= bytes_expansion_ratio

            if op in self._op_productivity:
                # Update using EMA
                prev_alpha, prev_beta, prev_gamma_heap = self._op_productivity[op]

                updated_alpha = self._get_ema(cur_alpha, prev_alpha)
                updated_beta = self._get_ema(cur_beta, prev_beta)
                updated_gamma_heap = self._get_ema(cur_gamma_heap, prev_gamma_heap)
            else:
                updated_alpha = cur_alpha
                updated_beta = cur_beta
                updated_gamma_heap = cur_gamma_heap

            self._op_productivity[op] = (
                updated_alpha,
                updated_beta,
                updated_gamma_heap,
            )

    def _get_ema(self, new_val: float, prev_val: float):
        return self._ema_factor * prev_val + (1 - self._ema_factor) * new_val

    @staticmethod
    def _get_operator_max_rate_cap(
        op: PhysicalOperator, op_productivity: Tuple[float, float, float]
    ) -> float:
        """Get operator's maximum rate cap based on concurrency limits."""
        max_concurrency = op.get_max_concurrency_limit()
        if max_concurrency is None:
            return float("inf")

        alpha, _, _ = op_productivity
        if alpha <= 0:
            return float("inf")

        per_task_resource_allocation = op.per_task_resource_allocation()

        # Max throughput rate (MiB/s) is defined as
        #   Max task concurrency * Task resource allocation / Productivity coefficient (s/MiB)
        #
        # TODO should be min(cpu, gpu, mem) of rates
        return max_concurrency * per_task_resource_allocation.cpu / alpha

    def _allocate_object_store(
        self,
        allocatable_ops: List[PhysicalOperator],
        total_allocatable_object_store_bytes: int,
        op_byte_expansion_ratios: Dict[PhysicalOperator, float],
    ) -> Dict[PhysicalOperator, int]:

        if not allocatable_ops:
            return {}

        # Object Store is allocated according to the following principles:
        #
        #   1. Object Store is allocated *exclusively* for task's outputs (this is the
        #   only currently considered way of consuming the object store by the
        #   operators).
        #
        #   2. Object store is allocated such that to allow for every operator to
        #   reach the same required target (throughput) rate (rows/sec)
        #
        #   3. Given that the size (in bytes) of every row could vary b/w the operators,
        #   Object Store is allocated proportionally to gross byte *expansion ratio*
        #   (defined below)
        #
        # Row's expansion ratio of the operator is defined as
        #
        #   Row's expansion ratio = Avg output row byte-size / Avg input row byte-size
        #
        # NOTE: To make sure weights are normalized, weight of the individual op in the
        #       object store allocation is determined as its expansion ratio relative to
        #       the input row size of the very first operator in the chain.

        # Normalized task output bytes tracks how much output bytes a task of the
        # current operator generates from the very first operator in this pipeline.
        #
        # Let's take following pipeline A -> B:
        #   - A receives 100 bytes per task, produces 200 bytes (expansion ratio of 2)
        #   - B receives 100 bytes per task, produces 300 bytes (expansion ratio of 3)
        #
        # Outputs of operators normalized to the input of the first operator (A):
        #   - A: 100 * (200 / 100) = 200 bytes
        #   - B: 200 * (300 / 100) = 600 bytes
        #
        # Which means that from 100 bytes of the input of A, B produces 600 bytes
        first_op = allocatable_ops[0]
        normalized_task_output_bytes = (
            first_op.metrics.average_bytes_inputs_per_task or 1
        )

        # Determine individual operator's weights
        weights = []

        for op in allocatable_ops:
            # NOTE: If there's no outputs produced yet or at all, we consider
            #       row expansion ratio alas the operator's target weight to be 0
            bytes_expansion_ratio = op_byte_expansion_ratios[op]
            normalized_task_output_bytes *= bytes_expansion_ratio or 0

            # Normalized task output bytes are used as a weight for the corresponding
            # operator
            weights.append(normalized_task_output_bytes)

        total_object_store_bytes = total_allocatable_object_store_bytes
        total_weights = sum(weights)

        if total_weights == 0:
            # Fallback to blind fair split
            weights = [1 for _ in allocatable_ops]
            total_weights = len(allocatable_ops)

        allocations = {
            # NOTE: For liveness purposes we always allocate at least 1 byte
            op: math.ceil(total_object_store_bytes * w / total_weights) + 1
            for op, w in zip(allocatable_ops, weights)
        }

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("=== Water-Filling Allocator: Object Store ===")
            logger.debug("Allocations:")
            for idx, op in enumerate(allocatable_ops):
                logger.debug(
                    f"  {str(op)}: weights={weights[idx]}, allocation={allocations[op] / GiB:.2f}GB"
                )

        assert len(allocations) == len(allocatable_ops)

        return allocations


def _get_baseline_allocatable_ratios(
    allocatable_ops: List[PhysicalOperator],
) -> List[float]:
    """This method produces a sequence of exponentially decreasing ratios of available resources to be
    allocated to the provided sequence of operators feature following properties:

        - Sum of ratios must be 1
        - Ratios are decreasing exponentially (w/ the base of 2)

    These ratios are meant to be used in following scenarios:

        - To limit initially running set of operators from greedily hogging all resources, starving
        downstream ones
        - Use these as a fallback, default allocation strategy when productivity rates aren't available yet
    """
    if len(allocatable_ops) <= 1:
        return [1.0]

    return [
        # 1 running operator (out of N) can claim 1/2th of total resources
        # 2 running operators (out of N) can claim 3/4th of total resources
        # 3 running operators (out of N) can claim 7/8th of total resources
        #
        # All running operators have to claim 100% of total resources.
        1 / 2 ** min(i + 1, len(allocatable_ops) - 1)
        for i, op in enumerate(allocatable_ops)
    ]


def _estimate_byte_expansion_ratio(op: PhysicalOperator) -> Optional[float]:
    """Calculate the bytes expansion ratio for an operator.

    Expansion ratio = output_bytes_per_task / input_bytes_per_task

    This represents how many output bytes a single input byte is expanded into.
    """

    return (
        (op.metrics.average_bytes_outputs_per_task or 0)
        /
        # NOTE: In some cases tasks could legitimately produce 0 output bytes (for ex,
        #       filtering tasks). In that case we simply fall back to 1 to avoid
        #       division by zero error.
        (op.metrics.average_bytes_inputs_per_task or 1.0)
    )
