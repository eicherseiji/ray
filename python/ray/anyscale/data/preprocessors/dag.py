from collections import defaultdict
from typing import List, Set, Callable, Sequence

from ray.data import Preprocessor
from ray.data.aggregate import AggregateFnV2


class _AggregationNode:
    """
    Represents a node in the stat aggregation DAG, wrapping a preprocessor and its associated aggregation.

    Each node tracks column-level dependencies:
    - A node depends on another if it reads from any columns produced by that node.
    - This enables topological scheduling of aggregations based on column lineage.

    Args:
        preprocessor: The preprocessor associated with this node.
        agg_fn: The aggregation function (AggregateFnV2) to compute stats.
        post_process_fn: Optional function applied to the aggregation result.
        post_key_fn: Optional function to generate output key for post-processed stats.
        column: The column to aggregate.
    """

    def __init__(
        self,
        preprocessor: Preprocessor,
        agg_fn: AggregateFnV2,
        post_process_fn: Callable,
        post_key_fn: Callable[[str], str],
        column: str,
    ):
        self.preprocessor: Preprocessor = preprocessor
        self.agg_fn: AggregateFnV2 = agg_fn
        self.post_process_fn: Callable = post_process_fn
        self.post_key_fn: Callable[[str], str] = post_key_fn
        self.column: str = column
        self.dependencies: Set[_AggregationNode] = set()
        self.dependents: Set[_AggregationNode] = set()
        self.completed: bool = False
        self.read_cols: Set[str] = set(getattr(preprocessor, "columns", []))
        self.write_cols: Set[str] = set(
            getattr(
                preprocessor, "output_columns", getattr(preprocessor, "columns", [])
            )
        )

    def is_ready(self):
        """Returns True if all dependent nodes have completed."""
        return all(dep.completed for dep in self.dependencies)


def _build_aggregation_dag(
    preprocessors: Sequence[Preprocessor],
) -> List[_AggregationNode]:
    """
    Constructs a directed acyclic graph (DAG) of aggregation nodes from a list of preprocessors.

    Each node represents a single aggregation. Edges are added based on column dependencies:
    if one node reads from columns written by another, a dependency is established.

    Args:
        preprocessors: A list of preprocessors, each containing one or more aggregations.

    Returns:
        A list of aggregation nodes with their dependency relationships set.
    """
    all_nodes: List[_AggregationNode] = []
    pre_to_nodes = defaultdict(list)

    for p in preprocessors:
        # Phase 1: Create nodes
        for agg_spec in p.stat_computation_plan:  # type: AggregateStatSpec
            node = _AggregationNode(
                preprocessor=p,
                agg_fn=agg_spec.stat_fn,
                post_process_fn=agg_spec.post_process_fn,
                post_key_fn=agg_spec.post_key_fn,
                column=agg_spec.column,
            )
            pre_to_nodes[p].append(node)
            all_nodes.append(node)
    # Phase 2: Add edges based on read/write overlap
    for i, current_p in enumerate(preprocessors):
        for j in range(i):
            prev_p = preprocessors[j]
            prev_nodes = pre_to_nodes[prev_p]
            if not prev_nodes:
                continue
            for cur_node in pre_to_nodes[current_p]:
                for prev_node in pre_to_nodes[prev_p]:
                    # Add an edge if current reads something previous writes
                    if cur_node.read_cols & prev_node.write_cols:
                        cur_node.dependencies.add(prev_node)
                        prev_node.dependents.add(cur_node)

    _validate_dag_is_acyclic(all_nodes)
    return all_nodes


def _validate_dag_is_acyclic(nodes):
    visited = set()
    stack = set()

    def visit(node: _AggregationNode):
        if node in stack:
            raise RuntimeError("Cycle detected in aggregation DAG")
        if node in visited:
            return
        stack.add(node)
        for dep in node.dependencies:
            visit(dep)
        stack.remove(node)
        visited.add(node)

    for node in nodes:
        visit(node)
