from collections import defaultdict
from typing import List, Set, Callable, Sequence

from ray.data.preprocessor import Preprocessor
from ray.data.aggregate import AggregateFnV2


class _DAGNode:
    """
    Base class for nodes in the stat aggregation DAG.

    Each node tracks column-level dependencies:
    - A node depends on another if it reads from any columns produced by that node.
    - This enables topological scheduling of aggregations based on column lineage.

    Args:
        preprocessor: The preprocessor associated with this node.
    """

    def __init__(self, preprocessor: Preprocessor):
        self.preprocessor: Preprocessor = preprocessor
        self.dependencies: Set["_DAGNode"] = set()
        self.dependents: Set["_DAGNode"] = set()
        self.completed: bool = False
        self.read_cols: Set[str] = set(preprocessor.get_input_columns())
        self.write_cols: Set[str] = set(preprocessor.get_output_columns())

    def is_ready(self):
        """Returns True if all dependent nodes have completed."""
        return all(dep.completed for dep in self.dependencies)

    @property
    def is_placeholder(self) -> bool:
        """Returns True if this is a placeholder node."""
        return isinstance(self, _PlaceholderNode)


class _AggregationNode(_DAGNode):
    """
    Node representing an actual aggregation operation.

    Args:
        preprocessor: The preprocessor associated with this node.
        agg_fn: The aggregation function (AggregateFnV2) to compute stats.
        post_process_fn: Function applied to the aggregation result.
        post_key_fn: Function to generate output key for post-processed stats.
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
        super().__init__(preprocessor)
        self.agg_fn: AggregateFnV2 = agg_fn
        self.post_process_fn: Callable = post_process_fn
        self.post_key_fn: Callable[[str], str] = post_key_fn
        self.column: str = column


class _PlaceholderNode(_DAGNode):
    """
    Placeholder node for non-fittable preprocessors.

    These nodes have no aggregation to run but participate in dependency
    tracking to ensure correct execution order of transforms.

    Args:
        preprocessor: The preprocessor associated with this node.
    """

    def __init__(self, preprocessor: Preprocessor):
        super().__init__(preprocessor)
        # Placeholder nodes are immediately completed since they have no aggregation
        self.completed = True


def _build_aggregation_dag(
    preprocessors: Sequence[Preprocessor],
) -> List[_DAGNode]:
    """
    Constructs a directed acyclic graph (DAG) of aggregation nodes from a list of preprocessors.

    Each node represents a single aggregation. Edges are added based on column dependencies:
    if one node reads from columns written by another, a dependency is established.

    Args:
        preprocessors: A list of preprocessors, each containing one or more aggregations.

    Returns:
        A list of aggregation nodes with their dependency relationships set.
    """
    all_nodes: List[_DAGNode] = []
    pre_to_nodes = defaultdict(list)

    for p in preprocessors:
        # Phase 1: Create nodes
        # Non-fittable preprocessors don't have aggregations, so create placeholder nodes
        if not p._is_fittable:
            placeholder_node = _PlaceholderNode(preprocessor=p)
            pre_to_nodes[p].append(placeholder_node)
            all_nodes.append(placeholder_node)
            continue

        # Fittable preprocessors: create nodes for their aggregations
        has_aggregations = False
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
            has_aggregations = True

        # Fittable preprocessors must have aggregations
        # Exception: Chain preprocessors delegate to their contained preprocessors
        from ray.anyscale.data.preprocessors.turbo_chain import Chain

        if not has_aggregations and not isinstance(p, Chain):
            raise ValueError(
                f"Preprocessor {p.__class__.__name__} is marked as fittable "
                f"(_is_fittable=True) but has no aggregations in stat_computation_plan. "
                f"Either add aggregations or set _is_fittable=False."
            )
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
