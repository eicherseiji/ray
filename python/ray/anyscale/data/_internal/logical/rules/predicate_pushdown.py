from typing import TYPE_CHECKING, Optional, Union as TypingUnion

from ray.anyscale.data._internal.logical.graph_utils import (
    add_op_between,
    make_copy_of_dag,
    remove_op,
)
from ray.anyscale.data._internal.logical.operators.read_files_operator import ReadFiles
from ray.data._internal.logical.interfaces import LogicalOperator, LogicalPlan, Rule
from ray.data._internal.logical.operators.map_operator import Filter
from ray.data._internal.logical.operators.n_ary_operator import Union

if TYPE_CHECKING:
    import pyarrow.dataset as pd
    from ray.data.expressions import Expr


class PredicatePushdown(Rule):
    """Pushes down predicates across the graph.

    If Filter operators chaining is found with filter expression, combine the
    filter expressions and fuse the Filter operator.

    If Filter operator is found, we combine the filter expressions and
    pushdown the combined filter expression to the ReadFiles operator.

    If Union operator is found, we duplicate the filter expression for each branch
    and pushdown the combined filter expression to the ReadFiles operator.

    For read files operator, we set the filter expression on the read files operator.
    """

    # TODO: (srinathk) There is one more optimization to this filter rule, i.e.
    # if Filter ops are mixed with UserDefinedFunction and expressions, we can just
    # reorder all the Filter expressions and combine together and attempt pushdown into
    # `ReadFiles`.
    # Refer: https://anyscale1.atlassian.net/browse/DATA-229

    # TODO: (srinathk)
    # Annotate Logical plan to indicate predicate pushdown occured
    # https://anyscale1.atlassian.net/browse/DATA-244
    def apply(self, plan: LogicalPlan) -> LogicalPlan:
        dag_copy = make_copy_of_dag(plan.dag)
        plan = LogicalPlan(dag_copy, plan.context)
        plan = self._process_operator(plan.dag, None, None, plan)
        return plan

    def _process_operator(
        self,
        op: LogicalOperator,
        prev_filter: Optional[Filter] = None,
        predicate_expr_to_pushdown: Optional[
            TypingUnion["Expr", "pd.Expression"]
        ] = None,
        plan: Optional[LogicalPlan] = None,
    ) -> LogicalPlan:
        """Process a sub-DAG rooted at the given operator to push down predicates.

        Args:
            op: The operator to process
            prev_filter: The filter to push down
            predicate_expr_to_pushdown: The predicate expression to pushdown
            plan: The logical plan

        Returns:
            The modified logical plan
        """
        if prev_filter is not None:
            assert predicate_expr_to_pushdown is not None
        if isinstance(op, Filter):
            if prev_filter is None:
                if op.is_expression_based():
                    prev_filter = op
                    predicate_expr_to_pushdown = op._predicate_expr
            elif not op.is_expression_based():
                # Filter Op pushdown supported for only filter expressions
                # so we reset the filter pushdown here
                prev_filter = None
                predicate_expr_to_pushdown = None
            else:
                # Opportunity to combine filter expressions
                from ray.data.expressions import Expr

                is_pushdown_expr = isinstance(predicate_expr_to_pushdown, Expr)
                is_op_expr = isinstance(op._predicate_expr, Expr)
                can_combine = False

                if is_pushdown_expr == is_op_expr:
                    # Both are the same type (either both Ray Data or both PyArrow)
                    can_combine = True
                elif is_pushdown_expr or is_op_expr:
                    # Mixed types - convert Ray Data Expr to PyArrow
                    try:
                        if is_pushdown_expr:
                            predicate_expr_to_pushdown = (
                                predicate_expr_to_pushdown.to_pyarrow()
                            )
                        if is_op_expr:
                            op._predicate_expr = op._predicate_expr.to_pyarrow()
                        can_combine = True
                    except (ValueError, TypeError):
                        # Conversion failed (e.g., UDF expressions), reset pushdown
                        can_combine = False

                if can_combine:
                    # Combine the expressions and remove the previous filter
                    predicate_expr_to_pushdown &= op._predicate_expr
                    plan = remove_op(prev_filter, plan)
                    prev_filter = op
                    prev_filter._predicate_expr = predicate_expr_to_pushdown
                else:
                    # Can't combine, reset pushdown
                    prev_filter = None
                    predicate_expr_to_pushdown = None
        elif isinstance(op, ReadFiles) and prev_filter:
            read_files = op
            # Always push down expression-based predicates to ReadFiles
            read_files.pushdown_predicate(predicate_expr_to_pushdown)
            plan = remove_op(prev_filter, plan)
            prev_filter = None
            predicate_expr_to_pushdown = None

        elif isinstance(op, Union) and prev_filter:
            # For union operations, we need to process each branch independently
            # So we duplicate the filter expression for each branch
            from ray.data.expressions import Expr

            # Only handle Ray Data expressions for union pushdown
            if isinstance(predicate_expr_to_pushdown, Expr):
                for input_dep in op.input_dependencies:
                    # Create a new Filter operator for each branch
                    branch_filter = Filter(
                        input_dep,
                        predicate_expr=predicate_expr_to_pushdown,
                    )
                    add_op_between(
                        branch_filter,
                        upstream_op=input_dep,
                        downstream_op=op,
                    )
                    # Process the branch with the new filter
                    plan = self._process_operator(
                        branch_filter,
                        None,
                        predicate_expr_to_pushdown,
                        plan,
                    )
                # Remove the original filter after processing all branches
                plan = remove_op(prev_filter, plan)
                return plan
            else:
                # Can't handle PyArrow expressions in union pushdown, reset
                prev_filter = None
                predicate_expr_to_pushdown = None

        else:
            prev_filter = None
            predicate_expr_to_pushdown = None
        # DFS traversal of the DAG
        for input_dep in op.input_dependencies:
            plan = self._process_operator(
                input_dep,
                prev_filter,
                predicate_expr_to_pushdown,
                plan,
            )
        return plan
