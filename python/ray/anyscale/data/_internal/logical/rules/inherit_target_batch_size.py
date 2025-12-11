from ray.data._internal.logical.interfaces import (
    Rule,
    Plan,
    LogicalPlan,
    LogicalOperator,
)
from ray.data._internal.logical.operators.map_operator import (
    StreamingRepartition,
    MapRows,
    FlatMap,
)


class FuseMapWithRepartitionRule(Rule):
    def apply(self, plan: Plan) -> Plan:
        assert isinstance(plan, LogicalPlan)

        def _inherit_target_batch_size(op: LogicalOperator) -> LogicalOperator:
            # StreamingRepartitions should have exactly 1 input
            if len(op.input_dependencies) != 1:
                return op

            input_op = op.input_dependencies[0]

            if isinstance(op, StreamingRepartition):
                if isinstance(input_op, MapRows):
                    new_op = MapRows(
                        input_op=input_op.input_dependencies[0],
                        fn=input_op._fn,
                        fn_args=input_op._fn_args,
                        fn_kwargs=input_op._fn_kwargs,
                        fn_constructor_args=input_op._fn_constructor_args,
                        fn_constructor_kwargs=input_op._fn_constructor_kwargs,
                        compute=input_op._compute,
                        ray_remote_args_fn=input_op._ray_remote_args_fn,
                        ray_remote_args=input_op._ray_remote_args,
                    )
                    new_op._target_num_rows_per_block = op.target_num_rows_per_block
                    return new_op
                elif isinstance(input_op, FlatMap):
                    new_op = FlatMap(
                        input_op=input_op.input_dependencies[0],
                        fn=input_op._fn,
                        fn_args=input_op._fn_args,
                        fn_kwargs=input_op._fn_kwargs,
                        fn_constructor_args=input_op._fn_constructor_args,
                        fn_constructor_kwargs=input_op._fn_constructor_kwargs,
                        compute=input_op._compute,
                        ray_remote_args_fn=input_op._ray_remote_args_fn,
                        ray_remote_args=input_op._ray_remote_args,
                    )
                    new_op._target_num_rows_per_block = op.target_num_rows_per_block
                    return new_op
                else:
                    return op
            # For all other operators, return as-is.
            return op

        original_dag = plan.dag
        transformed_dag = original_dag._apply_transform(_inherit_target_batch_size)

        if transformed_dag is original_dag:
            return plan
        return LogicalPlan(dag=transformed_dag, context=plan.context)
