from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ray.data._internal.planner.planner import Planner


def create_planner() -> "Planner":
    # Import here to avoid circular import.
    from ray.anyscale.data.planner import RayTurboPlanner as Planner

    return Planner()


__all__ = ["create_planner"]
