"""RayTurbo-specific dependency checking utilities."""

import importlib.util
import warnings
from typing import Optional

from ray._private.ray_logging import log_once


def check_rayturbo_optional_dependency(
    package_name: str,
    min_version: Optional[str] = None,
    install_hint: Optional[str] = None,
    performance_impact: Optional[str] = None,
    warn_once_key: Optional[str] = None,
) -> bool:
    """Check if an optional dependency is available for RayTurbo optimizations.

    This function is only used in Anyscale's RayTurbo and will not affect OSS Ray.

    Args:
        package_name: Name of the package to check
        min_version: Minimum required version (e.g., ">=0.61")
        install_hint: Custom installation instruction
        performance_impact: Description of performance impact if missing
        warn_once_key: Key for warn-once behavior. If None, uses package_name

    Returns:
        True if dependency is available, False otherwise
    """
    if importlib.util.find_spec(package_name) is None:
        warning_key = warn_once_key or f"rayturbo_missing_{package_name}"

        if log_once(warning_key):
            # Build warning message
            install_msg = install_hint or f"Install {package_name}"
            if min_version:
                install_msg += min_version

            perf_msg = performance_impact or "to get better performance in RayTurbo"

            warnings.warn(
                f"{package_name.title()} isn't available. {install_msg} {perf_msg}. "
                f"Falling back to slower Python implementation for RayTurbo optimizations."
            )
        return False
    return True


def check_numba_for_hash_partitioning() -> bool:
    """Check numba availability for RayTurbo hash partitioning optimizations.

    NOTE: Ray workers perform the hash partitioning. So, if we emit a warning
          in the hash partitioning code, each worker would repeat the same
          warning, and the output becomes extremely spammy. To avoid this,
          we emit the warning on the driver, even though it's not where the
          fallback occurs.
    """
    return check_rayturbo_optional_dependency(
        package_name="numba",
        min_version=">=0.61",
        install_hint="Install numba>=0.61",
        performance_impact="to get better performance for hash partitioning operations",
        warn_once_key="rayturbo_numba_hash_partitioning",
    )
