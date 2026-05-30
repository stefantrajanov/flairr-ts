"""
FLAIRR-TS package.
Import lazily to avoid circular imports at load time.
"""


def run_flairr_ts(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
    """Thin shim – delegates to orchestrator.run_flairr_ts at call time."""
    from .orchestrator import run_flairr_ts as _run  # noqa: PLC0415
    return _run(*args, **kwargs)

