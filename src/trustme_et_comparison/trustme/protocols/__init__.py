"""Zoja-style TrustMe ET protocol runner package."""

__all__ = ["run_zoja_protocols"]


def run_zoja_protocols(*args, **kwargs):
    """Lazy import wrapper for protocol runner entrypoint."""

    from .runner import run_zoja_protocols as _run_zoja_protocols

    return _run_zoja_protocols(*args, **kwargs)
