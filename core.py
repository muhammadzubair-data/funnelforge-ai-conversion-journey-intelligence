"""Backward-compatible module: the original ``from core import ...`` keeps working.

The implementation now lives in ``funnelforge/core.py``.
"""
from funnelforge.core import *  # noqa: F401,F403
from funnelforge.core import CHANNELS, STAGES, daily, funnel, generate, scenario, segments, validate  # noqa: F401
