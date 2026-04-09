"""stage_2b_train — deprecated shim over pipeline.training.

.. deprecated:: 0.3.0
   Import directly from ``qwen_image_19.pipeline.training`` instead.
   This module will be removed in a future release.
"""
from __future__ import annotations

import warnings

warnings.warn(
    "qwen_image_19.stage_2b_train is deprecated — use qwen_image_19.pipeline.training",
    DeprecationWarning,
    stacklevel=2,
)

from qwen_image_19.pipeline.training import (
    execute_training,
    load_training_config,
    plan_training,
)

__all__ = ["execute_training", "load_training_config", "plan_training"]
