"""Standard ``fit`` plus an optional learning-rate schedule (``alpha``) callback."""

from __future__ import annotations

from typing import Any

import keras

from morpho_net.training.procedures.standard_fit import StandardFitProcedure


class AlphaSchedulerProcedure(StandardFitProcedure):
    """Attaches :class:`keras.callbacks.LearningRateScheduler` from config.

    YAML example::

        training:
          update_method: alpha_scheduler
          alpha_scheduler:
            decay_rate: 0.99   # multiply LR each epoch (default 0.99)
    """

    def extra_callbacks(self, config: dict[str, Any]) -> list[keras.callbacks.Callback]:
        sch = config.get("training", {}).get("alpha_scheduler") or {}
        if not sch:
            return []
        decay_rate = float(sch.get("decay_rate", 0.99))

        def schedule(epoch: int, lr: float) -> float:
            del epoch
            return float(lr * decay_rate)

        return [keras.callbacks.LearningRateScheduler(schedule)]
