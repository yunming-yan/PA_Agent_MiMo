"""Normalize common Stage 1 AI JSON variants before schema validation."""
from __future__ import annotations

import copy
import logging
from typing import Any

from pa_agent.ai.trace_normalize import normalize_stage1_traces

logger = logging.getLogger(__name__)


def normalize_stage1(obj: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *obj* with known AI quirks corrected."""
    out = copy.deepcopy(obj)

    if "strategy_files_needed" not in out or out.get("strategy_files_needed") is None:
        alt = out.pop("recommended_strategy_files", None)
        if alt is not None:
            out["strategy_files_needed"] = list(alt) if isinstance(alt, list) else []
            logger.debug("Mapped recommended_strategy_files -> strategy_files_needed")
        elif out.get("cycle_position") and out.get("direction"):
            try:
                from pa_agent.ai.router import route_strategy_files

                out["strategy_files_needed"] = route_strategy_files(out)
                logger.debug("Filled strategy_files_needed from router")
            except Exception as exc:  # noqa: BLE001
                logger.debug("router fallback for strategy_files_needed failed: %s", exc)
                out.setdefault("strategy_files_needed", [])

    normalize_stage1_traces(out)

    return out
