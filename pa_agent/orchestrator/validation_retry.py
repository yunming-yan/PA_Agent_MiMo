"""Validate model output with optional continuation retry."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Literal

from pa_agent.ai.json_validator import Ok, ValidationError
from pa_agent.ai.retry_feedback import build_retry_feedback, parse_previous_for_cheat
from pa_agent.ai.retry_policy import detect_cheat, should_retry

logger = logging.getLogger(__name__)

StageName = Literal["stage1", "stage2"]

# Effort downgrade ladder: when thinking exhaustion is detected, step down one level.
_EFFORT_LADDER: list[str] = ["xhigh", "max", "high", "medium", "low", "none"]


def _detect_thinking_exhaustion(reply: Any) -> bool:
    """Return True when the model spent too many tokens thinking and produced
    insufficient content (content was truncated or nearly empty).

    Heuristic: reasoning_content > 50,000 chars AND content < 500 chars.
    """
    reasoning = getattr(reply, "reasoning_content", None) or ""
    content = getattr(reply, "content", None) or ""
    return len(reasoning) > 50_000 and len(content.strip()) < 500


def _downgrade_effort(current_effort: str | None) -> str | None:
    """Step down one level on the effort ladder. Returns None if already at bottom."""
    eff = (current_effort or "medium").strip().lower()
    try:
        idx = _EFFORT_LADDER.index(eff)
    except ValueError:
        return "medium"
    if idx + 1 >= len(_EFFORT_LADDER):
        return None
    return _EFFORT_LADDER[idx + 1]


@dataclass
class ValidationRetryResult:
    result: Ok | ValidationError
    messages: list[dict[str, Any]]
    reply: Any
    attempts: int
    cheat_detected: bool = False


def validate_with_retry(
    *,
    stage: StageName,
    messages: list[dict[str, Any]],
    reply: Any,
    validator: Any,
    validation_settings: Any,
    validate_kwargs: dict[str, Any],
    call_api: Callable[[list[dict[str, Any]], bool, str | None], Any],
    thinking: bool = True,
    reasoning_effort: str | None = "max",
) -> ValidationRetryResult:
    """Validate *reply*; on retryable failure append feedback and re-call API.

    Parameters
    ----------
    call_api:
        Callable ``(messages, thinking, reasoning_effort) -> reply``.
        The *thinking* and *reasoning_effort* may be adjusted on retry when
        thinking exhaustion is detected (long reasoning but truncated content).
    thinking:
        Whether extended thinking is enabled for the initial call.
    reasoning_effort:
        The reasoning effort level for the initial call (e.g. "max", "high").
    """
    max_attempts = int(getattr(validation_settings, "retry_max", 3) or 0)
    if not getattr(validation_settings, "retry_enabled", True):
        max_attempts = 0
    if stage == "stage2" and not getattr(validation_settings, "retry_stage2", True):
        max_attempts = 0

    current_messages = list(messages)
    current_reply = reply
    attempt = 0
    previous_raw: str | None = None
    previous_obj: dict[str, Any] | None = None
    current_thinking = thinking
    current_effort = reasoning_effort

    while True:
        content = getattr(current_reply, "content", None) or ""
        result = validator.validate(stage, content, **validate_kwargs)

        if isinstance(result, Ok):
            if attempt > 0 and previous_obj is not None:
                cheats = detect_cheat(stage, previous_obj, result.obj)
                if cheats:
                    logger.warning(
                        "%s retry cheat detected after attempt %d: %s",
                        stage,
                        attempt,
                        "; ".join(cheats),
                    )
                    return ValidationRetryResult(
                        result=ValidationError(
                            category="c",
                            stage=stage,
                            raw_text=content,
                            message="重试后篡改了不可变字段: " + "; ".join(cheats),
                            invalid_fields=[f"cheat:{c}" for c in cheats],
                        ),
                        messages=current_messages,
                        reply=current_reply,
                        attempts=attempt + 1,
                        cheat_detected=True,
                    )
            return ValidationRetryResult(
                result=result,
                messages=current_messages,
                reply=current_reply,
                attempts=attempt + 1,
            )

        err = result
        if not should_retry(
            err.category,
            err.invalid_fields,
            err.missing_fields,
            attempt=attempt,
            settings=validation_settings,
        ):
            return ValidationRetryResult(
                result=err,
                messages=current_messages,
                reply=current_reply,
                attempts=attempt + 1,
            )

        # ── Thinking exhaustion detection: downgrade effort before retry ──
        if _detect_thinking_exhaustion(current_reply):
            new_effort = _downgrade_effort(current_effort)
            if new_effort is not None:
                logger.warning(
                    "%s thinking exhaustion detected (reasoning=%d chars, content=%d chars); "
                    "downgrading effort %s → %s for retry",
                    stage,
                    len(getattr(current_reply, "reasoning_content", None) or ""),
                    len(content),
                    current_effort,
                    new_effort,
                )
                current_effort = new_effort
            else:
                logger.warning(
                    "%s thinking exhaustion detected but effort already at minimum (%s); "
                    "retrying with same parameters",
                    stage,
                    current_effort,
                )

        attempt += 1
        logger.info(
            "%s validation failed (category=%s), retry %d/%d (thinking=%s, effort=%s)",
            stage,
            err.category,
            attempt,
            max_attempts,
            current_thinking,
            current_effort,
        )

        previous_raw = content
        previous_obj = parse_previous_for_cheat(previous_raw)

        feedback = build_retry_feedback(
            err,
            stage=stage,
            attempt=attempt,
            max_attempts=max_attempts,
            frame=validate_kwargs.get("kline_frame"),
            previous_raw=previous_raw,
        )
        current_messages = current_messages + [
            {"role": "assistant", "content": content},
            {"role": "user", "content": feedback},
        ]
        current_reply = call_api(current_messages, current_thinking, current_effort)
