"""DeepSeek AI client (OpenAI-compatible API)."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from pa_agent.util.threading import CancelToken

from pa_agent.config.settings import AIProviderSettings
from pa_agent.util.mask_secret import mask_secret

try:
    from openai import OpenAI as _OpenAI  # type: ignore[import]
except ImportError as _exc:
    _OpenAI = None  # type: ignore[assignment,misc]
    _OPENAI_IMPORT_ERROR = _exc
else:
    _OPENAI_IMPORT_ERROR = None

try:
    from anthropic import Anthropic as _Anthropic  # type: ignore[import]
except ImportError as _exc:
    _Anthropic = None  # type: ignore[assignment,misc]
    _ANTHROPIC_IMPORT_ERROR = _exc
else:
    _ANTHROPIC_IMPORT_ERROR = None


def _should_use_proxy(base_url: str) -> bool:
    """Check if the target URL should use a proxy (respects NO_PROXY)."""
    import os
    from urllib.parse import urlparse

    proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    if not proxy_url:
        return False
    no_proxy = os.environ.get("NO_PROXY", "")
    if no_proxy:
        host = urlparse(base_url).hostname or ""
        no_proxy_list = [h.strip().lower() for h in no_proxy.split(",")]
        if host in no_proxy_list or any(host.endswith("." + h) for h in no_proxy_list if h):
            return False
    return True


def _make_openai_client(base_url: str, api_key: str):
    """Create an OpenAI client, respecting HTTP(S)_PROXY and NO_PROXY env vars."""
    if _should_use_proxy(base_url):
        import httpx
        import os

        proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
        http_client = httpx.Client(proxy=proxy_url)
        return _OpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
    return _OpenAI(base_url=base_url, api_key=api_key)


def _is_anthropic_provider(base_url: str, model: str) -> bool:
    """True when the API uses Anthropic Messages format (e.g. mimo)."""
    url = (base_url or "").lower()
    m = (model or "").lower()
    return "xiaomimimo" in url or "anthropic" in url and "mimo" in m


def _strip_1m_suffix(model: str) -> str:
    """Strip the [1m] suffix from model name before sending to API.

    Claude Code uses [1m] to indicate 1M context capability, but strips it
    before sending to the provider. We follow the same convention.
    """
    if model and model.lower().endswith("[1m]"):
        return model[:-4]
    return model


def _make_anthropic_client(base_url: str, api_key: str):
    """Create an Anthropic client for mimo-style providers."""
    if _Anthropic is None:
        raise RuntimeError("anthropic package is not installed") from _ANTHROPIC_IMPORT_ERROR

    # Anthropic SDK expects base_url without /v1 suffix
    clean_url = base_url.rstrip("/")
    if clean_url.endswith("/v1"):
        clean_url = clean_url[:-3]

    if _should_use_proxy(base_url):
        import httpx
        import os

        proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
        http_client = httpx.Client(proxy=proxy_url)
        return _Anthropic(api_key=api_key, base_url=clean_url, http_client=http_client)
    return _Anthropic(api_key=api_key, base_url=clean_url)

logger = logging.getLogger(__name__)


@dataclass
class AIUsage:
    """Token usage from a single API call."""
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of prompt tokens served from KV cache (0.0–1.0).

        DeepSeek 硬盘缓存命中率。值越高，费用越低。
        0.0 = 无缓存命中；1.0 = 全部命中缓存。
        """
        if self.prompt_tokens <= 0:
            return 0.0
        return self.cached_prompt_tokens / self.prompt_tokens

    @property
    def cache_miss_tokens(self) -> int:
        """Prompt tokens that were NOT served from cache (billed at full rate)."""
        return max(0, self.prompt_tokens - self.cached_prompt_tokens)


@dataclass
class AIReply:
    """Structured response from a single AI API call."""
    content: str
    reasoning_content: str
    raw: dict[str, Any]          # full raw response dict for debug tab
    usage: AIUsage
    request_id: str
    latency_ms: float


class CancelledError(Exception):
    """Raised when a cancel_token is set before or during an API call."""


def _is_deepseek_native(base_url: str) -> bool:
    return "deepseek.com" in (base_url or "").lower()


def _is_deepseek_model(model: str) -> bool:
    """True for DeepSeek model ids; excludes QClaw ``openclaw`` Agent alias."""
    m = (model or "").lower()
    if m == "openclaw":
        return False
    return "deepseek" in m


def _is_qclaw_openclaw_agent(settings: AIProviderSettings) -> bool:
    """True when requests go through QClaw's public-gateway OpenClaw Agent."""
    from pa_agent.ai.qclaw_connector import detect_qclaw, is_openclaw_model

    return bool(is_openclaw_model(settings.model) and detect_qclaw())


def _openclaw_agent_request_extra(settings: AIProviderSettings) -> dict[str, Any]:
    """Ask QClaw Agent to answer in-chat only (no exec/write tool loop)."""
    if not _is_qclaw_openclaw_agent(settings):
        return {}
    return {"tool_choice": "none"}


def _is_kkai_openai_proxy(base_url: str) -> bool:
    """KKAI (api.kkone.vip) OpenAI-compatible gateway."""
    url = (base_url or "").lower()
    return "kkone.vip" in url


def _is_packyapi(base_url: str) -> bool:
    return "packyapi.com" in (base_url or "").lower()


def _is_minimax(base_url: str) -> bool:
    """MiniMax (api.minimax.io) OpenAI-compatible gateway."""
    url = (base_url or "").lower()
    return "minimax.io" in url or "minimax.com" in url


# Packy claude-officially returns 400 if max_tokens exceeds model output cap.
_PACKY_CLAUDE_MAX_OUTPUT_TOKENS = 128_000
# DeepSeek API: max_tokens must be in [1, 393216].
_DEEPSEEK_MAX_OUTPUT_TOKENS = 393_216


def _model_uses_claude_adaptive(model: str) -> bool:
    """Claude models that require thinking.type=adaptive (not budget_tokens)."""
    m = (model or "").lower()
    return any(
        token in m
        for token in (
            "claude-opus-4-7",
            "claude-opus-4-6",
            "claude-sonnet-4-6",
        )
    )


_EFFORT_TO_ADAPTIVE_OUTPUT: dict[str, str] = {
    "none": "low",
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "max": "max",
    "xhigh": "max",
}


def _adaptive_output_effort(reasoning_effort: str | None) -> str:
    key = (reasoning_effort or "medium").strip().lower()
    return _EFFORT_TO_ADAPTIVE_OUTPUT.get(key, "medium")


# Sent to OpenAI-compatible gateways; upstream may clamp below these values.
_PRACTICAL_UNLIMITED_MAX_TOKENS = 524288
# Anthropic-style thinking requires budget_tokens < max_tokens.
_PRACTICAL_UNLIMITED_THINKING_BUDGET = 524287


def _effort_budget_tokens(effort: str | None, *, max_output: int) -> int:
    """Thinking budget; must stay below max_output (Anthropic/Packy rule)."""
    del effort  # reserved for future per-effort tuning
    return min(_PRACTICAL_UNLIMITED_THINKING_BUDGET, max(1024, max_output - 1))


def _thinking_enabled(extra_body: dict[str, Any], effort: str | None) -> bool:
    if extra_body:
        return extra_body.get("thinking", {}).get("type") in ("enabled", "adaptive")
    return effort is not None and effort != "none"


def _packy_anthropic_messages_api(settings: AIProviderSettings) -> bool:
    """Packy claude-officially uses Anthropic Messages API (no role=system in messages)."""
    return _is_packyapi(settings.base_url) and "claude" in (settings.model or "").lower()


def _prepare_chat_messages(
    settings: AIProviderSettings,
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    """Hoist system turns to top-level ``system`` for Anthropic-native Packy routes."""
    if not _packy_anthropic_messages_api(settings):
        return messages, None
    system_parts: list[str] = []
    api_messages: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "system":
            text = msg.get("content", "")
            if isinstance(text, str) and text.strip():
                system_parts.append(text)
            continue
        api_messages.append(msg)
    system_param = "\n\n".join(system_parts) if system_parts else None
    return api_messages, system_param


def _provider_max_output_tokens(settings: AIProviderSettings) -> int:
    """Per-gateway completion cap (max_tokens); avoids 400 from provider limits."""
    model = (settings.model or "").lower()
    if _is_packyapi(settings.base_url) and "claude" in model:
        return _PACKY_CLAUDE_MAX_OUTPUT_TOKENS
    if _is_deepseek_native(settings.base_url):
        return _DEEPSEEK_MAX_OUTPUT_TOKENS
    return _PRACTICAL_UNLIMITED_MAX_TOKENS


def _completion_max_tokens(
    settings: AIProviderSettings,
    *,
    extra_body: dict[str, Any],
    effort: str | None,
) -> int:
    """Total completion budget (thinking + content) for OpenAI-compatible APIs."""
    del effort, extra_body
    return _provider_max_output_tokens(settings)


def _resolve_thinking_params(
    settings: AIProviderSettings,
    *,
    thinking: bool | None,
    reasoning_effort: str | None,
) -> tuple[dict[str, Any], str | None]:
    """Return (extra_body, reasoning_effort) for chat.completions.create."""
    _thinking = thinking if thinking is not None else settings.thinking
    _effort = reasoning_effort if reasoning_effort is not None else settings.reasoning_effort
    model = settings.model or ""

    if _is_deepseek_native(settings.base_url) or _is_deepseek_model(model):
        # DeepSeek v4+ requires thinking.type=adaptive + output_config.effort;
        # the old "enabled"/"disabled" values are no longer accepted.
        # Also covers DeepSeek models proxied through non-native gateways (e.g. QClaw).
        if _thinking:
            extra_body: dict[str, Any] = {
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": _adaptive_output_effort(_effort)},
            }
            return extra_body, _effort or "medium"
        else:
            extra_body = {
                "thinking": {"type": "disabled"},
            }
            return extra_body, None

    if _is_minimax(settings.base_url):
        # MiniMax (api.minimax.io):
        # - thinking.type only accepts "adaptive" (on) or "disabled" (off); no budget_tokens
        # - reasoning_split=True exposes thinking via reasoning_content / reasoning_details
        # - M2.x cannot disable thinking; "disabled" is accepted but ignored
        if _thinking:
            extra_body = {
                "thinking": {"type": "adaptive"},
                "reasoning_split": True,
            }
        else:
            extra_body = {
                "thinking": {"type": "disabled"},
                "reasoning_split": True,
            }
        # MiniMax does not use reasoning_effort
        return extra_body, None

    if not _thinking:
        return {}, None

    max_out = _completion_max_tokens(
        settings, extra_body={}, effort=_effort
    )

    if _is_packyapi(settings.base_url) and "claude" in model.lower():
        # Packy (e.g. claude-officially): budget_tokens only; reasoning_effort rejected.
        budget = _effort_budget_tokens(_effort, max_output=max_out)
        return (
            {"thinking": {"type": "enabled", "budget_tokens": budget}},
            None,
        )

    if _is_kkai_openai_proxy(settings.base_url):
        # KKAI claude-opus-4-5: reasoning_effort -> 503 paprika_mode on some routes.
        budget = _effort_budget_tokens(_effort, max_output=max_out)
        return (
            {"thinking": {"type": "enabled", "budget_tokens": budget}},
            None,
        )

    if _model_uses_claude_adaptive(model):
        # Yunwu / New-API style gateways: Opus 4.7+ needs adaptive thinking.
        return (
            {
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": _adaptive_output_effort(_effort)},
            },
            _effort or "medium",
        )

    if "claude" in model.lower():
        budget = _effort_budget_tokens(_effort, max_output=max_out)
        return (
            {"thinking": {"type": "enabled", "budget_tokens": budget}},
            _effort or "medium",
        )

    # Other models on OpenAI-compatible proxies (o-series, deepseek-reasoner, etc.)
    return {}, _effort or "medium"


# ── Anthropic-protocol helpers ────────────────────────────────────────────────

# MiMo-V2.5-Pro max output: 128K tokens (per Xiaomi docs).
# Anthropic Opus/Sonnet max output: 128K tokens (per Anthropic docs).
_ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS = 128_000


def _anthropic_max_tokens(settings: AIProviderSettings) -> int:
    """Return the max_tokens for Anthropic-protocol API calls.

    Uses ``settings.max_output_tokens`` if configured (> 0), otherwise
    falls back to the provider default (128K for MiMo/Anthropic).
    """
    configured = getattr(settings, "max_output_tokens", 0) or 0
    if configured > 0:
        return configured
    return _ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS


# Effort → thinking budget mapping for Anthropic protocol.
# budget_tokens must be < max_tokens (Anthropic spec constraint).
_ANTHROPIC_EFFORT_BUDGET: dict[str, int] = {
    "none": 0,
    "low": 4_096,
    "medium": 16_384,
    "high": 49_152,
    "max": 98_304,
    "xhigh": 98_304,
}


def _anthropic_thinking_budget(effort: str | None, *, max_tokens: int) -> int:
    """Return thinking budget_tokens, guaranteed to be < max_tokens.

    Anthropic spec: ``budget_tokens`` must be less than ``max_tokens``.
    If the effort-based budget would exceed the limit, it is clamped.
    """
    key = (effort or "medium").strip().lower()
    budget = _ANTHROPIC_EFFORT_BUDGET.get(key, 16_384)
    if budget <= 0:
        return 0
    # Anthropic constraint: budget_tokens < max_tokens
    # Leave at least 4096 tokens for content output
    max_budget = max(1024, max_tokens - 4096)
    return min(budget, max_budget)


class DeepSeekClient:
    """Thin wrapper around the OpenAI-compatible DeepSeek API."""

    def __init__(self, settings: AIProviderSettings, logger_: logging.Logger | None = None) -> None:
        self._settings = settings
        self._log = logger_ or logger

    def update_provider(self, settings: AIProviderSettings) -> None:
        """Replace in-memory provider settings (e.g. after QClaw auto-fallback)."""
        self._settings = settings

    def _anthropic_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
        cancel_token: "CancelToken | None" = None,
        timeout_s: float = 600.0,
    ) -> AIReply:
        """Send messages using Anthropic Messages API format (for mimo)."""
        if cancel_token is not None and cancel_token.is_set():
            raise CancelledError("Request cancelled before API call")

        _thinking = thinking if thinking is not None else self._settings.thinking
        _effort = reasoning_effort if reasoning_effort is not None else self._settings.reasoning_effort

        # Extract system message
        system_text = ""
        api_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_text += msg.get("content", "") + "\n"
            else:
                api_messages.append(msg)

        client = _make_anthropic_client(self._settings.base_url, self._settings.api_key)
        masked_key = mask_secret(self._settings.api_key)
        self._log.debug(
            "Anthropic chat: model=%s thinking=%s effort=%s key=...%s msgs=%d",
            self._settings.model, _thinking, _effort,
            masked_key[-4:] if len(masked_key) >= 4 else "****",
            len(api_messages),
        )

        t0 = time.monotonic()

        # Resolve max_tokens: use configured value or default based on provider
        _max_tokens = _anthropic_max_tokens(self._settings)

        create_kwargs: dict[str, Any] = {
            "model": _strip_1m_suffix(self._settings.model),
            "messages": api_messages,
            "max_tokens": _max_tokens,
            "timeout": timeout_s,
        }
        if system_text.strip():
            create_kwargs["system"] = system_text.strip()

        # Thinking / extended thinking — budget_tokens must be < max_tokens (Anthropic spec)
        if _thinking:
            budget = _anthropic_thinking_budget(_effort, max_tokens=_max_tokens)
            create_kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}

        try:
            response = client.messages.create(**create_kwargs)
        except Exception as exc:
            latency_ms = (time.monotonic() - t0) * 1000
            self._log.error("Anthropic API error after %.0f ms: %s", latency_ms, exc)
            raise

        latency_ms = (time.monotonic() - t0) * 1000

        # Parse response
        content = ""
        reasoning_content = ""
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "thinking":
                reasoning_content += block.thinking

        usage = AIUsage(
            prompt_tokens=getattr(response.usage, "input_tokens", 0),
            cached_prompt_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
            completion_tokens=getattr(response.usage, "output_tokens", 0),
            total_tokens=getattr(response.usage, "input_tokens", 0) + getattr(response.usage, "output_tokens", 0),
        )
        request_id = getattr(response, "id", "") or ""

        raw: dict[str, Any] = {
            "id": request_id,
            "model": getattr(response, "model", ""),
            "content": content,
            "reasoning_content": reasoning_content,
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
            "latency_ms": latency_ms,
        }

        self._log.debug("Anthropic chat done: latency=%.0f ms tokens=%d/%d",
                        latency_ms, usage.prompt_tokens, usage.completion_tokens)

        return AIReply(
            content=content,
            reasoning_content=reasoning_content,
            raw=raw,
            usage=usage,
            request_id=request_id,
            latency_ms=latency_ms,
        )

    def _anthropic_stream_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        on_reasoning_token: Callable[[str], None] | None = None,
        on_content_token: Callable[[str], None] | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
        cancel_token: "CancelToken | None" = None,
        timeout_s: float = 600.0,
    ) -> AIReply:
        """Stream messages using Anthropic Messages API format (for mimo)."""
        if cancel_token is not None and cancel_token.is_set():
            raise CancelledError("Request cancelled before API call")

        _thinking = thinking if thinking is not None else self._settings.thinking
        _effort = reasoning_effort if reasoning_effort is not None else self._settings.reasoning_effort

        system_text = ""
        api_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_text += msg.get("content", "") + "\n"
            else:
                api_messages.append(msg)

        client = _make_anthropic_client(self._settings.base_url, self._settings.api_key)

        t0 = time.monotonic()

        # Resolve max_tokens: use configured value or default based on provider
        _max_tokens = _anthropic_max_tokens(self._settings)

        create_kwargs: dict[str, Any] = {
            "model": _strip_1m_suffix(self._settings.model),
            "messages": api_messages,
            "max_tokens": _max_tokens,
            "timeout": timeout_s,
        }
        if system_text.strip():
            create_kwargs["system"] = system_text.strip()

        # Thinking / extended thinking — budget_tokens must be < max_tokens (Anthropic spec)
        if _thinking:
            budget = _anthropic_thinking_budget(_effort, max_tokens=_max_tokens)
            create_kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}

        reasoning_content = ""
        content = ""
        request_id = ""
        prompt_tokens = 0
        completion_tokens = 0

        try:
            with client.messages.stream(**create_kwargs) as stream:
                for event in stream:
                    if cancel_token is not None and cancel_token.is_set():
                        raise CancelledError("Request cancelled during streaming")

                    if event.type == "content_block_start":
                        pass
                    elif event.type == "content_block_delta":
                        if hasattr(event.delta, "thinking"):
                            reasoning_content += event.delta.thinking
                            if on_reasoning_token:
                                on_reasoning_token(event.delta.thinking)
                        elif hasattr(event.delta, "text"):
                            content += event.delta.text
                            if on_content_token:
                                on_content_token(event.delta.text)
                    elif event.type == "message_start":
                        msg = event.message
                        request_id = getattr(msg, "id", "") or ""
                        if hasattr(msg, "usage"):
                            prompt_tokens = getattr(msg.usage, "input_tokens", 0)

                final_msg = stream.get_final_message()
                if final_msg and hasattr(final_msg, "usage"):
                    completion_tokens = getattr(final_msg.usage, "output_tokens", 0)

        except CancelledError:
            raise
        except Exception as exc:
            latency_ms = (time.monotonic() - t0) * 1000
            self._log.error("Anthropic stream error after %.0f ms: %s", latency_ms, exc)
            raise

        latency_ms = (time.monotonic() - t0) * 1000

        usage = AIUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

        raw: dict[str, Any] = {
            "id": request_id,
            "content": content,
            "reasoning_content": reasoning_content,
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
            "latency_ms": latency_ms,
        }

        self._log.info("Anthropic stream done: latency=%.0f ms reasoning=%d content=%d",
                        latency_ms, len(reasoning_content), len(content))

        return AIReply(
            content=content,
            reasoning_content=reasoning_content,
            raw=raw,
            usage=usage,
            request_id=request_id,
            latency_ms=latency_ms,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
        context_window: int | None = None,
        cancel_token: "CancelToken | None" = None,
        timeout_s: float = 600.0,
    ) -> AIReply:
        """Send *messages* to the DeepSeek API and return a structured reply.

        Raises CancelledError if cancel_token is set before the call.
        Never sends temperature/top_p/presence_penalty/frequency_penalty.
        """
        # Route to Anthropic format for mimo-style providers
        if _is_anthropic_provider(self._settings.base_url, self._settings.model):
            return self._anthropic_chat(
                messages, thinking=thinking, reasoning_effort=reasoning_effort,
                cancel_token=cancel_token, timeout_s=timeout_s,
            )

        # Check cancellation before making the network call
        if cancel_token is not None and cancel_token.is_set():
            raise CancelledError("Request cancelled before API call")

        extra_body, _effort = _resolve_thinking_params(
            self._settings, thinking=thinking, reasoning_effort=reasoning_effort
        )
        extra_body = {**extra_body, **_openclaw_agent_request_extra(self._settings)}
        api_messages, system_param = _prepare_chat_messages(self._settings, messages)
        if system_param:
            extra_body = {**extra_body, "system": system_param}
        _thinking_on = _thinking_enabled(extra_body, _effort)
        _max_tokens = _completion_max_tokens(
            self._settings, extra_body=extra_body, effort=_effort
        )

        masked_key = mask_secret(self._settings.api_key)
        self._log.debug(
            "DeepSeekClient.chat: model=%s thinking=%s effort=%s max_tokens=%s "
            "system_hoisted=%s key=...%s msgs=%d",
            self._settings.model,
            _thinking_on,
            _effort,
            _max_tokens,
            bool(system_param),
            masked_key[-4:] if len(masked_key) >= 4 else "****",
            len(api_messages),
        )

        if _OpenAI is None:
            raise RuntimeError("openai package is not installed") from _OPENAI_IMPORT_ERROR

        client = _make_openai_client(self._settings.base_url, self._settings.api_key)

        t0 = time.monotonic()
        create_kwargs: dict[str, Any] = {
            "model": self._settings.model,
            "messages": api_messages,
            "timeout": timeout_s,
            "max_tokens": _max_tokens,
        }
        if extra_body:
            create_kwargs["extra_body"] = extra_body
        if _effort is not None:
            create_kwargs["reasoning_effort"] = _effort
        # When thinking mode is OFF, set temperature=0 for maximum instruction-following
        # fidelity and JSON format compliance.  Thinking mode is incompatible with
        # temperature (DeepSeek/Anthropic spec), so we only inject it when safe.
        if not _thinking_on:
            create_kwargs["temperature"] = 0
        try:
            response = client.chat.completions.create(
                **create_kwargs,
                # IMPORTANT: do NOT add temperature, top_p, presence_penalty,
                # frequency_penalty — they are incompatible with thinking mode.
            )
        except Exception as exc:
            latency_ms = (time.monotonic() - t0) * 1000
            self._log.error("DeepSeekClient API error after %.0f ms: %s", latency_ms, exc)
            raise

        latency_ms = (time.monotonic() - t0) * 1000

        msg = response.choices[0].message
        content = msg.content or ""
        reasoning_content = getattr(msg, "reasoning_content", None) or ""
        # MiniMax with reasoning_split=True may also use reasoning_details
        if not reasoning_content:
            details = getattr(msg, "reasoning_details", None)
            if details:
                parts = []
                for detail in details:
                    t = detail.get("text") if isinstance(detail, dict) else getattr(detail, "text", None)
                    if t:
                        parts.append(t)
                reasoning_content = "".join(parts)

        # Build usage
        u = response.usage
        usage = AIUsage(
            prompt_tokens=getattr(u, "prompt_tokens", 0),
            cached_prompt_tokens=getattr(
                getattr(u, "prompt_tokens_details", None), "cached_tokens", 0
            ) if u else 0,
            completion_tokens=getattr(u, "completion_tokens", 0),
            total_tokens=getattr(u, "total_tokens", 0),
        )

        request_id = getattr(response, "id", "") or ""

        # Build raw dict for debug tab — mask API key if it somehow appears
        raw: dict[str, Any] = {
            "id": request_id,
            "model": getattr(response, "model", ""),
            "content": content,
            "reasoning_content": reasoning_content,
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "cached_prompt_tokens": usage.cached_prompt_tokens,
                "cache_miss_tokens": usage.cache_miss_tokens,
                "cache_hit_rate_pct": round(usage.cache_hit_rate * 100, 1),
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
            "latency_ms": latency_ms,
        }

        self._log.debug(
            "DeepSeekClient.chat done: latency=%.0f ms tokens=%d/%d",
            latency_ms, usage.prompt_tokens, usage.completion_tokens,
        )

        # Log KV-cache hit rate so operators can monitor savings.
        # DeepSeek硬盘缓存：prompt_cache_hit_tokens 是命中缓存的 token 数。
        if usage.prompt_tokens > 0:
            hit_rate = usage.cached_prompt_tokens / usage.prompt_tokens * 100
            self._log.info(
                "KV-cache: hit=%d miss=%d total_prompt=%d hit_rate=%.1f%%",
                usage.cached_prompt_tokens,
                usage.prompt_tokens - usage.cached_prompt_tokens,
                usage.prompt_tokens,
                hit_rate,
            )

        return AIReply(
            content=content,
            reasoning_content=reasoning_content,
            raw=raw,
            usage=usage,
            request_id=request_id,
            latency_ms=latency_ms,
        )

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        on_reasoning_token: Callable[[str], None] | None = None,
        on_content_token: Callable[[str], None] | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
        cancel_token: "CancelToken | None" = None,
        timeout_s: float = 600.0,
    ) -> AIReply:
        """Stream *messages* to the DeepSeek API, calling callbacks per token.

        Follows the official DeepSeek streaming example exactly:
        - reasoning_content tokens arrive first (thinking phase)
        - content tokens arrive after (answer phase)
        - delta.reasoning_content is None (not empty string) when absent

        Parameters
        ----------
        on_reasoning_token:
            Called with each reasoning/thinking token chunk as it arrives.
        on_content_token:
            Called with each content token chunk as it arrives.

        Returns the same AIReply as chat() once the stream is complete.
        Raises CancelledError if cancel_token is set before or during the call.
        """
        # Route to Anthropic format for mimo-style providers
        if _is_anthropic_provider(self._settings.base_url, self._settings.model):
            return self._anthropic_stream_chat(
                messages, on_reasoning_token=on_reasoning_token,
                on_content_token=on_content_token, thinking=thinking,
                reasoning_effort=reasoning_effort, cancel_token=cancel_token,
                timeout_s=timeout_s,
            )

        if cancel_token is not None and cancel_token.is_set():
            raise CancelledError("Request cancelled before API call")

        extra_body, _effort = _resolve_thinking_params(
            self._settings, thinking=thinking, reasoning_effort=reasoning_effort
        )
        extra_body = {**extra_body, **_openclaw_agent_request_extra(self._settings)}
        api_messages, system_param = _prepare_chat_messages(self._settings, messages)
        if system_param:
            extra_body = {**extra_body, "system": system_param}
        _thinking_on = _thinking_enabled(extra_body, _effort)
        _max_tokens = _completion_max_tokens(
            self._settings, extra_body=extra_body, effort=_effort
        )

        self._log.info(
            "DeepSeekClient.stream_chat: model=%s thinking=%s reasoning_effort=%s "
            "max_tokens=%s system_hoisted=%s msgs=%d",
            self._settings.model,
            _thinking_on,
            _effort,
            _max_tokens,
            bool(system_param),
            len(api_messages),
        )

        if _OpenAI is None:
            raise RuntimeError("openai package is not installed") from _OPENAI_IMPORT_ERROR

        client = _make_openai_client(self._settings.base_url, self._settings.api_key)

        t0 = time.monotonic()
        reasoning_content = ""
        content = ""
        request_id = ""
        model_name = ""
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        cached_tokens = 0

        try:
            # Build kwargs with stream_options to get usage in the final chunk.
            # Some providers may not support it; if the create() call itself
            # rejects stream_options we retry without it.
            stream_kwargs: dict[str, Any] = {
                "model": self._settings.model,
                "messages": api_messages,
                "timeout": timeout_s,
                "max_tokens": _max_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            if extra_body:
                stream_kwargs["extra_body"] = extra_body
            if _effort is not None:
                stream_kwargs["reasoning_effort"] = _effort

            try:
                stream = client.chat.completions.create(**stream_kwargs)
            except Exception:
                # Retry without stream_options if provider rejects it
                self._log.debug("stream_options not supported; retrying without it")
                stream_kwargs.pop("stream_options", None)
                stream = client.chat.completions.create(**stream_kwargs)

            for chunk in stream:
                # Check cancellation on each chunk
                if cancel_token is not None and cancel_token.is_set():
                    raise CancelledError("Request cancelled during streaming")

                # Extract usage from the final chunk (stream_options)
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    u = chunk.usage
                    prompt_tokens = getattr(u, "prompt_tokens", 0) or prompt_tokens
                    completion_tokens = getattr(u, "completion_tokens", 0) or completion_tokens
                    total_tokens = getattr(u, "total_tokens", 0) or total_tokens
                    details = getattr(u, "prompt_tokens_details", None)
                    cached_tokens = getattr(details, "cached_tokens", 0) if details else cached_tokens

                if not getattr(chunk, "choices", None):
                    continue

                request_id = request_id or (getattr(chunk, "id", "") or "")
                model_name = model_name or (getattr(chunk, "model", "") or "")

                choice0 = chunk.choices[0]
                delta = getattr(choice0, "delta", None)
                if delta is None:
                    continue

                # Official pattern: reasoning_content is None when absent, not ""
                # reasoning_content arrives first (thinking phase), then content
                # MiniMax with reasoning_split=True uses delta.reasoning_details[].text
                # instead of delta.reasoning_content.
                r = getattr(delta, "reasoning_content", None)
                if not r:
                    # MiniMax streaming: reasoning_details is a list of dicts
                    details = getattr(delta, "reasoning_details", None)
                    if details:
                        for detail in details:
                            t = detail.get("text") if isinstance(detail, dict) else getattr(detail, "text", None)
                            if t:
                                r = (r or "") + t
                if r:
                    reasoning_content += r
                    if on_reasoning_token is not None:
                        on_reasoning_token(r)
                elif delta.content:
                    content += delta.content
                    if on_content_token is not None:
                        on_content_token(delta.content)

        except CancelledError:
            raise
        except Exception as exc:
            latency_ms = (time.monotonic() - t0) * 1000
            self._log.error("DeepSeekClient stream error after %.0f ms: %s", latency_ms, exc)
            raise

        latency_ms = (time.monotonic() - t0) * 1000

        usage = AIUsage(
            prompt_tokens=prompt_tokens,
            cached_prompt_tokens=cached_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        raw: dict[str, Any] = {
            "id": request_id,
            "model": model_name,
            "content": content,
            "reasoning_content": reasoning_content,
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "cached_prompt_tokens": usage.cached_prompt_tokens,
                "cache_miss_tokens": usage.cache_miss_tokens,
                "cache_hit_rate_pct": round(usage.cache_hit_rate * 100, 1),
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
            "latency_ms": latency_ms,
        }

        self._log.info(
            "DeepSeekClient.stream_chat done: latency=%.0f ms "
            "reasoning_chars=%d content_chars=%d deepseek_thinking=%s effort=%s",
            latency_ms,
            len(reasoning_content),
            len(content),
            _thinking_on,
            _effort,
        )

        # Log KV-cache hit rate for stream calls as well.
        if usage.prompt_tokens > 0:
            hit_rate = usage.cached_prompt_tokens / usage.prompt_tokens * 100
            self._log.info(
                "KV-cache: hit=%d miss=%d total_prompt=%d hit_rate=%.1f%%",
                usage.cached_prompt_tokens,
                usage.prompt_tokens - usage.cached_prompt_tokens,
                usage.prompt_tokens,
                hit_rate,
            )
        if not content.strip():
            self._log.warning(
                "API returned empty content (model=%s base_url=%s). "
                "Check 原始 tab Raw Response; for KKAI/Claude ensure model ID and token group match.",
                self._settings.model,
                self._settings.base_url,
            )
        if _thinking_on and len(reasoning_content) < 80:
            self._log.warning(
                "Thinking enabled but reasoning_content is very short (%d chars). "
                "For KKAI/Claude use reasoning_effort (not DeepSeek extra_body); "
                "check model ID, token group, and reasoning_effort=%s.",
                len(reasoning_content),
                _effort,
            )

        return AIReply(
            content=content,
            reasoning_content=reasoning_content,
            raw=raw,
            usage=usage,
            request_id=request_id,
            latency_ms=latency_ms,
        )
