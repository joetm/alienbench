from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from openai import OpenAI, RateLimitError, APIStatusError

from alienbench.config import Config

logger = logging.getLogger(__name__)

_RETRY_STATUSES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 6
_BACKOFF_BASE = 2.0  # seconds

T = TypeVar("T")


class MalformedOpenRouterResponse(RuntimeError):
    """OpenRouter returned a response without a usable choices list."""


@dataclass
class Response:
    text: str
    model: str               # resolved upstream model string
    prompt_tokens: int
    completion_tokens: int
    generation_id: str = ""  # per-call id from the upstream API
    # "stop" for a complete answer, "length" when max_tokens cut it off.
    finish_reason: str = ""
    # The provider's own reason (e.g. "refusal", "MAX_TOKENS"). OpenRouter
    # maps it to finish_reason, but a refusal still maps to "stop".
    native_finish_reason: str = ""
    # Hidden reasoning tokens, included in completion_tokens. 0 if unreported.
    reasoning_tokens: int = 0
    # The request exactly as sent: the API it went to and every argument of
    # the call (messages, sampling parameters, routing, provider settings).
    request: dict = field(default_factory=dict)
    # The provider that served the call, as reported by OpenRouter.
    provider: str = ""


def _jsonable(value: Any) -> Any:
    """Convert SDK request arguments to JSON-serializable values."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def request_record(api: str, kwargs: dict) -> dict:
    """The stored form of a request: the API name and every call argument."""
    return {"api": api, **_jsonable(kwargs)}


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    label: str,
    is_retryable: Callable[[BaseException], bool],
) -> T:
    """Run ``fn``, retrying with exponential backoff on retryable errors.

    Shared by OpenRouter and the native judge clients so backoff behavior
    is identical across providers.
    """
    for attempt in range(_MAX_RETRIES):
        try:
            return fn()
        except BaseException as exc:
            if not is_retryable(exc):
                raise
            wait = _BACKOFF_BASE ** attempt
            logger.warning(
                "Retry %d/%d for %s (%s: %s) — waiting %.1fs",
                attempt + 1, _MAX_RETRIES, label, type(exc).__name__, exc, wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"Failed after {_MAX_RETRIES} retries for {label!r}")


class CutOffAtCeiling(RuntimeError):
    """A response was still cut off at the largest allowed output-token limit."""

    def __init__(self, label: str, ceiling: int, cut_offs: list[dict]) -> None:
        super().__init__(
            f"{label}: cut off at every limit up to the ceiling of {ceiling} tokens"
            f" ({len(cut_offs)} calls)"
        )
        self.cut_offs = cut_offs


def cut_off_entry(response: Response, max_tokens: int) -> dict:
    """Provenance of one call that was cut off at ``max_tokens``."""
    return {
        "call_id": response.generation_id,
        "max_tokens": max_tokens,
        "completion_tokens": response.completion_tokens,
        "reasoning_tokens": response.reasoning_tokens,
        "finish_reason": response.finish_reason,
        "native_finish_reason": response.native_finish_reason,
        "timestamp": time.time(),
        "request": response.request,
    }


def complete_within_limit(
    call: Callable[[int], Response],
    *,
    max_tokens: int,
    ceiling: int,
    label: str,
) -> tuple[Response, int, list[dict]]:
    """Call ``call(limit)`` until the response is not cut off at the limit.

    The first call uses ``max_tokens``. Each cut-off response is discarded,
    and the call is repeated with the limit doubled, up to ``ceiling``.
    Returns the response, the limit it was produced under, and one
    :func:`cut_off_entry` per discarded call. Raises :class:`CutOffAtCeiling`
    when the call at ``ceiling`` is cut off as well.
    """
    limit = max_tokens
    cut_offs: list[dict] = []
    while True:
        response = call(limit)
        if response.finish_reason != "length":
            return response, limit, cut_offs
        cut_offs.append(cut_off_entry(response, limit))
        if limit >= ceiling:
            raise CutOffAtCeiling(label, ceiling, cut_offs)
        new_limit = min(2 * limit, ceiling)
        logger.warning(
            "%s: cut off at max_tokens=%d (%d completion tokens). Discarded;"
            " retrying with max_tokens=%d.",
            label, limit, response.completion_tokens, new_limit,
        )
        limit = new_limit


def _openrouter_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (RateLimitError, MalformedOpenRouterResponse)):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code in _RETRY_STATUSES:
        return True
    return False


class OpenRouterClient:
    def __init__(self, config: Config) -> None:
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.openrouter_base_url,
        )
        self._extra_body: dict[str, Any] = {}
        if config.allowed_providers:
            self._extra_body["provider"] = {
                "order": config.allowed_providers,
                "allow_fallbacks": config.allow_provider_fallbacks,
            }
        else:
            # OpenRouter's default routing only tries major providers (OpenAI,
            # Anthropic, Google, etc.) which do not host open-weight models.
            # Explicitly listing all known providers ensures open-weight models
            # (Maverick, Llama, Qwen, Mistral) route to deepinfra/novita/etc.
            self._extra_body["provider"] = {
                "order": [
                    "openai", "anthropic", "google-vertex",
                    "amazon-bedrock",
                    "cohere",  "mistral",
                    "deepinfra", "novita", "parasail", "sambanova",
                    "nebius", "groq", "friendli", "cerebras",
                    "together", "fireworks", "cloudflare",
                ],
                "allow_fallbacks": True,
                # Do not restrict routing to providers that support every
                # request parameter. Some upstreams ignore ``seed`` or
                # other optional fields; with ``require_parameters: false``
                # OpenRouter routes there anyway (the parameter is silently
                # dropped) instead of returning a 400.
                "require_parameters": False,
            }

    def complete(
        self,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system: str | None = None,
        seed: int | None = None,
        provider: str | None = None,
        allow_fallbacks: bool = False,
    ) -> Response:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        extra_body = self._extra_body
        if provider:
            # A per-model provider slug (e.g. ``openai/flex``) pins routing
            # to that upstream and overrides the client-wide provider block.
            # ``allow_fallbacks`` lets OpenRouter route elsewhere when the
            # pinned upstream is unavailable.
            extra_body = {
                **self._extra_body,
                "provider": {
                    "order": [provider],
                    "allow_fallbacks": allow_fallbacks,
                    "require_parameters": False,
                },
            }

        def call() -> Response:
            kwargs: dict[str, Any] = dict(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body or None,
            )
            # OpenRouter forwards ``seed`` to upstream providers that
            # support it (OpenAI, Google) and ignores it otherwise. The
            # generation step uses this for best-effort reproducibility;
            # the Limitations section of the paper explains what "best
            # effort" means in practice (the upstream provider may not
            # honour the seed across infrastructure changes).
            if seed is not None:
                kwargs["seed"] = seed
            result = self._client.chat.completions.create(**kwargs)
            choices = getattr(result, "choices", None)
            if not choices:
                # OpenRouter reports upstream failures (e.g. "Flex processing
                # is temporarily unavailable") as an HTTP 200 whose body holds
                # an ``error`` object instead of ``choices``. Surface it.
                error = getattr(result, "error", None)
                if error is None and getattr(result, "model_extra", None):
                    error = result.model_extra.get("error")
                if isinstance(error, dict):
                    upstream = error.get("metadata", {}) or {}
                    raise MalformedOpenRouterResponse(
                        f"Upstream error for {model!r} "
                        f"(provider={provider or 'default'}, "
                        f"code={error.get('code')}, "
                        f"type={upstream.get('error_type')}): "
                        f"{error.get('message')}; "
                        f"id={getattr(result, 'id', '')!r}"
                    )
                raise MalformedOpenRouterResponse(
                    f"OpenRouter response for {model!r} contained neither "
                    f"choices nor an error object; "
                    f"id={getattr(result, 'id', '')!r}"
                )
            content = getattr(choices[0].message, "content", None) or ""
            finish_reason = getattr(choices[0], "finish_reason", "") or ""
            # A cut-off response may be empty when hidden reasoning used the
            # whole limit. complete_within_limit retries it at a higher limit.
            if not content.strip() and finish_reason != "length":
                # An empty 200 would otherwise be written to the JSONL
                # with response="" and the sample marked complete. Raise
                # so the stage's try/except defers it to the next run.
                raise MalformedOpenRouterResponse(
                    f"OpenRouter returned empty content for {model!r}; "
                    f"id={getattr(result, 'id', '')!r}"
                )
            return Response(
                text=content,
                model=result.model,
                prompt_tokens=result.usage.prompt_tokens if result.usage else 0,
                completion_tokens=result.usage.completion_tokens if result.usage else 0,
                generation_id=getattr(result, "id", "") or "",
                finish_reason=finish_reason,
                native_finish_reason=getattr(choices[0], "native_finish_reason", "") or "",
                reasoning_tokens=getattr(
                    getattr(result.usage, "completion_tokens_details", None),
                    "reasoning_tokens", 0) or 0,
                request=request_record("openrouter/chat.completions", kwargs),
                provider=getattr(result, "provider", "") or "",
            )

        return retry_with_backoff(call, label=model, is_retryable=_openrouter_retryable)
