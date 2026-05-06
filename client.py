from __future__ import annotations

import time
import logging
from dataclasses import dataclass
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
                "Retry %d/%d for %s (%s) — waiting %.1fs",
                attempt + 1, _MAX_RETRIES, label, type(exc).__name__, wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"Failed after {_MAX_RETRIES} retries for {label!r}")


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
    ) -> Response:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        def call() -> Response:
            kwargs: dict[str, Any] = dict(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=self._extra_body or None,
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
                raise MalformedOpenRouterResponse(
                    f"OpenRouter returned no choices for {model!r}; "
                    f"id={getattr(result, 'id', '')!r}"
                )
            content = getattr(choices[0].message, "content", None) or ""
            return Response(
                text=content,
                model=result.model,
                prompt_tokens=result.usage.prompt_tokens if result.usage else 0,
                completion_tokens=result.usage.completion_tokens if result.usage else 0,
                generation_id=getattr(result, "id", "") or "",
            )

        return retry_with_backoff(call, label=model, is_retryable=_openrouter_retryable)
