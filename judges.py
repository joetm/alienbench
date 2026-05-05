"""Judge clients dispatched per `judge_overrides` entry.

Aliases listed in ``Config.judge_models`` without an override route through
OpenRouter as before. Aliases with an override are routed to the named native
provider SDK using a dated ``model_id``, so the resolved upstream model is
pinned at submission time and recorded as ``judge_model_resolved`` on every
extraction row.
"""

from __future__ import annotations

import logging
import os
from typing import Protocol

from alienbench.client import (
    OpenRouterClient,
    Response,
    retry_with_backoff,
)
from alienbench.config import Config, JudgeOverride

logger = logging.getLogger(__name__)


class JudgeClient(Protocol):
    def complete(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system: str | None = None,
    ) -> Response: ...


# ---------------------------------------------------------------------------
# OpenRouter (default)
# ---------------------------------------------------------------------------

class _OpenRouterJudge:
    """Adapter around the existing OpenRouterClient with a fixed model id."""

    def __init__(self, alias: str, cfg: Config) -> None:
        self._alias = alias
        self._client = OpenRouterClient(cfg)

    def complete(self, prompt, temperature, max_tokens, system=None):
        return self._client.complete(
            model=self._alias,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
        )


# ---------------------------------------------------------------------------
# Native: Anthropic
# ---------------------------------------------------------------------------

class _AnthropicJudge:
    def __init__(self, model_id: str, api_key: str) -> None:
        import anthropic
        self._sdk = anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model_id = model_id
        self._supports_temperature = True

    def complete(self, prompt, temperature, max_tokens, system=None):
        kwargs = dict(
            model=self._model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if self._supports_temperature:
            kwargs["temperature"] = temperature
        if system:
            kwargs["system"] = system

        def call() -> Response:
            try:
                r = self._client.messages.create(**kwargs)
            except self._sdk.BadRequestError as e:
                # Newer Claude models (e.g. Opus 4.7) have deprecated the
                # temperature parameter. Retry once without it and cache the
                # result so subsequent calls skip it immediately.
                if self._supports_temperature and "temperature" in str(e).lower():
                    self._supports_temperature = False
                    kwargs.pop("temperature", None)
                    r = self._client.messages.create(**kwargs)
                else:
                    raise
            text = "".join(
                block.text for block in r.content if getattr(block, "type", "") == "text"
            )
            usage = getattr(r, "usage", None)
            return Response(
                text=text,
                model=getattr(r, "model", self._model_id),
                prompt_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
                completion_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
                generation_id=getattr(r, "id", "") or "",
            )

        def retryable(exc: BaseException) -> bool:
            if isinstance(exc, self._sdk.RateLimitError):
                return True
            if isinstance(exc, self._sdk.APIStatusError):
                return exc.status_code in {429, 500, 502, 503, 504}
            return False

        return retry_with_backoff(call, label=self._model_id, is_retryable=retryable)


# ---------------------------------------------------------------------------
# Native: OpenAI
# ---------------------------------------------------------------------------

class _OpenAIJudge:
    def __init__(self, model_id: str, api_key: str) -> None:
        from openai import OpenAI, RateLimitError, APIStatusError
        self._RateLimitError = RateLimitError
        self._APIStatusError = APIStatusError
        self._client = OpenAI(api_key=api_key)  # default base_url = api.openai.com
        self._model_id = model_id

    def complete(self, prompt, temperature, max_tokens, system=None):
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        def call() -> Response:
            r = self._client.chat.completions.create(
                model=self._model_id,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
            return Response(
                text=r.choices[0].message.content or "",
                model=r.model,
                prompt_tokens=r.usage.prompt_tokens if r.usage else 0,
                completion_tokens=r.usage.completion_tokens if r.usage else 0,
                generation_id=getattr(r, "id", "") or "",
            )

        def retryable(exc: BaseException) -> bool:
            if isinstance(exc, self._RateLimitError):
                return True
            if isinstance(exc, self._APIStatusError):
                return exc.status_code in {429, 500, 502, 503, 504}
            return False

        return retry_with_backoff(call, label=self._model_id, is_retryable=retryable)


# ---------------------------------------------------------------------------
# Native: Google
# ---------------------------------------------------------------------------

class _GoogleJudge:
    # Gemini reasoning models (2.5 Pro and 3.x preview) count internal
    # "thinking" tokens against ``max_output_tokens``. The visible answer
    # is therefore truncated mid-JSON if the caller's requested budget is
    # passed straight through. We add a flat buffer to the caller's budget
    # so only the visible answer is constrained by ``max_tokens``. The
    # value of 2000 was selected from pilot Ward extraction calls on
    # gemini-2.5-pro-preview-03-25, which spent up to ~1150 thinking
    # tokens before producing the JSON answer; 2000 leaves headroom
    # without inflating cost (thinking tokens are billed at the visible
    # output rate). If a future judge model is configured with a larger
    # reasoning budget, raise this buffer accordingly.
    _THINKING_TOKEN_BUFFER = 2000

    def __init__(self, model_id: str, api_key: str) -> None:
        from google import genai
        from google.genai import errors, types
        self._types = types
        self._errors = errors
        self._client = genai.Client(api_key=api_key)
        self._model_id = model_id

    def complete(self, prompt, temperature, max_tokens, system=None):
        gen_config = self._types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens + self._THINKING_TOKEN_BUFFER,
            response_mime_type="application/json",
        )

        def call() -> Response:
            r = self._client.models.generate_content(
                model=self._model_id,
                contents=prompt,
                config=gen_config,
            )
            usage = getattr(r, "usage_metadata", None)
            return Response(
                text=r.text or "",
                model=getattr(r, "model_version", "") or self._model_id,
                prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0 if usage else 0,
                completion_tokens=getattr(usage, "candidates_token_count", 0) or 0 if usage else 0,
                generation_id=getattr(r, "response_id", "") or "",
            )

        def retryable(exc: BaseException) -> bool:
            # google-genai raises APIError subclasses with an HTTP status code.
            if isinstance(exc, self._errors.APIError):
                return exc.code in {429, 500, 502, 503, 504}
            return False

        return retry_with_backoff(call, label=self._model_id, is_retryable=retryable)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_DEFAULT_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
}


def _resolve_key(override: JudgeOverride) -> str:
    env = override.api_key_env or _DEFAULT_KEY_ENV[override.provider]
    key = os.environ.get(env)
    if not key:
        raise EnvironmentError(
            f"Environment variable {env!r} not set for native judge "
            f"({override.provider} / {override.model_id})."
        )
    return key


def make_judge(alias: str, cfg: Config) -> JudgeClient:
    """Return a JudgeClient for ``alias``.

    If ``alias`` is in ``cfg.judge_overrides`` and the override targets a native
    provider, return that provider's client with the dated ``model_id`` baked
    in. Otherwise return an OpenRouter adapter that calls ``alias`` directly.
    """
    override = cfg.judge_overrides.get(alias)
    if override is None or override.provider == "openrouter":
        return _OpenRouterJudge(alias, cfg)

    key = _resolve_key(override)
    if override.provider == "anthropic":
        return _AnthropicJudge(override.model_id, key)
    if override.provider == "openai":
        return _OpenAIJudge(override.model_id, key)
    if override.provider == "google":
        return _GoogleJudge(override.model_id, key)
    raise ValueError(f"Unknown provider: {override.provider!r}")
