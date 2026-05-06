"""Judge clients dispatched per `judge_overrides` entry.

Aliases listed in ``Config.judge_models`` without an override route through
OpenRouter as before. Aliases with an override are routed to the named native
provider SDK using a dated ``model_id``, so the resolved upstream model is
pinned at submission time and recorded as ``judge_model_resolved`` on every
extraction row.
"""

from __future__ import annotations

import contextlib
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


class _Drop400BadRequest(logging.Filter):
    """Drop httpx INFO records that report a 400 Bad Request."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "400 Bad Request" not in record.getMessage()


@contextlib.contextmanager
def _hide_httpx_400_if(active: bool):
    """Drop httpx's "400 Bad Request" INFO line while ``active`` is true.

    Used by the Google judge to keep the expected probe-failure 400 from
    surfacing as a "Bad Request" line in the run log. The probe outcome
    is reported by the caller at a higher abstraction level instead.
    Other status codes (200 OK, 429, 5xx) still pass through, so retry
    diagnostics remain visible.
    """
    if not active:
        yield
        return
    httpx_logger = logging.getLogger("httpx")
    flt = _Drop400BadRequest()
    httpx_logger.addFilter(flt)
    try:
        yield
    finally:
        httpx_logger.removeFilter(flt)


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
    """Direct Anthropic-API judge with optional AWS Bedrock failover.

    When ``bedrock_client`` is provided, a 429 from the direct Anthropic
    API triggers a one-shot retry against Bedrock for the same call.
    Bedrock hosts the same Claude snapshots under a separate quota
    pool, so the failover absorbs per-minute rate-limit bursts on the
    direct API. After a 429 the judge stays on Bedrock for
    ``bedrock_cooldown_seconds`` (30 s by default) so subsequent calls
    skip the direct API and avoid re-tripping the same rate limit.
    Once the cooldown lapses the next call tries the direct API again,
    which lets traffic return to the primary path when the rate limit
    has cleared.

    Both clients return the same ``messages.create`` response shape, so
    the post-call extraction logic is shared. The ``model`` field on
    the resolved Response reflects whichever path served the call;
    Bedrock returns the ``us.anthropic.claude-...`` slug while the
    direct API returns the dated ``claude-opus-...`` slug. Mixed values
    across rows in a single run are expected and benign because both
    paths target the same underlying snapshot.
    """

    def __init__(
        self,
        model_id: str,
        api_key: str,
        *,
        bedrock_model_id: str | None = None,
        bedrock_region: str | None = None,
        bedrock_cooldown_seconds: float = 30.0,
    ) -> None:
        import anthropic
        self._sdk = anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model_id = model_id
        self._supports_temperature = True
        # Bedrock failover state. ``_bedrock_client`` is None when no
        # failover is configured; ``_bedrock_model_id`` is the Bedrock
        # publisher slug to call (different from ``model_id``).
        self._bedrock_client = None
        self._bedrock_model_id = bedrock_model_id
        self._bedrock_cooldown_seconds = bedrock_cooldown_seconds
        self._bedrock_until = 0.0  # monotonic timestamp; >0 means circuit is open
        if bedrock_model_id:
            # Build the Bedrock client lazily per judge so a missing
            # boto3/AWS install only blows up when Bedrock is actually
            # configured for use.
            kwargs = {}
            if bedrock_region:
                kwargs["aws_region"] = bedrock_region
            self._bedrock_client = anthropic.AnthropicBedrock(**kwargs)

    def _build_kwargs(self, prompt, temperature, max_tokens, system, *, model):
        kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if self._supports_temperature:
            kwargs["temperature"] = temperature
        if system:
            kwargs["system"] = system
        return kwargs

    def _format_response(self, r) -> Response:
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

    def complete(self, prompt, temperature, max_tokens, system=None):
        import time as _time

        def call_via_bedrock() -> Response:
            kwargs = self._build_kwargs(
                prompt, temperature, max_tokens, system,
                model=self._bedrock_model_id,
            )
            return self._format_response(
                self._bedrock_client.messages.create(**kwargs)
            )

        def call_via_anthropic() -> Response:
            kwargs = self._build_kwargs(
                prompt, temperature, max_tokens, system, model=self._model_id,
            )
            try:
                r = self._client.messages.create(**kwargs)
            except self._sdk.BadRequestError as e:
                # Newer Claude models (e.g. Opus 4.7) have deprecated
                # the temperature parameter. Retry once without it and
                # cache the result so subsequent calls skip it.
                if self._supports_temperature and "temperature" in str(e).lower():
                    self._supports_temperature = False
                    kwargs.pop("temperature", None)
                    r = self._client.messages.create(**kwargs)
                else:
                    raise
            return self._format_response(r)

        def call() -> Response:
            now = _time.monotonic()
            # Circuit open: the most recent direct-API call hit a 429,
            # so route this call straight to Bedrock until the cooldown
            # lapses. Avoids hammering a known-rate-limited endpoint.
            if self._bedrock_client is not None and now < self._bedrock_until:
                return call_via_bedrock()
            try:
                return call_via_anthropic()
            except self._sdk.RateLimitError:
                if self._bedrock_client is None:
                    raise
                self._bedrock_until = now + self._bedrock_cooldown_seconds
                logger.info(
                    "Anthropic API rate-limited for %s; failing over to Bedrock "
                    "(%s) for the next %.0fs.",
                    self._model_id,
                    self._bedrock_model_id,
                    self._bedrock_cooldown_seconds,
                )
                return call_via_bedrock()

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
    # Thinking is disabled (thinking_budget=0) by default for Ward extraction:
    # the task is deterministic structured-JSON output at temperature=0.0 and
    # does not benefit from extended reasoning. Disabling thinking keeps
    # latency and cost predictable and avoids thinking tokens consuming the
    # output budget. Some Gemini models (e.g. gemini-3.1-pro-preview) reject
    # ``thinking_budget=0`` with a 400 — for those, we fall back to omitting
    # ``thinking_config`` entirely (model default) and cache the decision so
    # subsequent calls skip the disabled-thinking config immediately.

    def __init__(
        self,
        model_id: str,
        *,
        api_key: str | None = None,
        vertex_project: str | None = None,
        vertex_location: str | None = None,
    ) -> None:
        from google import genai
        from google.genai import errors, types
        self._types = types
        self._errors = errors
        if vertex_project:
            # Vertex auth uses Application Default Credentials; no api_key.
            self._client = genai.Client(
                vertexai=True,
                project=vertex_project,
                location=vertex_location or "us-central1",
            )
        else:
            self._client = genai.Client(api_key=api_key)
        self._model_id = model_id
        self._supports_disabled_thinking = True
        self._thinking_probe_done = False

    def _build_config(self, system, temperature, max_tokens):
        kwargs = dict(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
        )
        if self._supports_disabled_thinking:
            kwargs["thinking_config"] = self._types.ThinkingConfig(thinking_budget=0)
        return self._types.GenerateContentConfig(**kwargs)

    def complete(self, prompt, temperature, max_tokens, system=None):
        def call() -> Response:
            try:
                # While we have not yet established whether the model
                # accepts thinking_budget=0, drop httpx's "400 Bad Request"
                # log line for this call. We report the probe outcome
                # ourselves below; other status codes (429/5xx) still log
                # normally so retry diagnostics remain visible.
                with _hide_httpx_400_if(not self._thinking_probe_done):
                    r = self._client.models.generate_content(
                        model=self._model_id,
                        contents=prompt,
                        config=self._build_config(system, temperature, max_tokens),
                    )
                self._thinking_probe_done = True
            except self._errors.APIError as e:
                msg = str(e).lower()
                if (
                    self._supports_disabled_thinking
                    and getattr(e, "code", None) == 400
                    and ("thinking" in msg or "budget 0" in msg)
                ):
                    # This model requires thinking mode. Drop the thinking
                    # override and retry with the model default. Cache so we
                    # do not pay this 400 again on subsequent calls.
                    self._supports_disabled_thinking = False
                    self._thinking_probe_done = True
                    logger.info(
                        "Thinking probe failed for %s (rejected "
                        "thinking_budget=0). Retrying without thinking_config, "
                        "cached for this run.",
                        self._model_id,
                    )
                    r = self._client.models.generate_content(
                        model=self._model_id,
                        contents=prompt,
                        config=self._build_config(system, temperature, max_tokens),
                    )
                else:
                    raise
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

    if override.provider == "google" and override.use_vertex:
        proj = os.environ.get(override.vertex_project_env)
        if not proj:
            raise EnvironmentError(
                f"{override.vertex_project_env!r} not set; required for "
                f"use_vertex=true on judge {alias!r}."
            )
        loc = os.environ.get(override.vertex_location_env) or "us-central1"
        return _GoogleJudge(
            override.model_id,
            vertex_project=proj,
            vertex_location=loc,
        )

    key = _resolve_key(override)
    if override.provider == "anthropic":
        # Bedrock failover is opt-in. Region resolution: explicit env
        # var first (default AWS_REGION), then fall back to whatever
        # boto3 picks up later. ``None`` is acceptable; the boto3
        # default chain will resolve a region from ~/.aws/config or
        # AWS_DEFAULT_REGION at call time.
        bedrock_region = (
            os.environ.get(override.bedrock_region_env)
            if override.bedrock_fallback else None
        )
        return _AnthropicJudge(
            override.model_id,
            key,
            bedrock_model_id=(
                override.bedrock_model_id if override.bedrock_fallback else None
            ),
            bedrock_region=bedrock_region,
        )
    if override.provider == "openai":
        return _OpenAIJudge(override.model_id, key)
    if override.provider == "google":
        return _GoogleJudge(override.model_id, api_key=key)
    raise ValueError(f"Unknown provider: {override.provider!r}")
