from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator, model_validator

load_dotenv()


class PromptVariant(BaseModel):
    id: str
    label: str
    text: str
    # Table-header acronym (LaTeX allowed) and the condition name the paper
    # prints. Both default to ``label`` when the config omits them.
    acronym: str | None = None
    name: str | None = None

    @property
    def short(self) -> str:
        return self.acronym or self.label

    @property
    def display(self) -> str:
        return self.name or self.label


class ModelInfo(BaseModel):
    """Human-readable facts about one subject model.

    ``models`` in the YAML accepts either a bare OpenRouter id or a mapping
    with these fields. ``Config.models`` stays a list of ids, and this record
    is looked up through ``Config.model_info``.
    """
    id: str
    name: str | None = None
    # Letters-only stem used in the generated ``\\ab<Key>...`` LaTeX macros.
    key: str | None = None
    tier: str | None = None
    resolved_version: str | None = None
    # Provider family; a judge and a subject model are same-family when equal.
    family: str | None = None
    # True when the provider publishes the weights, False for API-only models.
    open_weights: bool | None = None
    # OpenRouter provider slug that serves every request for this model, e.g.
    # ``openai`` or the cheaper ``openai/flex`` variant. None keeps the
    # default routing (``allowed_providers`` or the built-in provider order).
    provider: str | None = None
    # When True, OpenRouter may route to another provider if ``provider`` is
    # unavailable. Each record's ``served_provider`` names the one that replied.
    allow_fallbacks: bool = False


class JudgeInfo(BaseModel):
    """Human-readable facts about one judge-panel alias."""
    id: str
    name: str | None = None
    key: str | None = None
    # Release date of the pinned judge snapshot, as printed in the paper.
    date: str | None = None
    # Output-token limit of the first extraction call, and the ceiling the
    # limit doubles toward when a response is cut off. None uses the
    # top-level ``judge_max_tokens`` / ``judge_max_tokens_ceiling``.
    max_tokens: int | None = None
    max_tokens_ceiling: int | None = None
    # Provider slug for this judge's requests, e.g. ``openai/flex``. On the
    # OpenRouter path it pins the upstream provider (see ``ModelInfo``). With
    # a native ``judge_overrides`` entry the base name must match the
    # override's ``provider``, and the variant after the slash is sent as
    # OpenAI's ``service_tier`` (OpenAI only). None keeps the default routing.
    provider: str | None = None


class TierInfo(BaseModel):
    """A subject-model tier: the label the paper uses and its description."""
    description: str
    # Short parenthetical on parameter scale, e.g. ``~8B``. Optional.
    scale: str | None = None


class HumanBaseline(BaseModel):
    """Collection protocol of the human baseline (Prolific)."""
    id: str = "human/prolific-baseline"
    name: str = "Human baseline"
    platform: str = "Prolific"
    n_participants: int | None = None
    min_approval_rate: int | None = None
    min_prior_tasks: int | None = None
    hourly_pay: str | None = None
    median_minutes: int | None = None
    median_seconds: int | None = None
    n_excluded: int | None = None


class JudgeOverride(BaseModel):
    """Route a `judge_models` alias to a native provider SDK with a pinned dated model id.

    Aliases listed in ``Config.judge_models`` that lack an override go through OpenRouter
    as before. The alias remains the panel key used for paths and dataframes; ``model_id``
    is what is actually sent to the provider's API and recorded as ``judge_model_resolved``.
    """
    provider: Literal["anthropic", "openai", "google", "openrouter"]
    model_id: str
    api_key_env: str | None = None  # defaults applied per-provider in judges.make_judge
    # Google-only. When True, route through Vertex AI instead of AI Studio.
    # Project id and region are read from the env vars named below; auth uses
    # Application Default Credentials (gcloud auth application-default login,
    # or GOOGLE_APPLICATION_CREDENTIALS pointing to a service-account JSON).
    use_vertex: bool = False
    vertex_project_env: str = "GOOGLE_VERTEX_PROJECT"
    vertex_location_env: str = "GOOGLE_VERTEX_LOCATION"
    # Anthropic-only. When True, fail over to AWS Bedrock when the
    # direct Anthropic API returns 429 RateLimitError. Bedrock hosts
    # the same Claude snapshots under a separate quota pool. After a
    # 429 the judge stays on Bedrock for a short cooldown window so
    # subsequent calls do not re-trip the same rate limit. AWS auth
    # uses the boto3 default credential chain (env vars, ~/.aws/...,
    # or an IAM role); region is read from ``bedrock_region_env``.
    bedrock_fallback: bool = False
    # Required when bedrock_fallback=True. Bedrock model ids differ
    # from Anthropic API ids: e.g. ``us.anthropic.claude-opus-4-6-v1:0``
    # for the cross-region inference profile of Opus 4.6.
    bedrock_model_id: str | None = None
    bedrock_region_env: str = "AWS_REGION"


class ModelOverride(BaseModel):
    """Per-subject-model overrides for generation parameters.

    Subject models in ``Config.models`` use the top-level ``max_tokens`` and
    ``max_tokens_ceiling`` by default. A model listed here overrides either
    value, e.g. to lower the ceiling to the provider's output limit.
    """
    max_tokens: int | None = None
    max_tokens_ceiling: int | None = None


def _key_from(identifier: str) -> str:
    """Fallback macro stem: the id's letters, each word capitalized."""
    words = re.split(r"[^A-Za-z]+", identifier)
    return "".join(w[:1].upper() + w[1:] for w in words if w)


def _normalize_entries(raw: list[Any], info_cls) -> tuple[list[str], dict[str, Any]]:
    """Split a YAML list of ids or mappings into ids and per-id info records."""
    ids: list[str] = []
    info: dict[str, Any] = {}
    for entry in raw:
        record = info_cls(id=entry) if isinstance(entry, str) else info_cls(**entry)
        ids.append(record.id)
        info[record.id] = record
    return ids, info


class Config(BaseModel):
    models: list[str]
    judge_models: list[str]
    # Per-id records parsed from mapping entries of ``models`` / ``judge_models``.
    model_info: dict[str, ModelInfo] = {}
    judge_info: dict[str, JudgeInfo] = {}
    tiers: dict[str, TierInfo] = {}
    human_baseline: HumanBaseline = HumanBaseline()
    # Figures quoted from the literature, which no script can compute.
    literature: dict[str, Any] = {}
    prompt_variants: list[PromptVariant]
    prompt_paraphrases: list[PromptVariant] = []
    # Canonical per-cell sample count. Every pipeline stage
    # (generate / extract / score / analyze / human) clips to the first
    # ``samples_per_condition`` records per (subject_model, prompt_variant)
    # cell, ordered ascending by the ``sample_index`` field on each
    # generation record (set in generate.py). On-disk records with
    # ``sample_index >= samples_per_condition`` (e.g. from an earlier run
    # at a higher N) remain on disk but are silently skipped.
    samples_per_condition: int = 30
    # Per-cell sample count for the Prompt Paraphrase Sensitivity ablation
    # (alienbench/ablation_prompt.py). Lower than ``samples_per_condition``
    # because each paraphrase costs a full generate+extract+score pass over
    # every (subject_model, judge) cell; the ablation trades per-cell
    # precision for breadth across paraphrases. The ``baseline`` paraphrase
    # shares on-disk records with the main pipeline (see
    # ``_check_paraphrase_id_collision`` below) and therefore inherits the
    # main-pipeline per-cell N at that cell only.
    prompt_ablation_samples_per_condition: int = 10
    temperature: float = 1.0
    # Output-token limits. A call starts at ``max_tokens``. A response cut off
    # at the limit is not recorded. The call is repeated with the limit
    # doubled, up to ``max_tokens_ceiling``. A response still cut off at the
    # ceiling is not recorded either, and the run reports it as a failure.
    max_tokens: int = 16000
    max_tokens_ceiling: int = 64000
    judge_max_tokens: int = 8000
    judge_max_tokens_ceiling: int = 32000
    data_dir: str = "data"
    results_dir: str = "results"
    api_key_env: str = "OPENROUTER_API_KEY"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    allowed_providers: list[str] | None = None
    allow_provider_fallbacks: bool = False
    judge_overrides: dict[str, JudgeOverride] = {}
    model_overrides: dict[str, ModelOverride] = {}
    primary_metric: str = "ward"

    @model_validator(mode="before")
    @classmethod
    def _split_entries(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if "models" in data and "model_info" not in data:
            data["models"], data["model_info"] = _normalize_entries(data["models"], ModelInfo)
        if "judge_models" in data and "judge_info" not in data:
            data["judge_models"], data["judge_info"] = _normalize_entries(
                data["judge_models"], JudgeInfo
            )
        return data

    def only_models(self, selectors: list[str]) -> "Config":
        """Return a copy whose subject models are those named in ``selectors``.

        A selector is a model's id, its ``key``, or its ``name``; key and
        name match case-insensitively. The copy keeps the config order and
        the full judge panel. Raises ``ValueError`` for an unknown selector.
        """
        def matches(model: str, selector: str) -> bool:
            info = self.model_info.get(model)
            names = {self.model_key(model).lower()}
            if info is not None and info.name:
                names.add(info.name.lower())
            return selector == model or selector.lower() in names

        unknown = [s for s in selectors if not any(matches(m, s) for m in self.models)]
        if unknown:
            raise ValueError(
                f"Unknown model(s) {unknown}. Use an id, key, or name from "
                f"`models` in the config: {', '.join(self.models)}."
            )
        chosen = [m for m in self.models if any(matches(m, s) for s in selectors)]
        return self.model_copy(update={"models": chosen})

    def model_name(self, model: str) -> str:
        """Display name of a subject model, falling back to the id's last segment."""
        info = self.model_info.get(model)
        if info is not None and info.name:
            return info.name
        return model.split("/", 1)[-1]

    def model_key(self, model: str) -> str:
        """Letters-only macro stem of a subject model."""
        info = self.model_info.get(model)
        return info.key if info is not None and info.key else _key_from(model)

    def judge_name(self, judge: str) -> str:
        info = self.judge_info.get(judge)
        if info is not None and info.name:
            return info.name
        return judge.split("/", 1)[-1]

    def judge_key(self, judge: str) -> str:
        info = self.judge_info.get(judge)
        return info.key if info is not None and info.key else _key_from(judge)

    def provider_for(self, model: str) -> str | None:
        """OpenRouter provider slug pinned for a subject model, or None."""
        info = self.model_info.get(model)
        return info.provider if info is not None else None

    def allow_fallbacks_for(self, model: str) -> bool:
        """Whether a subject model may route away from its pinned provider."""
        info = self.model_info.get(model)
        return info.allow_fallbacks if info is not None else False

    def judge_provider_for(self, judge: str) -> str | None:
        """Provider slug pinned for a judge alias, or None."""
        info = self.judge_info.get(judge)
        return info.provider if info is not None else None

    def max_tokens_for(self, model: str) -> int:
        """Output-token limit of the first generation call for a subject model."""
        override = self.model_overrides.get(model)
        if override is not None and override.max_tokens is not None:
            return override.max_tokens
        return self.max_tokens

    def max_tokens_ceiling_for(self, model: str) -> int:
        """Largest output-token limit a generation call may escalate to."""
        override = self.model_overrides.get(model)
        if override is not None and override.max_tokens_ceiling is not None:
            return override.max_tokens_ceiling
        return self.max_tokens_ceiling

    def judge_max_tokens_for(self, judge: str) -> int:
        """Output-token limit of the first extraction call for a judge."""
        info = self.judge_info.get(judge)
        if info is not None and info.max_tokens is not None:
            return info.max_tokens
        return self.judge_max_tokens

    def judge_max_tokens_ceiling_for(self, judge: str) -> int:
        """Largest output-token limit an extraction call may escalate to."""
        info = self.judge_info.get(judge)
        if info is not None and info.max_tokens_ceiling is not None:
            return info.max_tokens_ceiling
        return self.judge_max_tokens_ceiling

    @model_validator(mode="after")
    def _check_token_limits(self) -> "Config":
        """Every start limit must be positive and at most its ceiling."""
        pairs = [(m, self.max_tokens_for(m), self.max_tokens_ceiling_for(m))
                 for m in self.models]
        pairs += [(j, self.judge_max_tokens_for(j), self.judge_max_tokens_ceiling_for(j))
                  for j in self.judge_models]
        bad = [f"{who}: {start} > {ceiling}" for who, start, ceiling in pairs
               if start < 1 or start > ceiling]
        if bad:
            raise ValueError(
                "max_tokens must be positive and at most its ceiling: " + "; ".join(bad))
        return self

    @field_validator("samples_per_condition")
    @classmethod
    def positive_samples(cls, v: int) -> int:
        if v < 1:
            raise ValueError("samples_per_condition must be at least 1")
        return v

    @field_validator("prompt_ablation_samples_per_condition")
    @classmethod
    def positive_prompt_ablation_samples(cls, v: int) -> int:
        if v < 1:
            raise ValueError("prompt_ablation_samples_per_condition must be at least 1")
        return v

    @field_validator("prompt_variants", "prompt_paraphrases")
    @classmethod
    def unique_prompt_ids(cls, v: list[PromptVariant]) -> list[PromptVariant]:
        ids = [p.id for p in v]
        dupes = {x for x in ids if ids.count(x) > 1}
        if dupes:
            raise ValueError(f"prompt ids must be unique, duplicates: {sorted(dupes)}")
        return v

    @model_validator(mode="after")
    def _check_model_entries(self) -> "Config":
        """Catch model entries that would break the paper stage late in a run.

        Model ids and explicit macro keys must be unique, and keys letters only, or
        ``alienbench paper`` fails after the API stages have run. When any
        model states ``open_weights``, all must, or a model would drop out of
        both groups of the open-weight comparison without a warning.
        """
        dupes = sorted({m for m in self.models if self.models.count(m) > 1})
        if dupes:
            raise ValueError(f"model ids must be unique, duplicates: {dupes}")
        # Explicit keys only. A bare id (as in small test configs) falls back
        # to a derived key that only matters if the paper stage runs.
        keys = [self.model_info[m].key for m in self.models
                if m in self.model_info and self.model_info[m].key]
        bad = [k for k in keys if not k.isalpha()]
        if bad:
            raise ValueError(f"model keys must contain letters only: {bad}")
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        if dupes:
            raise ValueError(f"model keys must be unique, duplicates: {dupes}")
        stated = [m for m in self.models
                  if m in self.model_info and self.model_info[m].open_weights is not None]
        if stated and len(stated) != len(self.models):
            missing = [m for m in self.models if m not in stated]
            raise ValueError(f"open_weights is missing for: {missing}")
        return self

    @model_validator(mode="after")
    def _check_paraphrase_id_collision(self) -> "Config":
        main_ids = {p.id for p in self.prompt_variants}
        para_ids = {p.id for p in self.prompt_paraphrases}
        overlap = main_ids & para_ids
        # Baseline is allowed to appear in both — it anchors the paraphrase
        # comparison against the wording used in the primary analysis and
        # shares the same on-disk generations/scores.
        overlap.discard("baseline")
        if overlap:
            raise ValueError(
                "`prompt_variants` and `prompt_paraphrases` share non-baseline ids: "
                f"{sorted(overlap)}. Rename the paraphrase entries to avoid collisions "
                "in data_dir (paths are keyed by prompt id)."
            )
        return self

    @model_validator(mode="after")
    def _check_bedrock_fields(self) -> "Config":
        for alias, override in self.judge_overrides.items():
            if override.bedrock_fallback and not override.bedrock_model_id:
                raise ValueError(
                    f"Judge override {alias!r} sets bedrock_fallback=true but no "
                    f"bedrock_model_id. Set the Bedrock model id (e.g. "
                    f"'us.anthropic.claude-opus-4-6-v1:0') so the failover knows "
                    f"which Bedrock snapshot to call."
                )
            if override.bedrock_fallback and override.provider != "anthropic":
                raise ValueError(
                    f"Judge override {alias!r} sets bedrock_fallback=true on "
                    f"provider {override.provider!r}; Bedrock failover is only "
                    f"defined for the Anthropic judge."
                )
        return self

    @model_validator(mode="after")
    def _check_judge_providers(self) -> "Config":
        """A native judge's provider slug must name the override's SDK.

        Only OpenAI exposes a variant (``service_tier``) on the native path.
        """
        for alias in self.judge_models:
            slug = self.judge_provider_for(alias)
            override = self.judge_overrides.get(alias)
            if not slug or override is None or override.provider == "openrouter":
                continue
            base, _, variant = slug.partition("/")
            if base != override.provider:
                raise ValueError(
                    f"Judge {alias!r} sets provider {slug!r}, but its "
                    f"judge_overrides entry routes to {override.provider!r}."
                )
            if variant and override.provider != "openai":
                raise ValueError(
                    f"Judge {alias!r} sets provider variant {variant!r}. "
                    f"Variants are only supported on the native OpenAI judge."
                )
        return self

    @model_validator(mode="after")
    def _check_override_keys(self) -> "Config":
        unknown = set(self.judge_overrides) - set(self.judge_models)
        if unknown:
            raise ValueError(
                f"`judge_overrides` references aliases not in `judge_models`: {sorted(unknown)}."
            )
        unknown_subjects = set(self.model_overrides) - set(self.models)
        if unknown_subjects:
            raise ValueError(
                f"`model_overrides` references models not in `models`: {sorted(unknown_subjects)}."
            )
        return self

    @property
    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise EnvironmentError(
                f"Environment variable {self.api_key_env!r} not set. "
                "Copy .env.example to .env and paste your OpenRouter API key."
            )
        return key


def load_config(path: str | Path = "config.yaml") -> Config:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Config file not found: {p}. "
            "See config.yaml for an annotated example."
        )
    with open(p) as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}
    cfg = Config(**data)

    # Resolve relative `data_dir` / `results_dir` against the config file's
    # directory so the benchmark is self-contained: invocation works from any
    # CWD, and outputs land next to the config that produced them.
    base = p.parent.resolve()
    if not Path(cfg.data_dir).is_absolute():
        cfg.data_dir = str(base / cfg.data_dir)
    if not Path(cfg.results_dir).is_absolute():
        cfg.results_dir = str(base / cfg.results_dir)
    return cfg
