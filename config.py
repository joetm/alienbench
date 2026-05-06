from __future__ import annotations

import os
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

    Subject models in ``Config.models`` use the top-level ``max_tokens`` by
    default. A model listed here overrides that value. Used for thinking
    models whose hidden-reasoning tokens count against ``max_tokens`` and
    therefore need a larger budget than non-thinking subjects.
    """
    max_tokens: int | None = None


class Config(BaseModel):
    models: list[str]
    judge_models: list[str]
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
    max_tokens: int = 800
    data_dir: str = "data"
    results_dir: str = "results"
    api_key_env: str = "OPENROUTER_API_KEY"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    allowed_providers: list[str] | None = None
    allow_provider_fallbacks: bool = False
    judge_overrides: dict[str, JudgeOverride] = {}
    model_overrides: dict[str, ModelOverride] = {}
    primary_metric: str = "ward"

    def max_tokens_for(self, model: str) -> int:
        """Return the effective ``max_tokens`` for a subject model.

        Per-model overrides win over the top-level default. Used by
        ``generate.run`` so thinking models can be given a larger budget
        than non-thinking subjects without disturbing the rest of the panel.
        """
        override = self.model_overrides.get(model)
        if override is not None and override.max_tokens is not None:
            return override.max_tokens
        return self.max_tokens
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
