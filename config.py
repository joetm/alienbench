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


class Config(BaseModel):
    models: list[str]
    judge_models: list[str]
    prompt_variants: list[PromptVariant]
    prompt_paraphrases: list[PromptVariant] = []
    samples_per_condition: int = 50
    temperature: float = 1.0
    max_tokens: int = 800
    data_dir: str = "data"
    results_dir: str = "results"
    api_key_env: str = "OPENROUTER_API_KEY"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    allowed_providers: list[str] | None = None
    allow_provider_fallbacks: bool = False
    judge_overrides: dict[str, JudgeOverride] = {}
    primary_metric: str = "ward"
    @field_validator("samples_per_condition")
    @classmethod
    def positive_samples(cls, v: int) -> int:
        if v < 1:
            raise ValueError("samples_per_condition must be at least 1")
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
    def _check_override_keys(self) -> "Config":
        unknown = set(self.judge_overrides) - set(self.judge_models)
        if unknown:
            raise ValueError(
                f"`judge_overrides` references aliases not in `judge_models`: {sorted(unknown)}."
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
