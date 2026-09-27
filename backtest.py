"""
Backtest harness for `FableForecastBot` (main.py) against already-resolved
Metaculus questions -- gate G0 evidence (project-backlog#601).

Rules this file exists to respect:
  - Only ever fetches questions with status "resolved" and refuses (raises)
    if any question's `close_time` isn't safely in the past -- see
    `guard_against_open_questions`. Never publishes
    (`publish_reports_to_metaculus=False` is hardcoded below, not a flag).
  - Drives the existing `main.FableForecastBot` unmodified. It only ever
    swaps the bot's `llms=` config per `--config`; no prompt or aggregation
    change lands in main.py.
  - `forecasting_tools.cp_benchmarking.benchmarker.Benchmarker` is NOT used
    here even though the ticket names it as an available input: importing
    it (via `BenchmarkForBot` -> `CustomizableBot` -> `agent_wrappers`)
    requires the optional `openai-agents` package, which isn't a declared
    dependency in pyproject.toml and is out of this file's allowed changes
    to add. Instead this drives `ForecastBot.forecast_questions` +
    `MonetaryCostManager` directly -- the same primitives Benchmarker
    itself uses internally -- and reproduces `BinaryReport`'s own
    `expected_baseline_score`/Brier formulas (data_models/binary_report.py)
    against our own ensemble prediction (see `score_prediction` below).

Usage:
    poetry run python backtest.py --questions 30 --config A --max-usd 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

REQUIRED_ENV_VAR = "OPENROUTER_API_KEY"

RESULTS_DIR = Path("backtests")

# ---------------------------------------------------------------------------
# Model configs (ticket #601, research into what wins these tournaments:
# a median of 2-5 diverse models beats a single model). Each entry is a
# (model, sample_count) pair: one `FableForecastBot` instance is built per
# entry, with `predictions_per_research_report=sample_count` -- that bot's
# own median-of-`sample_count` prediction (main.py/forecasting_tools'
# unmodified aggregation) is then pooled with the other entries' bot-level
# predictions and re-medianed here for the ensemble score. For a
# single-entry config (A, B) this degenerates to exactly that one bot's own
# prediction -- no special-casing needed.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    model: str
    samples: int


SONNET_5 = "openrouter/anthropic/claude-sonnet-5"
OPUS_5 = "openrouter/anthropic/claude-opus-5"
HAIKU_4_5 = "openrouter/anthropic/claude-haiku-4.5"
KIMI_K3 = "openrouter/moonshotai/kimi-k3"

CONFIGS: dict[str, list[ModelSpec]] = {
    "A": [ModelSpec(SONNET_5, 5)],
    "B": [ModelSpec(OPUS_5, 5)],
    "C": [ModelSpec(SONNET_5, 3), ModelSpec(OPUS_5, 2)],
    "D": [ModelSpec(SONNET_5, 2), ModelSpec(KIMI_K3, 2), ModelSpec(HAIKU_4_5, 1)],
}


def build_models_for_config(config: str) -> list[ModelSpec]:
    """Pure/no-network: the (model, sample_count) list for one --config letter."""
    try:
        return CONFIGS[config]
    except KeyError:
        raise ValueError(
            f"Unknown config {config!r}; choose one of {sorted(CONFIGS)}"
        ) from None


# ---------------------------------------------------------------------------
# Open-question guard (stop condition: never forecast on open questions).
# ---------------------------------------------------------------------------


class OpenQuestionError(RuntimeError):
    """Raised when a question intended for backtesting isn't safely closed."""


def guard_against_open_questions(questions: Sequence, now: datetime | None = None) -> None:
    """
    Refuses (raises OpenQuestionError) if any question's `close_time` is
    missing or in the future, or its `state` is explicitly "open"/"upcoming".
    A missing close_time can't be proven closed, so it's refused too --
    gate G0's "never forecast on open questions" rule has no exceptions.
    """
    now = now or datetime.now(timezone.utc)
    offenders = []
    for question in questions:
        close_time = getattr(question, "close_time", None)
        state = getattr(question, "state", None)
        state_value = getattr(state, "value", state)
        is_open_state = state_value in ("open", "upcoming")
        if close_time is None or close_time > now or is_open_state:
            offenders.append(
                getattr(question, "id_of_question", None)
                or getattr(question, "page_url", "<unknown question>")
            )
    if offenders:
        raise OpenQuestionError(
            f"Refusing to backtest on {len(offenders)} question(s) not safely "
            f"closed (missing/future close_time or open/upcoming state): {offenders}"
        )


# ---------------------------------------------------------------------------
# Missing-key guard (stop condition: no key in the sandbox -> no live run).
# ---------------------------------------------------------------------------


def missing_env_var(name: str = REQUIRED_ENV_VAR) -> str | None:
    """Returns `name` if unset/blank, else None. Pure, no network."""
    value = os.environ.get(name)
    if value and value.strip():
        return None
    return name


# ---------------------------------------------------------------------------
# Scoring: reproduces BinaryReport.expected_baseline_score / Brier against
# our own (possibly cross-model) ensemble prediction, since that report
# class is built for a single bot's own single prediction.
# ---------------------------------------------------------------------------


@dataclass
class QuestionResult:
    question_id: int | str | None
    page_url: str | None
    prediction: float
    community_prediction: float | None
    brier: float | None
    baseline_score: float | None
    usd: float


def score_prediction(prediction: float, community_prediction: float | None) -> tuple[float | None, float | None]:
    """(brier, expected_baseline_score) vs the community prediction, or (None, None)
    if the question has no community prediction on file."""
    if community_prediction is None:
        return None, None
    p = min(max(prediction, 1e-6), 1 - 1e-6)
    c = community_prediction
    brier = (p - c) ** 2
    baseline_score = 100.0 * (
        c * (math.log2(p) + 1.0) + (1.0 - c) * (math.log2(1.0 - p) + 1.0)
    )
    return brier, baseline_score


# ---------------------------------------------------------------------------
# Backtest result + markdown table rendering (no-network, directly testable).
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    config: str
    questions_requested: int
    wall_time_seconds: float
    partial: bool
    question_results: list[QuestionResult] = field(default_factory=list)

    @property
    def questions_scored(self) -> int:
        return len(self.question_results)

    @property
    def mean_baseline_score(self) -> float | None:
        scores = [q.baseline_score for q in self.question_results if q.baseline_score is not None]
        return statistics.fmean(scores) if scores else None

    @property
    def mean_brier(self) -> float | None:
        briers = [q.brier for q in self.question_results if q.brier is not None]
        return statistics.fmean(briers) if briers else None

    @property
    def mean_usd_per_question(self) -> float:
        costs = [q.usd for q in self.question_results]
        return statistics.fmean(costs) if costs else 0.0

    @property
    def max_usd_per_question(self) -> float:
        costs = [q.usd for q in self.question_results]
        return max(costs) if costs else 0.0

    def to_json_dict(self) -> dict:
        return {
            "config": self.config,
            "questions_requested": self.questions_requested,
            "questions_scored": self.questions_scored,
            "partial": self.partial,
            "mean_baseline_score": self.mean_baseline_score,
            "mean_brier": self.mean_brier,
            "mean_usd_per_question": self.mean_usd_per_question,
            "max_usd_per_question": self.max_usd_per_question,
            "wall_time_seconds": self.wall_time_seconds,
            "question_results": [asdict(q) for q in self.question_results],
        }


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_markdown_table(results: Sequence[BacktestResult]) -> str:
    """Markdown table from one or more BacktestResult -- pure, no network,
    matches deliverable #1's column list exactly."""
    header = (
        "| config | questions | baseline score | brier vs community | "
        "mean $/question | max $/question | wall time |"
    )
    separator = "|---|---|---|---|---|---|---|"
    rows = [header, separator]
    for result in results:
        questions_cell = str(result.questions_scored)
        if result.partial:
            questions_cell += f"/{result.questions_requested} (partial)"
        rows.append(
            "| {config} | {questions} | {baseline} | {brier} | {mean_usd} | {max_usd} | {wall_time:.1f}s |".format(
                config=result.config,
                questions=questions_cell,
                baseline=_fmt(result.mean_baseline_score, 2),
                brier=_fmt(result.mean_brier),
                mean_usd=_fmt(result.mean_usd_per_question),
                max_usd=_fmt(result.max_usd_per_question),
                wall_time=result.wall_time_seconds,
            )
        )
    return "\n".join(rows)


def write_result_json(result: BacktestResult, results_dir: Path = RESULTS_DIR) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path = results_dir / f"{date_str}-{result.config}.json"
    out_path.write_text(json.dumps(result.to_json_dict(), indent=2, sort_keys=True))
    return out_path


# ---------------------------------------------------------------------------
# Live pipeline (network + LLM calls; never exercised by tests -- no key in
# the sandbox, per the ticket's stop conditions).
# ---------------------------------------------------------------------------


async def fetch_resolved_binary_questions(num_questions: int):
    """Resolved (closed + scored) binary questions from the main site --
    NOT `MetaculusClient.get_benchmark_questions`, which as shipped in
    forecasting-tools 0.3.1 filters `allowed_statuses=["open"]` (see
    helpers/metaculus_client.py) and would violate the never-forecast-on-
    open-questions rule. Applies `guard_against_open_questions` as a second,
    independent safety net regardless of what the API filter returns."""
    from forecasting_tools import ApiFilter, MetaculusClient

    client = MetaculusClient()
    api_filter = ApiFilter(
        allowed_statuses=["resolved"],
        allowed_types=["binary"],
        community_prediction_exists=True,
        num_forecasters_gte=30,
    )
    questions = await client.get_questions_matching_filter(
        api_filter,
        num_questions=num_questions,
        randomly_sample=True,
    )
    guard_against_open_questions(questions)
    return questions


async def run_backtest_for_config(config: str, questions: Sequence, max_usd: float) -> BacktestResult:
    from forecasting_tools import GeneralLlm, MonetaryCostManager

    from main import FableForecastBot, select_researcher

    model_specs = build_models_for_config(config)
    researcher = select_researcher(
        os.getenv("ASKNEWS_CLIENT_ID"), os.getenv("ASKNEWS_SECRET")
    )

    predictions_by_question: dict = {}
    usd_by_question: dict = {}
    partial = False

    start = time.monotonic()
    with MonetaryCostManager(hard_limit=max_usd) as cost_manager:
        for spec in model_specs:
            if cost_manager.hard_limit and cost_manager.amount_left <= 0:
                partial = True
                break
            bot = FableForecastBot(
                research_reports_per_question=1,
                predictions_per_research_report=spec.samples,
                use_research_summary_to_forecast=False,
                publish_reports_to_metaculus=False,
                folder_to_save_reports_to=None,
                skip_previously_forecasted_questions=False,
                llms={
                    "default": GeneralLlm(
                        model=spec.model, temperature=0.3, timeout=60, allowed_tries=2
                    ),
                    "summarizer": HAIKU_4_5,
                    "researcher": researcher,
                    "parser": HAIKU_4_5,
                },
            )
            reports = await bot.forecast_questions(list(questions), return_exceptions=True)
            for report in reports:
                if isinstance(report, BaseException):
                    partial = True
                    continue
                qid = report.question.id_of_question or report.question.id_of_post
                predictions_by_question.setdefault(qid, []).append(report.prediction)
                usd_by_question[qid] = usd_by_question.get(qid, 0.0) + (
                    report.price_estimate or 0.0
                )
        if cost_manager.hard_limit and cost_manager.current_usage >= cost_manager.hard_limit:
            partial = True
    wall_time_seconds = time.monotonic() - start

    question_results: list[QuestionResult] = []
    for question in questions:
        qid = question.id_of_question or question.id_of_post
        predictions = predictions_by_question.get(qid)
        if not predictions:
            partial = True
            continue
        ensemble_prediction = statistics.median(predictions)
        community_prediction = getattr(question, "community_prediction_at_access_time", None)
        brier, baseline_score = score_prediction(ensemble_prediction, community_prediction)
        question_results.append(
            QuestionResult(
                question_id=qid,
                page_url=getattr(question, "page_url", None),
                prediction=ensemble_prediction,
                community_prediction=community_prediction,
                brier=brier,
                baseline_score=baseline_score,
                usd=usd_by_question.get(qid, 0.0),
            )
        )

    return BacktestResult(
        config=config,
        questions_requested=len(questions),
        wall_time_seconds=wall_time_seconds,
        partial=partial,
        question_results=question_results,
    )


async def _run(args: argparse.Namespace) -> BacktestResult:
    questions = await fetch_resolved_binary_questions(args.questions)
    result = await run_backtest_for_config(args.config, questions, args.max_usd)
    write_result_json(result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest FableForecastBot against already-resolved Metaculus "
            "questions (gate G0 evidence). Never forecasts on open "
            "questions and never publishes."
        )
    )
    parser.add_argument(
        "--questions", type=int, default=30, help="Number of resolved binary questions to sample (default: 30)"
    )
    parser.add_argument(
        "--config", type=str, required=True, choices=sorted(CONFIGS), help="Model config to backtest"
    )
    parser.add_argument(
        "--max-usd", type=float, default=5.0, help="Hard USD budget cap for this run (default: 5.0)"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    missing = missing_env_var()
    if missing:
        print(f"Missing required environment variable: {missing}", file=sys.stderr)
        return 1

    result = asyncio.run(_run(args))
    print(render_markdown_table([result]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
