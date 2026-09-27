"""
Backtest harness for `FableForecastBot` (main.py) against OPEN, main-site
Metaculus questions -- gate G0 evidence (project-backlog#601, corrected by
#616 to fix outcome leakage: the #601 version forecast on resolved
questions and scored against their long-converged community prediction,
which both trivializes the score and lets live research find the actual
outcome).

Rules this file exists to respect:
  - Uses Metaculus's own Benchmarker method (`MetaculusClient.
    get_benchmark_questions`'s `ApiFilter`, forecasting_tools 0.3.1):
    open, main-site, binary questions with a real community prediction and
    >=30 forecasters, sampled randomly. Never publishes
    (`publish_reports_to_metaculus=False` is hardcoded below, not a flag).
  - Refuses (raises) any question that belongs to a bot or cup tournament
    (FutureEval/AIB, MiniBench, Market Pulse, Metaculus Cup) before any LLM
    call -- see `guard_against_tournament_questions`. Only *tournament*
    questions are forbidden; the tournament rules explicitly allow testing
    against main-site questions.
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
    against our own ensemble prediction (see `score_prediction` below),
    scored vs the question's *current* community prediction.

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

from forecasting_tools import ApiFilter, MetaculusClient

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
# Tournament guard (stop condition: never forecast on a bot/cup tournament
# question -- open main-site questions are fine, per project-backlog#616).
# ---------------------------------------------------------------------------


class TournamentQuestionError(RuntimeError):
    """Raised when a question intended for backtesting belongs to a bot or
    cup tournament, or its tournament membership can't be determined at all."""


# The four "current" bot/cup tournament identifiers this guard refuses --
# sourced directly from forecasting_tools.helpers.metaculus_client.
# MetaculusClient (0.3.1) rather than hardcoded, since these rotate every
# season. Two are numeric project ids, two are string slugs -- see the
# comment in guard_against_tournament_questions below for how each type is
# matched against a question.
FORBIDDEN_TOURNAMENT_IDS: tuple[str | int, ...] = (
    MetaculusClient.CURRENT_AI_COMPETITION_ID,  # FutureEval/AIB, e.g. 33121 (int project id)
    MetaculusClient.CURRENT_MINIBENCH_ID,  # "minibench" (string slug)
    MetaculusClient.CURRENT_METACULUS_CUP_ID,  # Metaculus Cup, e.g. 33108 (int project id)
    MetaculusClient.CURRENT_MARKET_PULSE_ID,  # e.g. "market-pulse-26q4" (string slug)
)


def guard_against_tournament_questions(questions: Sequence) -> None:
    """
    Refuses (raises TournamentQuestionError) any question that belongs to a
    bot or cup tournament -- the tournament rules forbid backtesting against
    open FutureEval/AIB, MiniBench, Market Pulse or Metaculus Cup questions,
    even though open main-site questions are explicitly allowed. Must run
    before any LLM call.

    Tournament membership comes from two `MetaculusQuestion` fields exposed
    by forecasting_tools 0.3.1 (data_models/questions.py:99-100):
      - `tournament_slugs: list[str]` -- string slugs (e.g. "minibench"),
        populated from the API's `projects.tournament` /
        `projects.question_series` lists (questions.py:143-151).
      - `default_project_id: int | None` -- the numeric id of the
        question's default project (`projects.default_project.id`,
        questions.py:194-198); this is what lines up with the *integer*
        tournament ids (`CURRENT_AI_COMPETITION_ID`,
        `CURRENT_METACULUS_CUP_ID`), which `tournament_slugs` (string
        slugs) never will.
    Both are checked against `FORBIDDEN_TOURNAMENT_IDS` above, plus a
    `page_url` check for a "/tournament/" URL as a second, independent net.

    If a question object exposes neither field at all, this refuses to
    guess and raises -- ticket #616's stop condition ("if question objects
    don't expose tournament membership, stop and report").
    """
    offenders = []
    for question in questions:
        if not hasattr(question, "tournament_slugs") and not hasattr(
            question, "default_project_id"
        ):
            raise TournamentQuestionError(
                "Question object exposes no tournament-membership field "
                "(tournament_slugs/default_project_id) -- refusing to guess; "
                "see project-backlog#616 stop condition."
            )
        tournament_slugs = list(getattr(question, "tournament_slugs", None) or [])
        default_project_id = getattr(question, "default_project_id", None)
        page_url = getattr(question, "page_url", None) or ""

        is_forbidden = (
            "/tournament/" in page_url
            or default_project_id in FORBIDDEN_TOURNAMENT_IDS
            or any(str(tid) in tournament_slugs for tid in FORBIDDEN_TOURNAMENT_IDS)
        )
        if is_forbidden:
            offenders.append(
                getattr(question, "id_of_question", None)
                or getattr(question, "page_url", "<unknown question>")
            )
    if offenders:
        raise TournamentQuestionError(
            f"Refusing to backtest on {len(offenders)} question(s) that "
            f"belong to a bot/cup tournament: {offenders}"
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
        "| config | questions | baseline score | brier vs current community prediction | "
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


async def fetch_open_binary_questions(num_questions: int):
    """Open, main-site binary questions -- Metaculus's own Benchmarker
    method. Rebuilds the exact `ApiFilter`
    `MetaculusClient.get_benchmark_questions` itself builds (forecasting-
    tools 0.3.1, helpers/metaculus_client.py:468-477:
    `allowed_statuses=["open"]`, `allowed_types=["binary"]`,
    `includes_bots_in_aggregates=False`, `community_prediction_exists=True`,
    `num_forecasters_gte=30`) rather than calling that method directly,
    since it calls `asyncio.run` internally and can't be awaited from this
    already-running event loop. Applies `guard_against_tournament_questions`
    as a second, independent safety net regardless of what the API filter
    returns -- only bot/cup tournament questions are forbidden; the
    tournament rules explicitly allow testing against main-site questions."""
    client = MetaculusClient()
    api_filter = ApiFilter(
        allowed_statuses=["open"],
        allowed_types=["binary"],
        includes_bots_in_aggregates=False,
        community_prediction_exists=True,
        num_forecasters_gte=30,
    )
    questions = await client.get_questions_matching_filter(
        api_filter,
        num_questions=num_questions,
        randomly_sample=True,
    )
    guard_against_tournament_questions(questions)
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
        # Question is open (fetched via fetch_open_binary_questions), so
        # this is the *current* community prediction, not a converged
        # resolved-question value (project-backlog#616).
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
    questions = await fetch_open_binary_questions(args.questions)
    result = await run_backtest_for_config(args.config, questions, args.max_usd)
    write_result_json(result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest FableForecastBot against open, main-site Metaculus "
            "questions (gate G0 evidence), scored vs the current community "
            "prediction. Never forecasts on a bot/cup tournament question "
            "and never publishes."
        )
    )
    parser.add_argument(
        "--questions", type=int, default=30, help="Number of open, main-site binary questions to sample (default: 30)"
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
