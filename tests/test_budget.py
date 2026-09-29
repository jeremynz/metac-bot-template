"""
Gate G0 spend-guard tests (project-backlog#618): pure formatting/config
helpers, the per-question over-cap warning, the tournament-mode
budget-exhausted skip, and the keyless tournament-mode exit.

No network: MonetaryCostManager/HardLimitManager (imported from the
installed forecasting_tools package, same as backtest.py's own precedent)
are pure in-process bookkeeping -- entering the context manager and calling
its classmethod to record usage does not touch litellm or the network.
The keyless-skip test spawns `python3 main.py --mode tournament` as a local
subprocess (no network call itself) because that behavior lives in the
`if __name__ == "__main__":` block, which never runs on a plain `import
main` (see tests/test_config.py's own docstring on that).
"""

import asyncio
import logging
import os
import subprocess
import sys
from unittest import mock

from forecasting_tools import ForecastBot, GeneralLlm, MonetaryCostManager
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    HardLimitManager,
)

from bot_helpers import has_llm_key
from main import (
    FableForecastBot,
    format_bot_budget_exhausted_line,
    format_bot_cost_over_question_cap_line,
    get_max_usd_per_question,
    get_max_usd_per_run,
    run_tournament_mode,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _StubQuestion:
    id_of_question = 42
    id_of_post = None
    page_url = "https://www.metaculus.com/questions/42/"


class _StubReport:
    def __init__(self, price_estimate: float) -> None:
        self.price_estimate = price_estimate


def _minimal_bot() -> FableForecastBot:
    return FableForecastBot(
        llms={
            "default": GeneralLlm(model="openrouter/anthropic/claude-sonnet-5"),
            "summarizer": "openrouter/anthropic/claude-haiku-4.5",
            "researcher": "asknews/news-summaries",
            "parser": "openrouter/anthropic/claude-haiku-4.5",
        },
    )


# ---- pure config helpers ---------------------------------------------------


def test_get_max_usd_per_run_default(monkeypatch):
    monkeypatch.delenv("METAC_MAX_USD_PER_RUN", raising=False)
    assert get_max_usd_per_run() == 3.0


def test_get_max_usd_per_run_reads_env(monkeypatch):
    monkeypatch.setenv("METAC_MAX_USD_PER_RUN", "12.5")
    assert get_max_usd_per_run() == 12.5


def test_get_max_usd_per_question_default(monkeypatch):
    monkeypatch.delenv("METAC_MAX_USD_PER_QUESTION", raising=False)
    assert get_max_usd_per_question() == 0.60


def test_get_max_usd_per_question_reads_env(monkeypatch):
    monkeypatch.setenv("METAC_MAX_USD_PER_QUESTION", "1.25")
    assert get_max_usd_per_question() == 1.25


# ---- pure formatters --------------------------------------------------------


def test_format_bot_budget_exhausted_line():
    line = format_bot_budget_exhausted_line(spent=3.1234, limit=3.0)
    assert line == "event=bot_budget_exhausted spent=3.1234 limit=3.00"


def test_format_bot_cost_over_question_cap_line():
    line = format_bot_cost_over_question_cap_line(question_id=42, usd=0.75, cap=0.60)
    assert line == "event=bot_cost_over_question_cap question_id=42 usd=0.7500 cap=0.60"


# ---- per-question cap: warning only, never skips ---------------------------


def test_run_individual_question_warns_when_over_question_cap(monkeypatch, caplog):
    monkeypatch.setenv("METAC_MAX_USD_PER_QUESTION", "0.60")

    async def fake_parent_run_individual_question(self, question):
        return _StubReport(price_estimate=0.75)

    bot = _minimal_bot()
    with mock.patch.object(
        ForecastBot, "_run_individual_question", fake_parent_run_individual_question
    ):
        with caplog.at_level(logging.WARNING, logger="main"):
            report = asyncio.run(bot._run_individual_question(_StubQuestion()))

    # Warning only -- the question already ran, its report is returned as-is.
    assert report.price_estimate == 0.75
    assert bot._question_costs_usd == [0.75]
    [logged_line] = [
        r.message for r in caplog.records if "event=bot_cost_over_question_cap" in r.message
    ]
    assert logged_line == (
        "event=bot_cost_over_question_cap question_id=42 usd=0.7500 cap=0.60"
    )


def test_run_individual_question_does_not_warn_under_question_cap(monkeypatch, caplog):
    monkeypatch.setenv("METAC_MAX_USD_PER_QUESTION", "0.60")

    async def fake_parent_run_individual_question(self, question):
        return _StubReport(price_estimate=0.10)

    bot = _minimal_bot()
    with mock.patch.object(
        ForecastBot, "_run_individual_question", fake_parent_run_individual_question
    ):
        with caplog.at_level(logging.WARNING, logger="main"):
            asyncio.run(bot._run_individual_question(_StubQuestion()))

    assert not any(
        "event=bot_cost_over_question_cap" in r.message for r in caplog.records
    )


# ---- run-level budget guard: stops dispatching further questions ----------


class _FakeClient:
    CURRENT_AI_COMPETITION_ID = "seasonal-id"
    CURRENT_MINIBENCH_ID = "minibench-id"


def test_tournament_mode_skips_minibench_when_run_budget_exhausted(monkeypatch, caplog):
    """
    Red case: without the guard, the MiniBench call (question N+1's batch)
    runs unconditionally regardless of spend. With the guard, once the
    seasonal call alone exhausts METAC_MAX_USD_PER_RUN, the MiniBench call
    must never be dispatched.
    """
    monkeypatch.setenv("METAC_MAX_USD_PER_RUN", "1.0")
    bot = _minimal_bot()
    calls: list[str] = []

    async def fake_forecast_on_tournament(tournament_id, return_exceptions=True):
        calls.append(tournament_id)
        if tournament_id == _FakeClient.CURRENT_AI_COMPETITION_ID:
            # Simulate the seasonal call alone spending past the $1 cap --
            # this is the real HardLimitManager mechanism litellm callbacks
            # drive in production (forecasting_tools' own MonetaryCostManager
            # calls the same classmethod from its litellm cost callback).
            HardLimitManager.increase_current_usage_in_parent_managers(2.0)
            return [_StubReport(2.0)]
        return [_StubReport(0.0)]  # would only run if the guard failed

    bot.forecast_on_tournament = fake_forecast_on_tournament

    with caplog.at_level(logging.WARNING, logger="main"):
        with MonetaryCostManager(hard_limit=get_max_usd_per_run()) as run_cost_manager:
            reports = run_tournament_mode(bot, _FakeClient(), run_cost_manager)

    assert calls == [_FakeClient.CURRENT_AI_COMPETITION_ID]  # minibench never dispatched
    assert len(reports) == 1
    [logged_line] = [
        r.message for r in caplog.records if "event=bot_budget_exhausted" in r.message
    ]
    assert logged_line == "event=bot_budget_exhausted spent=2.0000 limit=1.00"


def test_tournament_mode_runs_minibench_when_budget_not_exhausted(monkeypatch):
    """Control/green case for the test above: with budget to spare, both
    calls dispatch."""
    monkeypatch.setenv("METAC_MAX_USD_PER_RUN", "10.0")
    bot = _minimal_bot()
    calls: list[str] = []

    async def fake_forecast_on_tournament(tournament_id, return_exceptions=True):
        calls.append(tournament_id)
        HardLimitManager.increase_current_usage_in_parent_managers(0.5)
        return [_StubReport(0.5)]

    bot.forecast_on_tournament = fake_forecast_on_tournament

    with MonetaryCostManager(hard_limit=get_max_usd_per_run()) as run_cost_manager:
        reports = run_tournament_mode(bot, _FakeClient(), run_cost_manager)

    assert calls == [
        _FakeClient.CURRENT_AI_COMPETITION_ID,
        _FakeClient.CURRENT_MINIBENCH_ID,
    ]
    assert len(reports) == 2


def test_tournament_mode_no_hard_limit_never_skips():
    """hard_limit=0 means unlimited (HardLimitManager's own convention --
    amount_left comparisons are gated on `if hard_limit` throughout) --
    both calls must still run."""
    bot = _minimal_bot()
    calls: list[str] = []

    async def fake_forecast_on_tournament(tournament_id, return_exceptions=True):
        calls.append(tournament_id)
        HardLimitManager.increase_current_usage_in_parent_managers(999.0)
        return [_StubReport(999.0)]

    bot.forecast_on_tournament = fake_forecast_on_tournament

    with MonetaryCostManager(hard_limit=0) as run_cost_manager:
        run_tournament_mode(bot, _FakeClient(), run_cost_manager)

    assert calls == [
        _FakeClient.CURRENT_AI_COMPETITION_ID,
        _FakeClient.CURRENT_MINIBENCH_ID,
    ]


# ---- has_llm_key --------------------------------------------------------


def test_has_llm_key_false_when_all_unset(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert has_llm_key() is False


def test_has_llm_key_false_for_template_placeholder(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "REPLACE_ME")
    assert has_llm_key() is False


def test_has_llm_key_true_when_one_set(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-real-key")
    assert has_llm_key() is True


# ---- keyless tournament-mode exits 0 ---------------------------------------


def test_tournament_mode_with_no_llm_key_exits_zero_with_one_log_line():
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
    }
    result = subprocess.run(
        [sys.executable, "main.py", "--mode", "tournament"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "event=bot_skip reason=no_llm_key" in result.stdout
