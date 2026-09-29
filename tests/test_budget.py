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
from forecasting_tools.ai_models.resource_managers.hard_limit_manager import (
    HardLimitExceededError,
)
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


class _StubResearchQuestion:
    """Enough of a MetaculusQuestion for run_research's prompt formatting
    and _run_individual_question's question_id lookup -- not a real
    MetaculusQuestion (avoids any network/Pydantic-model setup)."""

    def __init__(self, qid: int) -> None:
        self.id_of_question = qid
        self.id_of_post = None
        self.page_url = f"https://www.metaculus.com/questions/{qid}/"
        self.question_text = "Will X happen?"
        self.resolution_criteria = "Resolves YES if X happens."
        self.fine_print = ""


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


# ---- per-question stop: _run_individual_question guard ---------------------


def test_run_individual_question_does_not_dispatch_when_run_budget_exhausted(caplog):
    """Reviewer-requested per-question test (PR #7 round 1): parent mocked,
    run manager already exhausted, assert the parent is never called."""
    bot = _minimal_bot()
    parent_calls: list[object] = []

    async def fake_parent_run_individual_question(self, question):
        parent_calls.append(question)
        return _StubReport(price_estimate=0.0)

    with mock.patch.object(
        ForecastBot, "_run_individual_question", fake_parent_run_individual_question
    ):
        with caplog.at_level(logging.WARNING, logger="main"):
            with MonetaryCostManager(hard_limit=1.0):
                HardLimitManager.increase_current_usage_in_parent_managers(1.0)
                try:
                    asyncio.run(bot._run_individual_question(_StubResearchQuestion(1)))
                    raised = False
                except HardLimitExceededError:
                    raised = True

    assert raised is True
    assert parent_calls == []  # parent never called
    [logged_line] = [
        r.message for r in caplog.records if "event=bot_budget_exhausted" in r.message
    ]
    assert "spent=1.0000 limit=1.00" in logged_line


def test_run_individual_question_dispatches_when_run_budget_not_exhausted():
    bot = _minimal_bot()
    parent_calls: list[object] = []

    async def fake_parent_run_individual_question(self, question):
        parent_calls.append(question)
        return _StubReport(price_estimate=0.1)

    with mock.patch.object(
        ForecastBot, "_run_individual_question", fake_parent_run_individual_question
    ):
        with MonetaryCostManager(hard_limit=1.0):
            report = asyncio.run(
                bot._run_individual_question(_StubResearchQuestion(1))
            )

    assert len(parent_calls) == 1
    assert report.price_estimate == 0.1


# ---- per-question stop: the real serialization point (run_research) -------


def test_run_research_stops_question_after_run_budget_exhausted_mid_batch(monkeypatch):
    """
    This is the test the deliverable actually needs: "budget exhausted
    after N questions and question N+1 isn't run" -- demonstrated against
    real concurrent dispatch, the way forecast_questions() itself dispatches
    a tournament's questions (one asyncio.gather over all of them), not a
    sequential red/green pair of direct calls.

    A check placed only at the top of _run_individual_question, before any
    await (see the test above), does NOT stop question N+1 within a single
    batch like this: asyncio.gather() schedules all N question-tasks up
    front, and each one's synchronous prefix -- everything up to its own
    first real suspension -- runs before question 1's research call
    returns and updates current_usage, so all N would read the same
    pre-exhaustion budget (reproduced independently against a bare
    asyncio.Semaphore(1) while diagnosing this review round: all 5 tasks in
    a 5-task batch observed usage=0 at their pre-check, even though the
    first task's completion alone exceeded the cap).

    run_research's `_concurrency_limiter` (`_max_concurrent_questions = 1`)
    is the one place questions are actually serialized -- the next
    question only enters after the previous one's research call has
    returned -- so a check made right after acquiring it does see an
    up-to-date budget and does stop question N+1. This test dispatches 3
    questions concurrently via asyncio.gather (matching forecast_questions'
    own dispatch); a fake researcher call increases usage by $0.60 and
    yields via a real asyncio.sleep (simulating network latency) before
    returning, so later questions' turn at the semaphore only comes after
    that cost has landed.
    """
    monkeypatch.setenv("METAC_MAX_USD_PER_RUN", "1.0")
    # A GeneralLlm researcher (rather than _minimal_bot()'s asknews one) --
    # AskNewsSearcher() itself raises ValueError in this sandbox with no
    # ASKNEWS_* credentials configured, before ever reaching run_research's
    # cost-incurring call, which would falsely read as "stopped by the
    # budget guard". GeneralLlm.invoke needs no credentials to construct.
    bot = FableForecastBot(
        llms={
            "default": GeneralLlm(model="openrouter/anthropic/claude-sonnet-5"),
            "summarizer": "openrouter/anthropic/claude-haiku-4.5",
            "researcher": GeneralLlm(model="openrouter/moonshotai/kimi-k3"),
            "parser": "openrouter/anthropic/claude-haiku-4.5",
        },
    )
    researcher_calls: list[int] = []

    async def fake_invoke(self, prompt, **kwargs):
        researcher_calls.append(1)
        await asyncio.sleep(0.01)
        HardLimitManager.increase_current_usage_in_parent_managers(0.6)
        return "research"

    monkeypatch.setattr(GeneralLlm, "invoke", fake_invoke)

    questions = [_StubResearchQuestion(i) for i in range(3)]

    async def run_all():
        with MonetaryCostManager(hard_limit=get_max_usd_per_run()):
            return await asyncio.gather(
                *[bot.run_research(q) for q in questions], return_exceptions=True
            )

    results = asyncio.run(run_all())

    # The check is `amount_left <= 0`, evaluated *before* that question's
    # own spend: question 1 passes at $0/$1.0, spends to $0.60 (still
    # $0.40 left, not yet exhausted); question 2 passes at $0.60/$1.0,
    # spends to $1.20 (now exhausted); question 3's check then sees
    # amount_left <= 0 and is stopped before ever calling the researcher.
    assert len(researcher_calls) == 2
    assert results[0] == "research"
    assert results[1] == "research"
    assert isinstance(results[2], HardLimitExceededError)


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
