"""
No-network config tests (project-backlog#599): the `llms` researcher
selection, and the per-question/run-total cost log line formatting.

Importing `main` here does trigger its module-level `from forecasting_tools
import (...)` -- that's a local package import, not network -- but never
executes the `if __name__ == "__main__":` block (so no METACULUS_TOKEN /
OPENROUTER_API_KEY / network calls happen just by importing).
"""

import asyncio
import logging
from unittest import mock

from forecasting_tools import ForecastBot, GeneralLlm

from main import (
    FableForecastBot,
    format_bot_cost_line,
    format_bot_cost_total_line,
    select_researcher,
)


class StubReport:
    """Stand-in for a forecasting_tools ForecastReport: only the one
    attribute FableForecastBot._run_individual_question actually reads."""

    def __init__(self, price_estimate):
        self.price_estimate = price_estimate


def test_select_researcher_uses_asknews_when_both_env_vars_set():
    assert select_researcher("client-id", "secret") == "asknews/news-summaries"


def test_select_researcher_falls_back_when_client_id_missing():
    assert select_researcher(None, "secret") == "openrouter/moonshotai/kimi-k3"


def test_select_researcher_falls_back_when_secret_missing():
    assert select_researcher("client-id", None) == "openrouter/moonshotai/kimi-k3"


def test_select_researcher_falls_back_when_both_missing():
    assert select_researcher(None, None) == "openrouter/moonshotai/kimi-k3"


def test_select_researcher_falls_back_on_empty_strings():
    # An env var present but empty should not count as "set".
    assert select_researcher("", "") == "openrouter/moonshotai/kimi-k3"


def test_format_bot_cost_line_priced():
    stub_report = StubReport(price_estimate=0.1234)
    line = format_bot_cost_line(
        question_id=42,
        url="https://www.metaculus.com/questions/42/",
        usd=stub_report.price_estimate,
        researcher="asknews/news-summaries",
        default_model="openrouter/anthropic/claude-sonnet-5",
    )
    assert line == (
        "event=bot_cost question_id=42 url=https://www.metaculus.com/questions/42/ "
        "usd=0.1234 researcher=asknews/news-summaries default=openrouter/anthropic/claude-sonnet-5"
    )
    assert "input_tokens" not in line


def test_format_bot_cost_line_falls_back_to_tokens_when_unpriced():
    stub_report = StubReport(price_estimate=0.0)
    line = format_bot_cost_line(
        question_id=7,
        url="https://www.metaculus.com/questions/7/",
        usd=stub_report.price_estimate,
        researcher="openrouter/moonshotai/kimi-k3",
        default_model="openrouter/anthropic/claude-sonnet-5",
        input_tokens=1500,
        output_tokens=300,
    )
    assert "usd=0.0000" in line
    assert "input_tokens=1500 output_tokens=300" in line


def test_format_bot_cost_line_falls_back_to_tokens_when_usd_rounds_to_zero():
    # usd is nonzero but rounds to "0.0000" at 4dp -- the fallback must still
    # fire (round(usd, 4) <= 0.0), not just the literal usd == 0.0 case above.
    stub_report = StubReport(price_estimate=0.00004)
    line = format_bot_cost_line(
        question_id=8,
        url="https://www.metaculus.com/questions/8/",
        usd=stub_report.price_estimate,
        researcher="openrouter/moonshotai/kimi-k3",
        default_model="openrouter/anthropic/claude-sonnet-5",
        input_tokens=50,
        output_tokens=10,
    )
    assert "usd=0.0000" in line
    assert "input_tokens=50 output_tokens=10" in line


def test_format_bot_cost_total_line():
    line = format_bot_cost_total_line(questions=4, total_usd=0.80)
    assert line == "event=bot_cost_total questions=4 usd=0.8000 mean_usd=0.2000"


def test_format_bot_cost_total_line_zero_questions_does_not_divide_by_zero():
    line = format_bot_cost_total_line(questions=0, total_usd=0.0)
    assert line == "event=bot_cost_total questions=0 usd=0.0000 mean_usd=0.0000"


class _StubQuestion:
    """Stand-in for a MetaculusQuestion: only the three attributes
    FableForecastBot._run_individual_question reads off the question."""

    id_of_question = 42
    id_of_post = None
    page_url = "https://www.metaculus.com/questions/42/"


def test_run_individual_question_logs_bot_cost_and_records_price(caplog):
    """
    Exercises the actual SDK-override path (main.py's
    FableForecastBot._run_individual_question), not just the pure
    formatter it calls. Monkeypatches the PARENT
    ForecastBot._run_individual_question -- the forecasting_tools 0.3.1
    method this override wraps -- so no research/forecast/network runs,
    then asserts on the logged `event=bot_cost` line and on
    `_question_costs_usd`, closing the gap the review flagged: nothing
    previously exercised this hook, so a signature/attribute break in
    forecasting_tools (e.g. `price_estimate` renamed or the method
    removed) would go undetected.

    Uses asyncio.run rather than pytest-asyncio -- that plugin isn't a
    declared dev dependency here and this test doesn't need it.
    """

    class _StubReport:
        def __init__(self, price_estimate: float) -> None:
            self.price_estimate = price_estimate

    async def fake_parent_run_individual_question(self, question):
        return _StubReport(price_estimate=0.1234)

    bot = FableForecastBot(
        llms={
            "default": GeneralLlm(model="openrouter/anthropic/claude-sonnet-5"),
            "summarizer": "openrouter/anthropic/claude-haiku-4.5",
            "researcher": "asknews/news-summaries",
            "parser": "openrouter/anthropic/claude-haiku-4.5",
        },
    )

    with mock.patch.object(
        ForecastBot,
        "_run_individual_question",
        fake_parent_run_individual_question,
    ):
        with caplog.at_level(logging.INFO, logger="main"):
            report = asyncio.run(bot._run_individual_question(_StubQuestion()))

    assert report.price_estimate == 0.1234
    assert bot._question_costs_usd == [0.1234]
    [logged_line] = [
        record.message for record in caplog.records if "event=bot_cost" in record.message
    ]
    assert logged_line == (
        "event=bot_cost question_id=42 "
        "url=https://www.metaculus.com/questions/42/ usd=0.1234 "
        "researcher=asknews/news-summaries "
        "default=openrouter/anthropic/claude-sonnet-5"
    )
