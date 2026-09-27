"""
No-network config tests (project-backlog#599): the `llms` researcher
selection, and the per-question/run-total cost log line formatting.

Importing `main` here does trigger its module-level `from forecasting_tools
import (...)` -- that's a local package import, not network -- but never
executes the `if __name__ == "__main__":` block (so no METACULUS_TOKEN /
OPENROUTER_API_KEY / network calls happen just by importing).
"""

from main import format_bot_cost_line, format_bot_cost_total_line, select_researcher


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
