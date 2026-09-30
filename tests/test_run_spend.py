"""Run-spend telemetry and budget-stop exit tests (project-backlog#676). No network."""

from unittest import mock

import pytest
from forecasting_tools.ai_models.resource_managers.hard_limit_manager import (
    HardLimitExceededError,
)

import main
from tests.test_budget import _minimal_bot


@pytest.mark.parametrize("slug", list(main.PINNED_MODEL_PRICES))
def test_pinned_models_have_nonzero_mock_cost(slug):
    main.ensure_model_pricing()
    assert main.mock_cost_usd(slug) > 0


def test_ensure_model_pricing_registers_when_cost_is_zero(monkeypatch):
    calls = []
    monkeypatch.setattr(main, "mock_cost_usd", lambda m: 0.0)
    monkeypatch.setattr(main.litellm, "register_model", lambda d: calls.append(d))
    assert main.ensure_model_pricing() == list(main.PINNED_MODEL_PRICES)
    haiku = calls[1]["openrouter/anthropic/claude-haiku-4.5"]
    assert haiku["input_cost_per_token"] == 1e-6
    assert haiku["output_cost_per_token"] == 5e-6


def test_ensure_model_pricing_skips_priced_models(monkeypatch):
    monkeypatch.setattr(main, "mock_cost_usd", lambda m: 0.01)
    monkeypatch.setattr(main.litellm, "register_model", lambda d: pytest.fail("called"))
    assert main.ensure_model_pricing() == []


def test_split_budget_stops_keeps_real_errors():
    budget = HardLimitExceededError("over")
    wrapped = RuntimeError("wrapped")
    wrapped.__cause__ = HardLimitExceededError("over")
    real = ValueError("boom")
    kept, stops = main.split_budget_stops(["report", real, budget, wrapped])
    assert kept == ["report", real]
    assert stops == [budget, wrapped]


def _ok_report():
    report = mock.MagicMock()
    report.explanation = "ok"
    return report


def test_only_budget_stops_do_not_raise_in_log_report_summary():
    bot = _minimal_bot()
    reports = [HardLimitExceededError("x"), _ok_report()]
    kept, stops = main.split_budget_stops(reports)
    assert len(stops) == 1
    bot.log_report_summary(kept)
    bot.log_report_summary(main.split_budget_stops([HardLimitExceededError("x")])[0])


def test_real_error_still_raises_in_log_report_summary():
    bot = _minimal_bot()
    kept, _ = main.split_budget_stops(
        [HardLimitExceededError("x"), ValueError("boom"), _ok_report()]
    )
    with pytest.raises(RuntimeError):
        bot.log_report_summary(kept)


def test_spend_line_format():
    assert main.format_bot_run_spend_line(1.23456, 3.0, 4, 1, 2) == (
        "event=bot_run_spend usd=1.2346 limit=3.0 questions_ok=4 "
        "questions_failed=1 questions_budget_stopped=2"
    )
