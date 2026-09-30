"""Run-spend telemetry and budget-stop exit tests (project-backlog#676). No network."""

import pytest
from forecasting_tools.ai_models.resource_managers.hard_limit_manager import (
    HardLimitExceededError,
)

import main


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


def test_only_budget_stops_leaves_nothing_for_log_report_summary():
    kept, stops = main.split_budget_stops([HardLimitExceededError("x")])
    assert kept == [] and len(stops) == 1


def test_spend_line_format():
    assert main.format_bot_run_spend_line(1.23456, 3.0, 4, 1, 2) == (
        "event=bot_run_spend usd=1.2346 limit=3.0 questions_ok=4 "
        "questions_failed=1 questions_budget_stopped=2"
    )
