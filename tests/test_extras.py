"""#782 flag-gated extras: outside view + Platt calibration. No network."""
import asyncio
import json
import math
import os
import random
import subprocess
import sys

import pytest
from forecasting_tools import GeneralLlm, MonetaryCostManager
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    HardLimitManager,
)

from forecast_extras import calibrate, fit_platt, logit, sigmoid
from main import FableForecastBot
from test_budget import _StubResearchQuestion

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _bot():
    return FableForecastBot(
        llms={
            "default": GeneralLlm(model="openrouter/anthropic/claude-sonnet-5"),
            "summarizer": "openrouter/anthropic/claude-haiku-4.5",
            "researcher": GeneralLlm(model="openrouter/moonshotai/kimi-k3"),
            "parser": "openrouter/anthropic/claude-haiku-4.5",
        },
    )


def _run_research(monkeypatch, cost_per_call=0.0):
    prompts = []

    async def fake_invoke(self, prompt, **kw):
        prompts.append(prompt)
        HardLimitManager.increase_current_usage_in_parent_managers(cost_per_call)
        if "reference class" in prompt:
            return "Reference class: widgets. Base rate: 30%"
        return "research"

    monkeypatch.setattr(GeneralLlm, "invoke", fake_invoke)

    async def go():
        with MonetaryCostManager(hard_limit=10) as m:
            r = await _bot().run_research(_StubResearchQuestion(1))
            return r, m.current_usage

    r, usd = asyncio.run(go())
    return r, usd, prompts


def test_flags_unset_output_identical(monkeypatch):
    monkeypatch.delenv("METAC_OUTSIDE_VIEW", raising=False)
    monkeypatch.delenv("METAC_CALIBRATION", raising=False)
    r, _, prompts = _run_research(monkeypatch)
    assert r == "research"
    assert len(prompts) == 1  # no extra call

    async def agg():
        return await _bot()._aggregate_predictions([0.2, 0.4, 0.9], _binary_q())

    from forecasting_tools import BinaryQuestion  # noqa: F401
    assert asyncio.run(agg()) == pytest.approx(0.4)


def _binary_q():
    from forecasting_tools import BinaryQuestion
    return BinaryQuestion(question_text="q", id_of_post=1, id_of_question=1, page_url="u")


def test_outside_view_reaches_research_and_costs(monkeypatch):
    monkeypatch.setenv("METAC_OUTSIDE_VIEW", "1")
    r, usd, prompts = _run_research(monkeypatch, cost_per_call=0.05)
    assert len(prompts) == 2
    assert "Outside view" in r and "Base rate: 30%" in r and r.startswith("research")
    assert usd == pytest.approx(0.10)  # extra call counted in the cost manager


def test_outside_view_skipped_when_run_budget_exhausted(monkeypatch):
    monkeypatch.setenv("METAC_OUTSIDE_VIEW", "1")
    calls = []

    async def fake_invoke(self, prompt, **kw):
        calls.append(prompt)
        HardLimitManager.increase_current_usage_in_parent_managers(0.06)
        return "research"

    monkeypatch.setattr(GeneralLlm, "invoke", fake_invoke)

    async def go():
        with MonetaryCostManager(hard_limit=0.05):
            return await _bot().run_research(_StubResearchQuestion(1))

    # research call exhausts the run budget -> outside-view call is not made
    assert asyncio.run(go()) == "research"
    assert len(calls) == 1


def test_calibration_transform_and_identity(tmp_path):
    assert calibrate(0.3, None) == 0.3
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"a": 2.0, "b": 0.5}))
    expect = sigmoid(2.0 * logit(0.3) + 0.5)
    assert calibrate(0.3, str(f)) == pytest.approx(expect)
    f.write_text(json.dumps({"a": 1, "b": 0}))
    assert calibrate(0.3, str(f)) == pytest.approx(0.3)


def test_calibration_hook_applied_post_aggregation(monkeypatch, tmp_path):
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"a": 1.0, "b": 1.0}))
    monkeypatch.setenv("METAC_CALIBRATION", str(f))
    out = asyncio.run(_bot()._aggregate_predictions([0.2, 0.4, 0.9], _binary_q()))
    assert out == pytest.approx(sigmoid(logit(0.4) + 1.0))


def test_fit_recovers_known_params(tmp_path):
    rng = random.Random(0)
    a, b = 0.7, -0.4
    ps, ys = [], []
    for _ in range(20000):
        p = rng.uniform(0.03, 0.97)
        ps.append(p)
        ys.append(1 if rng.random() < sigmoid(a * logit(p) + b) else 0)
    fa, fb = fit_platt(ps, ys)
    assert fa == pytest.approx(a, abs=0.05) and fb == pytest.approx(b, abs=0.05)
    csv_path = tmp_path / "p.csv"
    csv_path.write_text("forecast,outcome\n" + "\n".join(f"{p},{y}" for p, y in zip(ps, ys)))
    out = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "scripts", "fit_calibration.py"), str(csv_path)],
        capture_output=True, text=True, check=True,
    )
    d = json.loads(out.stdout)
    assert d["a"] == pytest.approx(a, abs=0.05) and d["b"] == pytest.approx(b, abs=0.05)
