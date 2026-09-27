"""
No-network tests for backtest.py (project-backlog#601, #616): table
rendering, the tournament guard, the open-question-source filter, config
->model-list building, and the missing-key exit path. Mirrors
tests/test_config.py's style (plain pytest functions, stub objects instead
of real network objects) -- see that file's own module docstring for why
importing here doesn't trigger any network call. `backtest` imports
`forecasting_tools` at module level (same as `main.py`), so this file
transitively depends on that package being installed -- same as
test_config.py already depends on it via `main` -- but never makes a
network call itself: every `MetaculusClient`/`ApiFilter` use below is
either a stub, a monkeypatched fake, or a pure in-memory pydantic object.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from forecasting_tools import MetaculusClient

import backtest
from backtest import (
    CONFIGS,
    BacktestResult,
    HAIKU_4_5,
    KIMI_K3,
    ModelSpec,
    OPUS_5,
    QuestionResult,
    SONNET_5,
    TournamentQuestionError,
    build_models_for_config,
    fetch_open_binary_questions,
    guard_against_tournament_questions,
    main,
    missing_env_var,
    render_markdown_table,
    score_prediction,
)


# ---------------------------------------------------------------------------
# Config -> model list
# ---------------------------------------------------------------------------


def test_config_a_is_sonnet_5_times_5():
    assert build_models_for_config("A") == [ModelSpec(SONNET_5, 5)]


def test_config_b_is_opus_5_times_5():
    assert build_models_for_config("B") == [ModelSpec(OPUS_5, 5)]


def test_config_c_is_sonnet_3_plus_opus_2():
    assert build_models_for_config("C") == [
        ModelSpec(SONNET_5, 3),
        ModelSpec(OPUS_5, 2),
    ]


def test_config_d_is_sonnet_2_plus_kimi_2_plus_haiku_1():
    assert build_models_for_config("D") == [
        ModelSpec(SONNET_5, 2),
        ModelSpec(KIMI_K3, 2),
        ModelSpec(HAIKU_4_5, 1),
    ]


def test_all_four_configs_defined():
    assert set(CONFIGS) == {"A", "B", "C", "D"}


def test_unknown_config_raises():
    with pytest.raises(ValueError):
        build_models_for_config("Z")


def test_every_config_total_samples_between_2_and_5():
    # Ticket's own research note: "a median of 2-5 diverse models beats a
    # single model" -- each config's total sample count should land there.
    for config, specs in CONFIGS.items():
        total_samples = sum(spec.samples for spec in specs)
        assert 2 <= total_samples <= 5, f"config {config} has {total_samples} samples"


# ---------------------------------------------------------------------------
# Tournament guard (project-backlog#616: replaces the old open-question
# guard now that forecasting open main-site questions is the point).
# ---------------------------------------------------------------------------


class _StubQuestion:
    def __init__(
        self,
        id_of_question=1,
        page_url=None,
        tournament_slugs=None,
        default_project_id=None,
    ):
        self.id_of_question = id_of_question
        self.page_url = page_url or f"https://www.metaculus.com/questions/{id_of_question}/"
        self.tournament_slugs = tournament_slugs or []
        self.default_project_id = default_project_id


def test_guard_allows_a_main_site_question():
    question = _StubQuestion()
    guard_against_tournament_questions([question])  # must not raise


def test_guard_refuses_ai_competition_futureeval_question():
    question = _StubQuestion(default_project_id=MetaculusClient.CURRENT_AI_COMPETITION_ID)
    with pytest.raises(TournamentQuestionError):
        guard_against_tournament_questions([question])


def test_guard_refuses_minibench_question():
    question = _StubQuestion(tournament_slugs=[MetaculusClient.CURRENT_MINIBENCH_ID])
    with pytest.raises(TournamentQuestionError):
        guard_against_tournament_questions([question])


def test_guard_refuses_metaculus_cup_question():
    question = _StubQuestion(default_project_id=MetaculusClient.CURRENT_METACULUS_CUP_ID)
    with pytest.raises(TournamentQuestionError):
        guard_against_tournament_questions([question])


def test_guard_refuses_market_pulse_question():
    question = _StubQuestion(tournament_slugs=[MetaculusClient.CURRENT_MARKET_PULSE_ID])
    with pytest.raises(TournamentQuestionError):
        guard_against_tournament_questions([question])


def test_guard_refuses_a_tournament_url_even_without_project_fields():
    question = _StubQuestion(
        page_url="https://www.metaculus.com/tournament/fall-futureeval-2026/"
    )
    with pytest.raises(TournamentQuestionError):
        guard_against_tournament_questions([question])


def test_guard_reports_all_offenders_not_just_the_first():
    good = _StubQuestion(id_of_question=1)
    bad_one = _StubQuestion(
        id_of_question=2, tournament_slugs=[MetaculusClient.CURRENT_MINIBENCH_ID]
    )
    bad_two = _StubQuestion(
        id_of_question=3, default_project_id=MetaculusClient.CURRENT_METACULUS_CUP_ID
    )
    with pytest.raises(TournamentQuestionError) as excinfo:
        guard_against_tournament_questions([good, bad_one, bad_two])
    assert "2" in str(excinfo.value)
    assert "3" in str(excinfo.value)


def test_guard_refuses_when_question_exposes_no_tournament_field_at_all():
    # Ticket #616 stop condition: if a question object exposes no
    # tournament-membership field at all, refuse to guess -- don't guess.
    class _BareQuestion:
        id_of_question = 1
        page_url = "https://www.metaculus.com/questions/1/"

    with pytest.raises(TournamentQuestionError):
        guard_against_tournament_questions([_BareQuestion()])


# ---------------------------------------------------------------------------
# Open-question-source filter (deliverable #1: allowed_statuses=["open"],
# includes_bots_in_aggregates=False -- asserted on the real, constructed
# ApiFilter via a monkeypatched MetaculusClient, no network).
# ---------------------------------------------------------------------------


def test_fetch_open_binary_questions_builds_the_benchmarker_filter(monkeypatch):
    captured = {}

    class _FakeClient:
        async def get_questions_matching_filter(self, api_filter, **kwargs):
            captured["api_filter"] = api_filter
            captured["kwargs"] = kwargs
            return []

    monkeypatch.setattr(backtest, "MetaculusClient", lambda: _FakeClient())

    questions = asyncio.run(fetch_open_binary_questions(10))

    assert questions == []
    api_filter = captured["api_filter"]
    assert api_filter.allowed_statuses == ["open"]
    assert api_filter.allowed_types == ["binary"]
    assert api_filter.includes_bots_in_aggregates is False
    assert api_filter.community_prediction_exists is True
    assert api_filter.num_forecasters_gte == 30
    assert captured["kwargs"]["num_questions"] == 10
    assert captured["kwargs"]["randomly_sample"] is True


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_score_prediction_returns_none_without_community_prediction():
    assert score_prediction(0.7, None) == (None, None)


def test_score_prediction_brier_matches_squared_error():
    brier, _ = score_prediction(0.8, 0.6)
    assert brier == pytest.approx((0.8 - 0.6) ** 2)


def test_score_prediction_baseline_score_is_zero_for_a_perfect_call():
    # A prediction that exactly matches the community at 0.5 has zero log
    # score relative to itself either way -- baseline score should be ~0.
    _, baseline = score_prediction(0.5, 0.5)
    assert baseline == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Table rendering from a canned result
# ---------------------------------------------------------------------------


def _canned_result() -> BacktestResult:
    return BacktestResult(
        config="A",
        questions_requested=2,
        wall_time_seconds=12.5,
        partial=False,
        question_results=[
            QuestionResult(
                question_id=1,
                page_url="https://www.metaculus.com/questions/1/",
                prediction=0.7,
                community_prediction=0.6,
                brier=0.01,
                baseline_score=5.5,
                usd=0.05,
            ),
            QuestionResult(
                question_id=2,
                page_url="https://www.metaculus.com/questions/2/",
                prediction=0.3,
                community_prediction=0.4,
                brier=0.01,
                baseline_score=6.5,
                usd=0.09,
            ),
        ],
    )


def test_render_markdown_table_has_the_required_columns():
    table = render_markdown_table([_canned_result()])
    header = table.splitlines()[0]
    for column in (
        "config",
        "questions",
        "baseline score",
        "brier",
        "$/question",
        "wall time",
    ):
        assert column in header


def test_render_markdown_table_reports_means_and_max():
    result = _canned_result()
    table = render_markdown_table([result])
    row = table.splitlines()[2]
    assert "| A |" in row
    assert "6.00" in row  # mean baseline score, (5.5 + 6.5) / 2
    assert "0.0900" in row  # max $/question
    assert "0.0700" in row  # mean $/question, (0.05 + 0.09) / 2
    assert "12.5s" in row


def test_render_markdown_table_flags_partial_results():
    result = _canned_result()
    result.partial = True
    table = render_markdown_table([result])
    assert "partial" in table


def test_backtest_result_json_dict_is_json_serializable():
    import json

    json.dumps(_canned_result().to_json_dict())  # must not raise


# ---------------------------------------------------------------------------
# Missing-key exit (deliverable #3)
# ---------------------------------------------------------------------------


def test_missing_env_var_reports_the_name_when_unset(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert missing_env_var() == "OPENROUTER_API_KEY"


def test_missing_env_var_is_none_when_set(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    assert missing_env_var() is None


def test_missing_env_var_treats_blank_string_as_unset(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "   ")
    assert missing_env_var() == "OPENROUTER_API_KEY"


def test_main_exits_non_zero_with_one_line_message_when_key_missing(monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    exit_code = main(["--config", "A", "--questions", "5"])
    assert exit_code != 0
    captured = capsys.readouterr()
    stderr_lines = [line for line in captured.err.splitlines() if line.strip()]
    assert len(stderr_lines) == 1
    assert "OPENROUTER_API_KEY" in stderr_lines[0]
