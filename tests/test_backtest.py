"""
No-network tests for backtest.py (project-backlog#601): table rendering,
the open-question guard, config->model-list building, and the missing-key
exit path. Mirrors tests/test_config.py's style (plain pytest functions,
stub objects instead of real network objects) -- see that file's own
module docstring for why importing here doesn't trigger any network call.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from backtest import (
    CONFIGS,
    BacktestResult,
    HAIKU_4_5,
    KIMI_K3,
    ModelSpec,
    OPUS_5,
    OpenQuestionError,
    QuestionResult,
    SONNET_5,
    build_models_for_config,
    guard_against_open_questions,
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
# Open-question guard
# ---------------------------------------------------------------------------


class _StubQuestion:
    def __init__(self, close_time=None, state=None, id_of_question=1):
        self.close_time = close_time
        self.state = state
        self.id_of_question = id_of_question
        self.page_url = f"https://www.metaculus.com/questions/{id_of_question}/"


def test_guard_allows_a_question_closed_in_the_past():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    question = _StubQuestion(close_time=now - timedelta(days=10))
    guard_against_open_questions([question], now=now)  # must not raise


def test_guard_refuses_a_question_with_close_time_in_the_future():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    question = _StubQuestion(close_time=now + timedelta(days=10))
    with pytest.raises(OpenQuestionError):
        guard_against_open_questions([question], now=now)


def test_guard_refuses_a_question_with_no_close_time():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    question = _StubQuestion(close_time=None)
    with pytest.raises(OpenQuestionError):
        guard_against_open_questions([question], now=now)


def test_guard_refuses_a_question_explicitly_marked_open():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    # Closed in the past by the clock, but still flagged "open" -- the
    # state check is an independent second net, not just close_time.
    question = _StubQuestion(close_time=now - timedelta(days=1), state="open")
    with pytest.raises(OpenQuestionError):
        guard_against_open_questions([question], now=now)


def test_guard_reports_all_offenders_not_just_the_first():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    good = _StubQuestion(close_time=now - timedelta(days=1), id_of_question=1)
    bad_one = _StubQuestion(close_time=now + timedelta(days=1), id_of_question=2)
    bad_two = _StubQuestion(close_time=None, id_of_question=3)
    with pytest.raises(OpenQuestionError) as excinfo:
        guard_against_open_questions([good, bad_one, bad_two], now=now)
    assert "2" in str(excinfo.value)
    assert "3" in str(excinfo.value)


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
