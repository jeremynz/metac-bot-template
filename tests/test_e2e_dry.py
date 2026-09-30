"""Offline end-to-end dry run (project-backlog#674): parse, aggregate and
publish payloads for all four question types with zero network, zero spend.

LLMs are `GeneralLlm(mock_response=...)` (litellm short-circuits before any
HTTP call); the Metaculus client is a stub that records what would be POSTed
and applies the real client's own validation rules.
"""

import logging
from datetime import datetime, timezone

import pytest
from forecasting_tools import (
    BinaryQuestion,
    DateQuestion,
    GeneralLlm,
    MetaculusClient,
    MultipleChoiceQuestion,
    NumericQuestion,
)

from main import FableForecastBot

URL = "https://www.metaculus.com/questions/{}/"


class StubClient:
    """Records POST args; mirrors MetaculusClient's validation."""

    def __init__(self) -> None:
        self.forecasts: list[tuple[int, dict]] = []
        self.comments: list[dict] = []

    def post_binary_question_prediction(self, question_id, prediction_in_decimal):
        if prediction_in_decimal < 0.001 or prediction_in_decimal > 0.999:
            raise ValueError("Prediction value must be between 0.001 and 0.999")
        self.forecasts.append((question_id, {"probability_yes": prediction_in_decimal}))

    def post_numeric_question_prediction(self, question_id, cdf_values):
        if not all(0 <= x <= 1 for x in cdf_values):
            raise ValueError("All CDF values must be between 0 and 1")
        if not all(a <= b for a, b in zip(cdf_values, cdf_values[1:])):
            raise ValueError("CDF values must be monotonically increasing")
        self.forecasts.append((question_id, {"continuous_cdf": cdf_values}))

    def post_multiple_choice_question_prediction(self, question_id, options):
        self.forecasts.append((question_id, {"probability_yes_per_category": options}))

    def post_question_comment(
        self, post_id, comment_text, is_private=True, included_forecast=True
    ):
        self.comments.append(
            {"post_id": post_id, "text": comment_text, "is_private": is_private}
        )


def _common(qid: int, post: int, text: str) -> dict:
    return dict(
        question_text=text,
        id_of_question=qid,
        id_of_post=post,
        page_url=URL.format(post),
        resolution_criteria="Resolves per official data.",
        fine_print="",
        background_info="Background.",
    )


def _bot(client, default_text, parser_text="unused"):
    return FableForecastBot(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=False,
        metaculus_client=client,
        llms={
            "default": GeneralLlm(model="openrouter/mock/default", mock_response=default_text),
            "summarizer": GeneralLlm(model="openrouter/mock/summ", mock_response="summary"),
            "researcher": GeneralLlm(model="openrouter/mock/research", mock_response="Research: nothing new."),
            "parser": GeneralLlm(model="openrouter/mock/parser", mock_response=parser_text),
        },
    )


def _binary(text="Will X happen?"):
    return BinaryQuestion(**_common(101, 1001, text))


def _mc():
    return MultipleChoiceQuestion(
        **_common(102, 1002, "Which?"), options=["A", "B", "C"]
    )


def _numeric():
    return NumericQuestion(
        **_common(103, 1003, "How many?"),
        upper_bound=100.0,
        lower_bound=0.0,
        open_upper_bound=True,
        open_lower_bound=True,
        unit_of_measure="units",
    )


def _date():
    return DateQuestion(
        **_common(104, 1004, "When?"),
        upper_bound=datetime(2027, 12, 31, tzinfo=timezone.utc),
        lower_bound=datetime(2026, 10, 1, tzinfo=timezone.utc),
        open_upper_bound=True,
        open_lower_bound=True,
    )


NUMERIC_TEXT = """Reasoning...
Percentile 10: 10
Percentile 20: 20
Percentile 40: 35
Percentile 60: 50
Percentile 80: 70
Percentile 90: 85"""

DATE_TEXT = """Reasoning...
Percentile 10: 2026-11-01
Percentile 20: 2026-12-01
Percentile 40: 2027-02-01
Percentile 60: 2027-04-01
Percentile 80: 2027-08-01
Percentile 90: 2027-11-01"""

MC_TEXT = "Reasoning...\nA: 50%\nB: 30%\nC: 20%"


def _cost_lines(caplog):
    return [r.getMessage() for r in caplog.records if "event=bot_cost " in r.getMessage()]


def _run(bot, question):
    import asyncio

    return asyncio.run(bot.forecast_questions([question], return_exceptions=True))


def _assert_one_post(client, qid, post_id, caplog):
    assert len(client.forecasts) == 1
    assert client.forecasts[0][0] == qid
    assert len(client.comments) == 1
    assert client.comments[0]["post_id"] == post_id
    assert client.comments[0]["is_private"] is True
    lines = _cost_lines(caplog)
    assert len(lines) == 1 and f"question_id={qid}" in lines[0]


def test_binary(caplog):
    caplog.set_level(logging.INFO)
    client = StubClient()
    _run(_bot(client, "Rationale.\nProbability: 62%"), _binary())
    _assert_one_post(client, 101, 1001, caplog)
    assert client.forecasts[0][1]["probability_yes"] == pytest.approx(0.62)


def test_multiple_choice(caplog):
    caplog.set_level(logging.INFO)
    client = StubClient()
    _run(_bot(client, MC_TEXT), _mc())
    _assert_one_post(client, 102, 1002, caplog)
    probs = client.forecasts[0][1]["probability_yes_per_category"]
    assert set(probs) == {"A", "B", "C"}
    assert sum(probs.values()) == pytest.approx(1.0)


def test_numeric(caplog):
    caplog.set_level(logging.INFO)
    client = StubClient()
    _run(_bot(client, NUMERIC_TEXT), _numeric())
    _assert_one_post(client, 103, 1003, caplog)
    cdf = client.forecasts[0][1]["continuous_cdf"]
    assert len(cdf) == 201
    assert all(a <= b for a, b in zip(cdf, cdf[1:]))
    assert 0 <= cdf[0] and cdf[-1] <= 1


def test_date(caplog):
    caplog.set_level(logging.INFO)
    client = StubClient()
    _run(_bot(client, DATE_TEXT), _date())
    _assert_one_post(client, 104, 1004, caplog)
    cdf = client.forecasts[0][1]["continuous_cdf"]
    assert len(cdf) == 201
    assert all(a <= b for a, b in zip(cdf, cdf[1:]))


def test_tournament_ids():
    assert MetaculusClient.CURRENT_AI_COMPETITION_ID == 33121
    assert MetaculusClient.CURRENT_MINIBENCH_ID == "minibench"


# ---- red cases: never POST an invalid payload -------------------------------


def test_red_binary_100_percent_never_posts_invalid():
    client = StubClient()
    _run(_bot(client, "Probability: 100%"), _binary())
    # main.py clamps to [0.01, 0.99]: 100% is posted as 0.99, never 1.0
    assert [p["probability_yes"] for _, p in client.forecasts] == [0.99]


def test_red_non_monotone_percentiles_never_post_invalid():
    bad = NUMERIC_TEXT.replace("Percentile 40: 35", "Percentile 40: 5")
    client = StubClient()
    # parser fallback also returns garbage -> forecast must fail, not POST
    _run(_bot(client, bad, parser_text="not json"), _numeric())
    assert client.forecasts == []
    assert client.comments == []


# ---- bot_latency: emitted only when a forecast was submitted (#690) ---------


def _latency_lines(caplog):
    return [r.getMessage() for r in caplog.records if "event=bot_latency" in r.getMessage()]


def test_bot_latency_once_for_published_question(caplog):
    caplog.set_level(logging.INFO)
    client = StubClient()
    _run(_bot(client, "Rationale.\nProbability: 62%"), _binary())
    assert len(client.forecasts) == 1
    lines = _latency_lines(caplog)
    assert len(lines) == 1 and "question_id=101" in lines[0]


def test_bot_latency_not_emitted_for_failed_question(caplog):
    caplog.set_level(logging.INFO)
    bad = NUMERIC_TEXT.replace("Percentile 40: 35", "Percentile 40: 5")
    client = StubClient()
    _run(_bot(client, bad, parser_text="not json"), _numeric())
    assert client.forecasts == []
    assert _latency_lines(caplog) == []


def test_bot_latency_not_emitted_when_publish_fails(caplog):
    caplog.set_level(logging.INFO)

    class FailingClient(StubClient):
        def post_binary_question_prediction(self, question_id, prediction_in_decimal):
            raise RuntimeError("POST failed")

    _run(_bot(FailingClient(), "Rationale.\nProbability: 62%"), _binary())
    assert _latency_lines(caplog) == []
