"""No-network tests for parsing.py's deterministic parse-first helpers
(project-backlog#613) plus one integration test proving main.py's
_binary_prompt_to_forecast falls back to structure_output when the
deterministic parse returns None.

Uses asyncio.run, not pytest-asyncio, matching tests/test_config.py's
existing style (that plugin isn't a declared dev dependency here).
"""

import asyncio
import logging
import math
from datetime import datetime
from unittest import mock

import pytest
from forecasting_tools import BinaryPrediction, GeneralLlm

from main import FableForecastBot
from parsing import (
    parse_binary,
    parse_date_percentiles,
    parse_multiple_choice,
    parse_numeric_percentiles,
)

# ---------------------------------------------------------------- binary ---

BINARY_VALID_FIXTURES = [
    ("clean", "Some reasoning here.\nProbability: 42%", 0.42),
    (
        "multiple_lines_last_wins",
        "First guess Probability: 10%\nOn reflection, Probability: 73%",
        0.73,
    ),
    ("decimal_percentage", "After analysis:\nProbability: 12.5%", 0.125),
    # BinaryPrediction's own validator clamps to [0.001, 0.999] (AI models
    # sometimes emit exact 0/1) -- 0% and 100% land there, not at 0.0/1.0.
    ("boundary_zero", "Probability: 0%", 0.001),
    ("boundary_hundred", "Probability: 100%", 0.999),
]

BINARY_INVALID_FIXTURES = [
    ("junk_text", "I have no idea what will happen here."),
    ("out_of_range_above_100", "Probability: 150%"),
    ("negative_not_matched", "Probability: -5%"),
]


@pytest.mark.parametrize("name, text, expected", BINARY_VALID_FIXTURES)
def test_parse_binary_valid(name, text, expected):
    result = parse_binary(text)
    assert result is not None, name
    assert isinstance(result, BinaryPrediction)
    assert math.isclose(result.prediction_in_decimal, expected), name


@pytest.mark.parametrize("name, text", BINARY_INVALID_FIXTURES)
def test_parse_binary_invalid(name, text):
    assert parse_binary(text) is None, name


# --------------------------------------------------------- multiple choice --

MC_VALID_FIXTURES = [
    ("clean", ["Yes", "No"], "Yes: 60%\nNo: 40%", {"Yes": 0.6, "No": 0.4}),
    ("reordered", ["Yes", "No"], "No: 40%\nYes: 60%", {"Yes": 0.6, "No": 0.4}),
    ("mis_cased", ["Yes", "No"], "yes: 70%\nNO: 30%", {"Yes": 0.7, "No": 0.3}),
    (
        "plain_fractions_not_percentages",
        ["Yes", "No"],
        "Yes: 0.6\nNo: 0.4",
        {"Yes": 0.6, "No": 0.4},
    ),
    (
        "normalizes_when_not_summing_to_one",
        ["Red", "Green", "Blue"],
        "Red: 3\nGreen: 1\nBlue: 1",
        {"Red": 0.6, "Green": 0.2, "Blue": 0.2},
    ),
]

MC_INVALID_FIXTURES = [
    ("missing_option", ["Yes", "No"], "Yes: 100%"),
    ("junk_text", ["Yes", "No"], "I cannot decide on this one."),
]


@pytest.mark.parametrize("name, options, text, expected", MC_VALID_FIXTURES)
def test_parse_multiple_choice_valid(name, options, text, expected):
    result = parse_multiple_choice(text, options)
    assert result is not None, name
    got = result.to_dict()
    for option, probability in expected.items():
        assert math.isclose(got[option], probability, abs_tol=0.05), name


@pytest.mark.parametrize("name, options, text", MC_INVALID_FIXTURES)
def test_parse_multiple_choice_invalid(name, options, text):
    assert parse_multiple_choice(text, options) is None, name


def test_parse_multiple_choice_keeps_explicit_zero_percent_option():
    # An explicit "0%" for an option must be kept as an entry, not
    # dropped/treated the same as a missing option.
    result = parse_multiple_choice("Yes: 0%\nNo: 100%", ["Yes", "No"])
    assert result is not None
    names = {option.option_name for option in result.predicted_options}
    assert names == {"Yes", "No"}


# --------------------------------------------------------------- numeric ---

_NUMERIC_TEMPLATE = (
    "Percentile 10: {p10}\n"
    "Percentile 20: {p20}\n"
    "Percentile 40: {p40}\n"
    "Percentile 60: {p60}\n"
    "Percentile 80: {p80}\n"
    "Percentile 90: {p90}\n"
)

NUMERIC_VALID_FIXTURES = [
    (
        "clean",
        _NUMERIC_TEMPLATE.format(p10=100, p20=200, p40=300, p60=400, p80=500, p90=600),
        [100, 200, 300, 400, 500, 600],
    ),
    (
        "commas",
        _NUMERIC_TEMPLATE.format(
            p10="1,200", p20="2,400", p40="3,600", p60="4,800", p80="6,000", p90="7,200"
        ),
        [1200, 2400, 3600, 4800, 6000, 7200],
    ),
    (
        "k_suffix",
        _NUMERIC_TEMPLATE.format(p10="1.2k", p20="2.4k", p40="3.6k", p60="4.8k", p80="6k", p90="7.2k"),
        [1200, 2400, 3600, 4800, 6000, 7200],
    ),
    (
        "M_suffix",
        _NUMERIC_TEMPLATE.format(p10="1M", p20="2M", p40="3M", p60="4M", p80="5M", p90="6M"),
        [1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000, 6_000_000],
    ),
    (
        "dollar_prefix",
        _NUMERIC_TEMPLATE.format(p10="$100", p20="$200", p40="$300", p60="$400", p80="$500", p90="$600"),
        [100, 200, 300, 400, 500, 600],
    ),
    (
        "trailing_units",
        _NUMERIC_TEMPLATE.format(
            p10="100 USD", p20="200 USD", p40="300 USD", p60="400 USD", p80="500 USD", p90="600 USD"
        ),
        [100, 200, 300, 400, 500, 600],
    ),
    (
        "negative_values",
        _NUMERIC_TEMPLATE.format(p10=-50, p20=-40, p40=-30, p60=-20, p80=-10, p90=0),
        [-50, -40, -30, -20, -10, 0],
    ),
    (
        "trailing_unit_starting_with_suffix_letter",
        # "minutes"/"kg" start with k/m/b -- must NOT be misread as a
        # 1e3/1e6 multiplier (project-backlog#613 review round 1).
        _NUMERIC_TEMPLATE.format(
            p10="10 minutes", p20="20 minutes", p40="30 minutes",
            p60="40 kg", p80="50 kg", p90="60 kg",
        ),
        [10, 20, 30, 40, 50, 60],
    ),
    (
        "magnitude_words",
        # "N million"/"N billion" -- the letter-suffix lookahead
        # (?![A-Za-z]) rejects "m" of "million" as a bare multiplier,
        # so the magnitude word itself must be matched and scaled
        # correctly instead of being dropped as a trailing unit
        # (project-backlog#613 review round 2).
        _NUMERIC_TEMPLATE.format(
            p10="1 thousand", p20="2 million", p40="1.5 billion",
            p60="2 trillion", p80="3 trillion", p90="4 trillion",
        ),
        [1_000, 2_000_000, 1_500_000_000, 2_000_000_000_000, 3_000_000_000_000, 4_000_000_000_000],
    ),
]

NUMERIC_INVALID_FIXTURES = [
    (
        "missing_percentile",
        (
            "Percentile 10: 100\nPercentile 20: 200\nPercentile 40: 300\n"
            "Percentile 80: 500\nPercentile 90: 600\n"
        ),
    ),
    (
        "non_monotonic",
        _NUMERIC_TEMPLATE.format(p10=100, p20=50, p40=300, p60=400, p80=500, p90=600),
    ),
    ("junk_text", "The forecast is too uncertain to give any numbers."),
    (
        "scientific_notation",
        # "1.2e6" must not silently parse as 1.2 (dropping "e6" as a
        # tolerated trailing unit) -- must fail so the LLM fallback,
        # which is instructed to convert scientific notation, runs.
        _NUMERIC_TEMPLATE.format(
            p10="1.0e6", p20="1.2e6", p40="1.4e6", p60="1.6e6", p80="1.8e6", p90="2.0e6"
        ),
    ),
]


@pytest.mark.parametrize("name, text, expected_values", NUMERIC_VALID_FIXTURES)
def test_parse_numeric_percentiles_valid(name, text, expected_values):
    result = parse_numeric_percentiles(text)
    assert result is not None, name
    assert [p.percentile for p in result] == [0.10, 0.20, 0.40, 0.60, 0.80, 0.90]
    assert [p.value for p in result] == pytest.approx(expected_values), name


@pytest.mark.parametrize("name, text", NUMERIC_INVALID_FIXTURES)
def test_parse_numeric_percentiles_invalid(name, text):
    assert parse_numeric_percentiles(text) is None, name


# ------------------------------------------------------------------ date ---

_DATE_TEMPLATE = (
    "Percentile 10: {p10}\n"
    "Percentile 20: {p20}\n"
    "Percentile 40: {p40}\n"
    "Percentile 60: {p60}\n"
    "Percentile 80: {p80}\n"
    "Percentile 90: {p90}\n"
)

DATE_VALID_FIXTURES = [
    (
        "clean",
        _DATE_TEMPLATE.format(
            p10="2027-01-01",
            p20="2027-02-01",
            p40="2027-04-01",
            p60="2027-06-01",
            p80="2027-08-01",
            p90="2027-10-01",
        ),
        [
            datetime(2027, 1, 1),
            datetime(2027, 2, 1),
            datetime(2027, 4, 1),
            datetime(2027, 6, 1),
            datetime(2027, 8, 1),
            datetime(2027, 10, 1),
        ],
    ),
    (
        "with_time_and_z_suffix",
        _DATE_TEMPLATE.format(
            p10="2027-01-01T00:00:00Z",
            p20="2027-02-01T00:00:00Z",
            p40="2027-04-01T00:00:00Z",
            p60="2027-06-01T00:00:00Z",
            p80="2027-08-01T00:00:00Z",
            p90="2027-10-01T00:00:00Z",
        ),
        [
            datetime(2027, 1, 1),
            datetime(2027, 2, 1),
            datetime(2027, 4, 1),
            datetime(2027, 6, 1),
            datetime(2027, 8, 1),
            datetime(2027, 10, 1),
        ],
    ),
]

DATE_INVALID_FIXTURES = [
    (
        "missing_percentile",
        (
            "Percentile 10: 2027-01-01\nPercentile 20: 2027-02-01\n"
            "Percentile 40: 2027-04-01\nPercentile 90: 2027-10-01\n"
        ),
    ),
    (
        "non_monotonic",
        _DATE_TEMPLATE.format(
            p10="2027-02-01",
            p20="2027-01-01",
            p40="2027-04-01",
            p60="2027-06-01",
            p80="2027-08-01",
            p90="2027-10-01",
        ),
    ),
    ("junk_text", "I am not confident enough to give exact dates."),
]


@pytest.mark.parametrize("name, text, expected_values", DATE_VALID_FIXTURES)
def test_parse_date_percentiles_valid(name, text, expected_values):
    result = parse_date_percentiles(text)
    assert result is not None, name
    assert [p.percentile for p in result] == [0.10, 0.20, 0.40, 0.60, 0.80, 0.90]
    assert [p.value for p in result] == expected_values, name


@pytest.mark.parametrize("name, text", DATE_INVALID_FIXTURES)
def test_parse_date_percentiles_invalid(name, text):
    assert parse_date_percentiles(text) is None, name


# ----------------------------------------------- fallback-to-LLM integration --


class _StubBinaryQuestion:
    page_url = "https://www.metaculus.com/questions/99/"
    id_of_question = 99
    id_of_post = None


def test_binary_prompt_to_forecast_falls_back_to_structure_output_when_parse_fails():
    """Deterministic parse fails on unparseable reasoning -> the existing
    structure_output LLM-parser path is still called, with its arguments
    unchanged, and the parse-path log line records the fallback."""
    bot = FableForecastBot(
        llms={
            "default": GeneralLlm(model="openrouter/anthropic/claude-sonnet-5"),
            "summarizer": "openrouter/anthropic/claude-haiku-4.5",
            "researcher": "asknews/news-summaries",
            "parser": "openrouter/anthropic/claude-haiku-4.5",
        },
    )

    junk_reasoning = "I really cannot commit to a number here."
    assert parse_binary(junk_reasoning) is None  # sanity: this is why fallback fires

    fallback_prediction = BinaryPrediction(prediction_in_decimal=0.77)
    fake_structure_output = mock.AsyncMock(return_value=fallback_prediction)

    async def fake_invoke(self, prompt):
        return junk_reasoning

    caplog_records: list[str] = []

    class _ListHandler(logging.Handler):
        def emit(self, record):
            caplog_records.append(record.getMessage())

    handler = _ListHandler()
    logging.getLogger("main").addHandler(handler)
    logging.getLogger("main").setLevel(logging.INFO)
    try:
        with mock.patch.object(GeneralLlm, "invoke", fake_invoke), mock.patch(
            "main.structure_output", fake_structure_output
        ):
            result = asyncio.run(
                bot._binary_prompt_to_forecast(_StubBinaryQuestion(), "irrelevant prompt")
            )
    finally:
        logging.getLogger("main").removeHandler(handler)

    fake_structure_output.assert_awaited_once()
    assert result.prediction_value == 0.77
    parse_path_lines = [line for line in caplog_records if "event=parse_path" in line]
    assert parse_path_lines == [
        "event=parse_path question_id=99 type=binary path=llm_fallback"
    ]
