"""No-network tests for tournament id resolution + expiry check (#773)."""
from datetime import datetime, timezone

import pytest

import main
from main import check_tournament_open, resolve_tournament_id

NOW = datetime(2026, 10, 5, tzinfo=timezone.utc)


def test_env_override_wins():
    assert resolve_tournament_id({"METACULUS_TOURNAMENT_ID": "99999"}) == 99999
    assert resolve_tournament_id({"METACULUS_TOURNAMENT_ID": "minibench"}) == "minibench"


def test_default_is_documented_fall_2026_id():
    assert resolve_tournament_id({}) == 33121
    assert main.MetaculusClient.FE_FALL_2026_ID == 33121


def test_empty_default_falls_back_with_warning(monkeypatch, caplog):
    monkeypatch.setattr(main, "DEFAULT_TOURNAMENT_ID", "")
    assert resolve_tournament_id({}) == main.MetaculusClient.CURRENT_AI_COMPETITION_ID
    assert "tournament_id_fallback" in caplog.text


def test_expired_tournament_aborts():
    with pytest.raises(SystemExit) as e:
        check_tournament_open(
            33022, lambda t: {"forecasting_end_date": "2026-09-01T00:00:00Z"}, now=NOW
        )
    assert "tournament_expired" in str(e.value)


def test_open_tournament_passes():
    check_tournament_open(
        33121, lambda t: {"forecasting_end_date": "2027-01-06T00:00:00Z"}, now=NOW
    )


def test_fetch_failure_does_not_block(caplog):
    def boom(t):
        raise RuntimeError("403")

    check_tournament_open(33121, boom, now=NOW)
    assert "expiry_check_skipped" in caplog.text
