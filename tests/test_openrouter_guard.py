"""OpenRouter preflight tests (project-backlog#675): stubbed requests.get, no network."""

import os
import subprocess
import sys
import textwrap
from unittest import mock

import openrouter_guard as g


def _resp(status=200, payload=None, bad_json=False):
    r = mock.Mock()
    r.status_code = status
    if bad_json:
        r.json.side_effect = ValueError("bad")
    else:
        r.json.return_value = payload
    return r


def _data(**kw):
    return {"data": {"limit": 20.0, "limit_remaining": 17.3, "usage": 2.7, "usage_daily": 0.4, **kw}}


def _run(resp, monkeypatch, capsys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret")
    monkeypatch.delenv("METAC_MIN_OPENROUTER_USD", raising=False)
    with mock.patch("requests.get", return_value=resp):
        out = g.preflight()
    return out, capsys.readouterr().out


def test_ok_logs_preflight(monkeypatch, capsys):
    (skip, _), out = _run(_resp(200, _data()), monkeypatch, capsys)
    assert not skip
    assert "event=bot_preflight limit=20.0 remaining=17.3 usage_daily=0.4" in out
    assert "sk-secret" not in out


def test_low_remaining_skips(monkeypatch, capsys):
    (skip, _), out = _run(_resp(200, _data(limit_remaining=0.10)), monkeypatch, capsys)
    assert skip
    assert "event=bot_skip reason=openrouter_credit_low remaining=0.1" in out


def test_null_limit_is_unlimited(monkeypatch, capsys):
    (skip, _), _ = _run(_resp(200, _data(limit=None, limit_remaining=None)), monkeypatch, capsys)
    assert not skip


def test_401_402_skip(monkeypatch, capsys):
    for code in (401, 402):
        (skip, _), out = _run(_resp(code), monkeypatch, capsys)
        assert skip and "reason=openrouter_credit_low" in out


def test_500_proceeds(monkeypatch, capsys):
    (skip, _), _ = _run(_resp(500), monkeypatch, capsys)
    assert not skip


def test_bad_json_and_network_error_proceed(monkeypatch, capsys):
    (skip, _), _ = _run(_resp(200, bad_json=True), monkeypatch, capsys)
    assert not skip
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret")
    with mock.patch("requests.get", side_effect=RuntimeError("boom sk-secret")):
        skip, _ = g.preflight()
    assert not skip
    assert "sk-secret" not in capsys.readouterr().out


def test_spend_delta(monkeypatch, capsys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    with mock.patch("requests.get", return_value=_resp(200, _data(usage=3.2))):
        g.log_spend({"status": 200, "data": {"usage": 2.7}})
    assert "event=bot_openrouter_spend delta_usd=0.5000" in capsys.readouterr().out


def test_spend_never_raises(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    with mock.patch("requests.get", side_effect=RuntimeError):
        g.log_spend({"status": 200, "data": {"usage": 1}})
    g.log_spend({})


def test_main_exits_0_before_research_on_low_credit():
    """Full __main__ in a subprocess with requests.get stubbed: exit 0, skip
    line printed, and the startup banner / bot construction never reached."""
    runner = textwrap.dedent(
        """
        import runpy, sys
        from unittest import mock
        r = mock.Mock(status_code=200)
        r.json.return_value = {"data": {"limit": 20, "limit_remaining": 0.10, "usage": 1, "usage_daily": 0}}
        sys.argv = ["main.py", "--mode", "tournament"]
        with mock.patch("requests.get", return_value=r):
            runpy.run_path("main.py", run_name="__main__")
        """
    )
    env = {**os.environ, "OPENROUTER_API_KEY": "sk-secret",
           "ASKNEWS_CLIENT_ID": "x", "ASKNEWS_SECRET": "y"}
    env.pop("METAC_MIN_OPENROUTER_USD", None)
    p = subprocess.run([sys.executable, "-c", runner], capture_output=True, text=True,
                       env=env, timeout=120,
                       cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert p.returncode == 0, p.stderr[-500:]
    assert "event=bot_skip reason=openrouter_credit_low remaining=0.1" in p.stdout
    assert "Tournament ids" not in p.stdout
    assert "sk-secret" not in p.stdout + p.stderr
