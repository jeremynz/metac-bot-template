"""OpenRouter credit preflight and ground-truth spend logging (project-backlog#675).

`GET /api/v1/key` is free. Every failure path here (network, bad JSON) is
logged and swallowed: the guard must never block a run. The key is never logged.
"""

import os

import requests

KEY_URL = "https://openrouter.ai/api/v1/key"
DEFAULT_MIN_USD = 0.50


def get_min_openrouter_usd() -> float:
    raw = os.getenv("METAC_MIN_OPENROUTER_USD", "")
    try:
        return float(raw) if raw.strip() else DEFAULT_MIN_USD
    except ValueError:
        return DEFAULT_MIN_USD


def fetch_key_info(api_key: str | None = None) -> dict:
    """Return {"status": int|None, "data": dict|None}. Never raises."""
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return {"status": None, "data": None}
    try:
        resp = requests.get(
            KEY_URL, headers={"Authorization": f"Bearer {api_key}"}, timeout=10
        )
    except Exception as e:  # network errors: log the type only
        print(f"event=bot_openrouter_guard_error kind={type(e).__name__}")
        return {"status": None, "data": None}
    status = resp.status_code
    if status != 200:
        return {"status": status, "data": None}
    try:
        data = resp.json()["data"]
        if not isinstance(data, dict):
            raise ValueError("data not an object")
    except Exception as e:
        print(f"event=bot_openrouter_guard_error kind={type(e).__name__}")
        return {"status": status, "data": None}
    return {"status": status, "data": data}


def _num(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def preflight(min_usd: float | None = None) -> tuple[bool, dict]:
    """Return (should_skip, info). Prints the skip/preflight event lines."""
    min_usd = get_min_openrouter_usd() if min_usd is None else min_usd
    info = fetch_key_info()
    status, data = info["status"], info["data"]
    if status in (401, 402):
        print(f"event=bot_skip reason=openrouter_credit_low remaining=http_{status}")
        return True, info
    if data is None:
        return False, info
    remaining = _num(data.get("limit_remaining"))  # None = unlimited
    if remaining is not None and remaining < min_usd:
        print(f"event=bot_skip reason=openrouter_credit_low remaining={remaining}")
        return True, info
    print(
        f"event=bot_preflight limit={data.get('limit')} "
        f"remaining={data.get('limit_remaining')} usage_daily={data.get('usage_daily')}"
    )
    return False, info


def log_spend(before: dict) -> None:
    """Fetch key info again and log usage delta. Never raises."""
    try:
        usage_before = _num((before.get("data") or {}).get("usage"))
        after = fetch_key_info()
        usage_after = _num((after["data"] or {}).get("usage"))
        if usage_before is None or usage_after is None:
            return
        print(f"event=bot_openrouter_spend delta_usd={usage_after - usage_before:.4f}")
    except Exception as e:
        print(f"event=bot_openrouter_guard_error kind={type(e).__name__}")
