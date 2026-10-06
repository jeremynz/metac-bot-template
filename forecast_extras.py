"""Flag-gated, default-off extras (project-backlog#782): outside-view step and
Platt calibration. Pure helpers, no network. With the flags unset nothing here
changes any production output."""

import json
import math
import os

EPS = 1e-6


def outside_view_enabled(environ=None) -> bool:
    env = os.environ if environ is None else environ
    return env.get("METAC_OUTSIDE_VIEW", "").strip() == "1"


def calibration_path(environ=None) -> str | None:
    env = os.environ if environ is None else environ
    return env.get("METAC_CALIBRATION", "").strip() or None


def outside_view_prompt(question_text: str, resolution_criteria: str, fine_print: str) -> str:
    return (
        "You are an assistant to a superforecaster. Do NOT research current news.\n"
        "For the question below, name the most appropriate reference class and "
        "give the historical base rate (a probability or typical value) for that "
        "class. Be brief: 2-4 sentences, ending with 'Base rate: <value>'.\n\n"
        f"Question:\n{question_text}\n\nResolution criteria:\n{resolution_criteria}\n\n{fine_print}"
    )


def append_outside_view(research: str, outside_view: str) -> str:
    return f"{research}\n\nOutside view (reference class and base rate):\n{outside_view.strip()}"


def logit(p: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    e = math.exp(x)
    return e / (1 + e)


def load_calibration(path: str | None) -> tuple[float, float]:
    """(a, b) from a JSON file {"a":..,"b":..}; no/missing path -> identity (1, 0)."""
    if not path:
        return 1.0, 0.0
    with open(path) as f:
        d = json.load(f)
    return float(d["a"]), float(d["b"])


def calibrate(p: float, path: str | None) -> float:
    """p' = sigmoid(a*logit(p)+b). Identity when no path is given."""
    if not path:
        return p
    a, b = load_calibration(path)
    return sigmoid(a * logit(p) + b)


def fit_platt(forecasts, outcomes, iters: int = 50) -> tuple[float, float]:
    """Max-likelihood logistic fit of outcome ~ sigmoid(a*logit(p)+b), Newton's
    method with a tiny ridge for stability. Pure stdlib."""
    xs = [logit(p) for p in forecasts]
    ys = [float(o) for o in outcomes]
    a, b = 1.0, 0.0
    for _ in range(iters):
        ga = gb = haa = hab = hbb = 0.0
        for x, y in zip(xs, ys):
            q = sigmoid(a * x + b)
            r = q - y
            w = q * (1 - q) + 1e-9
            ga += r * x
            gb += r
            haa += w * x * x
            hab += w * x
            hbb += w
        haa += 1e-9
        hbb += 1e-9
        det = haa * hbb - hab * hab
        if abs(det) < 1e-18:
            break
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        a -= da
        b -= db
        if abs(da) + abs(db) < 1e-10:
            break
    return a, b
