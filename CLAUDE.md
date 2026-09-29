# CLAUDE.md

## What this repo is

Jeremy's fork of `Metaculus/metac-bot-template`: a Metaculus AI-tournament
forecasting bot (FutureEval Fall 2026). `main.py` defines `FableForecastBot`
(a `forecasting_tools.ForecastBot` subclass) and pins cheap-by-design
OpenRouter models via its `llms=` block:
- `default`/judge: `openrouter/anthropic/claude-sonnet-5`
- `parser` + `summarizer`: `openrouter/anthropic/claude-haiku-4.5`
- `researcher`: `asknews/news-summaries` when both `ASKNEWS_CLIENT_ID` and
  `ASKNEWS_SECRET` are set, else `openrouter/moonshotai/kimi-k3`

`main_with_no_framework.py` is a separate, single-file reference
implementation with its own inline copies of the same helpers — it is
intentionally NOT wired to the pinned models above; don't assume changes to
`main.py` apply there.

Rules: the bot forecasts and comments via the Metaculus API with no human
in the loop (comments post as private notes by template default) — don't
add any interactive/manual-approval step to the forecast path.

## Secrets: env only

No secrets ever go in files. Copy `.env.template` to `.env` for local runs;
in GitHub Actions they're repo secrets, injected as env vars by the
workflow (`.github/workflows/*.yaml`, out of scope for this repo's own
Python changes — a separate ticket owns those). At minimum: `METACULUS_TOKEN`
and one LLM key (`OPENROUTER_API_KEY`, used by the pinned models above).
Optional: `ASKNEWS_CLIENT_ID`/`ASKNEWS_SECRET` (both must be set together to
enable AskNews research — see `select_researcher()` in `main.py`).

## How to run each mode

```bash
poetry install                              # local dev; installs the pinned forecasting-tools etc.
poetry run python main.py --mode test_questions   # smoke test, bot-testing-area, no publish-skip
poetry run python main.py --mode tournament       # live AIB tournament + MiniBench (default mode)
poetry run python main.py --mode metaculus_cup    # Metaculus Cup
poetry run python main_with_no_framework.py       # single-file reference impl, no --mode flag
```

`--mode tournament` (the default) forecasts on both
`MetaculusClient.CURRENT_AI_COMPETITION_ID` and `CURRENT_MINIBENCH_ID`; both
are printed at process startup (`Tournament ids: ...`) so a run's logs show
which live ids were actually resolved from the installed `forecasting-tools`
SDK version.

## Cost target (gate G0) and cost logs

Target: **<=US$0.40/question**, mean over a run. `FableForecastBot`
overrides `_run_individual_question` to log one line per question:
```
event=bot_cost question_id=<id> url=<page_url> usd=<0.0000> researcher=<name> default=<model>
```
and one run-total line at exit:
```
event=bot_cost_total questions=<n> usd=<sum> mean_usd=<x>
```
`usd` comes from `forecasting_tools`'s own `MonetaryCostManager`
(`ForecastBot._run_individual_question` already wraps each question in one
and stores the total as `report.price_estimate` — don't re-wrap with a
second `MonetaryCostManager`, it would just re-total the same litellm
callbacks). If litellm has no price entry for a pinned model and `usd`
stays `0.0000`, the same line also carries `input_tokens=<n>
output_tokens=<n>` (a litellm success-callback + `ContextVar`-scoped
counter, isolated per question the same way `MonetaryCostManager`/
`HardLimitManager` itself is — each question is a separate `asyncio.gather`
task with its own copy of the context, so concurrent questions never share
a counter).

## Spend guard (gate G0)

`METAC_MAX_USD_PER_RUN` (default 3.0) and `METAC_MAX_USD_PER_QUESTION`
(default 0.60), both in `.env.template`. In `__main__`, one
`MonetaryCostManager(hard_limit=METAC_MAX_USD_PER_RUN)` wraps a run.
`forecasting_tools`'s `forecast_questions()` dispatches all of ONE
tournament's questions via a single `asyncio.gather`, but
`FableForecastBot.run_research` checks the same run budget right after
acquiring `_concurrency_limiter` (`_max_concurrent_questions = 1`, the one
point questions are actually serialized -- the next question only enters
after the previous one's research call, its main cost, has returned and
updated `current_usage`), logging `event=bot_budget_exhausted spent=<x>
limit=<y>` and raising instead of dispatching. This stops question N+1
mid-batch, in-repo, no `forecasting_tools` fork needed -- see
`_active_run_budget_exhausted`'s docstring in `main.py` for why a check
placed only at the top of `_run_individual_question`, before any `await`,
would not be enough (all of a batch's question-tasks pass it before any one
of them finishes and updates the budget). `_run_individual_question` also
checks it, both because that's correct too (catches the budget already
exhausted before this wave of dispatch starts, e.g. between the seasonal and
MiniBench `forecast_on_tournament` calls -- `run_tournament_mode`'s
inter-tournament check is a coarse belt on top of the same thing) and to
fail fast before any non-serialized per-question setup work runs.
Per-question overrun of `METAC_MAX_USD_PER_QUESTION` is warning-only
(`event=bot_cost_over_question_cap`) -- a question already dispatched is
never skipped for going over (wave7 policy 8); that stays true here too --
this guard only ever stops dispatch of a question *before* it starts,
never a question already in flight.

Tournament mode with no LLM key configured exits 0 with one line
(`event=bot_skip reason=no_llm_key`) instead of proceeding to error on every
question -- see `has_llm_key()` in `bot_helpers.py`.

## Testing and tooling

- Repo declares deps via Poetry (`pyproject.toml` + `poetry.lock`); install
  with `poetry install` (adds a `pytest` dev dependency). No `requirements.txt`.
- Run tests with `poetry run pytest` (or `python3 -m pytest` in a sandbox
  without `poetry` installed — same test files, no network either way).
- `tests/test_config.py` is no-network: it imports `main` directly (safe —
  module-level code only imports `forecasting_tools` and defines the
  class/helpers; the `if __name__ == "__main__":` block, which needs
  `METACULUS_TOKEN`/API keys, never runs on import).
- A sandbox without `poetry` installed: don't try to regenerate
  `poetry.lock` — bump the version pin in `pyproject.toml` only and say so;
  `poetry lock --no-update` is expected to run in CI/the install step
  before `poetry install`.
