# Operating the bot

The bot forecasts unattended and spends real OpenRouter money once keys are set. This repo is **public**: Actions logs are world-readable, and PRs to it auto-merge. Secrets (`METACULUS_TOKEN`, `OPENROUTER_API_KEY`, `ASKNEWS_CLIENT_ID`, `ASKNEWS_SECRET`) are masked by GitHub and are never logged by the bot.

Workflow: `.github/workflows/run_bot_on_tournament.yaml` (cron `7,27,47 * * * *`, `workflow_dispatch` inputs `mode` and `max_questions`, concurrency group = workflow name, `cancel-in-progress: false`, 30 min timeout).

## 1. Stop it now

```bash
gh workflow disable run_bot_on_tournament.yaml -R jeremynz/metac-bot-template
# later:
gh workflow enable run_bot_on_tournament.yaml -R jeremynz/metac-bot-template
```

Disabling stops new runs; a run already in progress keeps going (cancel it in the Actions tab). To kill spend at the source, **revoke the OpenRouter key** in the OpenRouter dashboard. The preflight then sees HTTP 401/402 and exits 0 with `event=bot_skip`.

## 2. First keyed run

```bash
gh workflow run run_bot_on_tournament.yaml -R jeremynz/metac-bot-template \
  -f mode=tournament -f max_questions=2
```

Success looks like these lines in the "Run bot" step, in roughly this order (values vary):

```
event=bot_preflight limit=... remaining=... usage_daily=...
event=bot_question_cap kept=2 dropped=<n>          # only if more than 2 were eligible
event=bot_cost question_id=... url=... usd=0.xxxx researcher=... default=...
event=bot_cost_total questions=2 usd=... mean_usd=...
event=bot_run_spend usd=... limit=3.0 questions_ok=2 questions_failed=0 questions_budget_stopped=0
event=bot_openrouter_spend delta_usd=...
```

## 3. Spend check

- **Actions log**: grep the run log for `event=bot_run_spend` (run total per `forecasting_tools` cost accounting) and `event=bot_cost` (per question).
- **Step summary**: the run's summary page carries the `event=bot_run_spend` line (`main.py:1389`).
- **Ground truth**: the OpenRouter dashboard. `event=bot_openrouter_spend delta_usd` is the key-usage delta seen by the API, but the dashboard wins if they disagree.
- **Hard stop**: the in-code guards (`METAC_MAX_USD_PER_RUN`, default 3.0) are best-effort and per run; they do not bound daily spend across many runs. **Set a credit cap on the OpenRouter key/account: that is the only hard stop.** With the cap reached the preflight skips (`METAC_MIN_OPENROUTER_USD`, default 0.50).

Knobs (env; see `.env.template`): `METAC_MAX_USD_PER_RUN`, `METAC_MAX_USD_PER_QUESTION` (warn only), `METAC_MAX_QUESTIONS` (0 = unlimited; `--max-questions` wins), `METAC_MIN_OPENROUTER_USD`, `METAC_MIN_MINUTES_TO_CLOSE` (default 10).

## 4. Event glossary

Every `event=` the code emits (grepped from main; `main.py` / `openrouter_guard.py`):

| event | meaning | where |
|---|---|---|
| `bot_cost` | one question's cost, researcher, model | `main.py:100` |
| `bot_cost_total` | run total: questions, usd, mean | `main.py:111` |
| `bot_price_registered` | litellm had no price for a pinned model; one was registered (warning) | `main.py:158` |
| `bot_run_spend` | run total vs limit; ok/failed/budget-stopped counts (also in step summary) | `main.py:186` |
| `bot_budget_exhausted` | run budget hit; question not dispatched (`spent`/`limit`); exits green | `main.py:237`, `581`, `654`, `1378` |
| `bot_cost_over_question_cap` | a question exceeded the per-question cap; warning only | `main.py:292` |
| `parse_path` | which parser path a question type took | `main.py:352` |
| `bot_question_skip` | skipped, `reason=closing_soon` | `main.py:540` |
| `bot_question_cap` | `--max-questions` trimmed the batch (`kept`, `dropped`) | `main.py:556` |
| `bot_latency` | minutes to close when a question finished | `main.py:599` |
| `bot_skip` | run exits 0 without forecasting: `reason=no_llm_key` (`main.py:1254`), `reason=openrouter_credit_low` (`openrouter_guard.py:58`, `:64`). Credit skip only exits in tournament mode | see left |
| `bot_preflight` | credit check passed: `limit`, `remaining`, `usage_daily` | `openrouter_guard.py:67` |
| `bot_openrouter_spend` | key usage delta over the run, at exit | `openrouter_guard.py:81` |
| `bot_openrouter_guard_error` | preflight/spend lookup failed (`kind=`); never blocks the run | `openrouter_guard.py:33`, `:43`, `:83` |

## 5. Known limits

- **Cron is not every 20 min.** The schedule asks for 72 runs/day but GitHub fires about 5 a day (measured 2026-09-27 to 09-30). Don't assume timely coverage; use `workflow_dispatch` for a guaranteed run.
- **Pending decision on jeremynz/project-backlog#662** (adding keys / going live). Nothing here should be read as approval to enable keys.
