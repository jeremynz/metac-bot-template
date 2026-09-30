import argparse
import asyncio
import atexit
import logging
import os
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Literal

import dotenv
import litellm
import openrouter_guard
from litellm.integrations.custom_logger import CustomLogger as LitellmCustomLogger

# HardLimitExceededError/get_active_cost_managers() aren't re-exported from
# forecasting_tools' top-level package (only MonetaryCostManager is) -- same
# submodule path tests/test_budget.py already imports HardLimitManager from.
from forecasting_tools.ai_models.resource_managers.hard_limit_manager import (
    HardLimitExceededError,
    HardLimitManager,
)

# Runtime helpers (env validation, banners, dependency-warning suppression).
from bot_helpers import (
    check_environment,
    has_llm_key,
    print_run_summary_banner,
    print_startup_banner,
    silence_noisy_dependencies,
)

silence_noisy_dependencies()

from parsing import (
    numeric_percentiles_within_bounds,
    parse_binary,
    parse_date_percentiles,
    parse_multiple_choice,
    parse_numeric_percentiles,
)

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MonetaryCostManager,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    DateQuestion,
    DatePercentile,
    Percentile,
    ConditionalQuestion,
    ConditionalPrediction,
    PredictionTypes,
    PredictionAffirmed,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)

dotenv.load_dotenv()
logger = logging.getLogger(__name__)


def select_researcher(asknews_client_id: str | None, asknews_secret: str | None) -> str:
    """
    Pick the researcher: AskNews's news-summaries endpoint when both AskNews
    env vars are configured, else the pinned OpenRouter fallback model.
    Pure/no-network so it's directly unit-testable (tests/test_config.py).
    """
    if asknews_client_id and asknews_secret:
        return "asknews/news-summaries"
    return "openrouter/moonshotai/kimi-k3"


def format_bot_cost_line(
    question_id: int | str | None,
    url: str | None,
    usd: float,
    researcher: str,
    default_model: str,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> str:
    """
    One grep-able line per question: `event=bot_cost ...`. If litellm/
    MonetaryCostManager couldn't price the call (usd rounds to 0), the
    token-count fallback fields are appended so cost is still visible in
    the logs (gate G0 stop condition).
    """
    line = (
        f"event=bot_cost question_id={question_id} url={url} "
        f"usd={usd:.4f} researcher={researcher} default={default_model}"
    )
    if round(usd, 4) <= 0.0 and input_tokens is not None and output_tokens is not None:
        line += f" input_tokens={input_tokens} output_tokens={output_tokens}"
    return line


def format_bot_cost_total_line(questions: int, total_usd: float) -> str:
    """Run-total companion line to format_bot_cost_line, logged once at exit."""
    mean_usd = total_usd / questions if questions else 0.0
    return f"event=bot_cost_total questions={questions} usd={total_usd:.4f} mean_usd={mean_usd:.4f}"


# OpenRouter list prices per token (project-backlog#676). litellm 1.80.10
# doesn't map haiku-4.5 under the openrouter/ prefix, so its cost is 0 unless
# registered; ensure_model_pricing() registers only the slugs that price at 0.
PINNED_MODEL_PRICES: dict[str, tuple[float, float]] = {
    "openrouter/anthropic/claude-sonnet-5": (2e-6, 1e-5),
    "openrouter/anthropic/claude-haiku-4.5": (1e-6, 5e-6),
    "openrouter/moonshotai/kimi-k3": (3e-6, 1.5e-5),
}


def mock_cost_usd(model: str) -> float:
    """USD MonetaryCostManager records for one mocked call to `model`."""
    with MonetaryCostManager() as cm:
        asyncio.run(
            GeneralLlm(model=model, mock_response="hello " * 2000).invoke(
                "word " * 3000
            )
        )
    return cm.current_usage


def ensure_model_pricing() -> list[str]:
    """Register list prices for pinned models litellm prices at 0.

    Returns the slugs registered.
    """
    registered = []
    for model, (inp, out) in PINNED_MODEL_PRICES.items():
        if mock_cost_usd(model) > 0:
            continue
        entry = {
            "input_cost_per_token": inp,
            "output_cost_per_token": out,
            "litellm_provider": "openrouter",
            "mode": "chat",
        }
        prices = {model: entry}
        # litellm 1.80.10 resolves openrouter/anthropic/* responses to the
        # anthropic cost calculator with the bare model name, so the price must
        # also sit under that key with provider "anthropic".
        if model.startswith("openrouter/anthropic/"):
            bare = model.removeprefix("openrouter/anthropic/")
            prices[bare] = {**entry, "litellm_provider": "anthropic"}
        litellm.register_model(prices)
        logger.warning(f"event=bot_price_registered model={model}")
        registered.append(model)
    return registered


def _is_budget_stop(report: object) -> bool:
    err = report if isinstance(report, BaseException) else None
    while err is not None:
        if isinstance(err, HardLimitExceededError):
            return True
        err = err.__cause__ or err.__context__
    return False


def split_budget_stops(reports: list) -> tuple[list, list]:
    """(reports without budget stops, budget-stop exceptions)."""
    kept = [r for r in reports if not _is_budget_stop(r)]
    return kept, [r for r in reports if _is_budget_stop(r)]


def format_bot_run_spend_line(
    usd: float,
    limit: float,
    questions_ok: int,
    questions_failed: int,
    questions_budget_stopped: int = 0,
) -> str:
    return (
        f"event=bot_run_spend usd={usd:.4f} limit={limit} "
        f"questions_ok={questions_ok} questions_failed={questions_failed} "
        f"questions_budget_stopped={questions_budget_stopped}"
    )


def get_max_usd_per_run() -> float:
    """
    METAC_MAX_USD_PER_RUN env var (default 3.0) -- gate G0 hard cap on total
    spend for one process run (project-backlog#618). Read live (not cached
    at import time) so tests can monkeypatch the env var directly.
    """
    return float(os.getenv("METAC_MAX_USD_PER_RUN", "3.0"))


def get_max_questions() -> int:
    """METAC_MAX_QUESTIONS env var (default 0 = unlimited): max questions
    forecast per forecast_questions() call (project-backlog#673)."""
    return max(0, int(os.getenv("METAC_MAX_QUESTIONS", "0") or "0"))


def get_max_usd_per_question() -> float:
    """
    METAC_MAX_USD_PER_QUESTION env var (default 0.60) -- gate G0 per-question
    warning threshold (project-backlog#618). Warning only: per wave7 policy
    8, a question is never skipped for score reasons once it's dispatched.
    """
    return float(os.getenv("METAC_MAX_USD_PER_QUESTION", "0.60"))


def format_bot_budget_exhausted_line(spent: float, limit: float) -> str:
    """Logged when the run-level MonetaryCostManager hard limit is hit."""
    return f"event=bot_budget_exhausted spent={spent:.4f} limit={limit:.2f}"


def _active_run_budget_exhausted() -> tuple[float, float] | None:
    """
    Per-question spend guard (project-backlog#618 round 1). Checks
    HardLimitManager's own ContextVar-based active-manager stack
    (`get_active_cost_managers()` -- the same mechanism its litellm
    pre-API-call callback uses in `raise_error_if_limit_would_be_reached()`,
    which is how the installed forecasting_tools==0.3.1 already raises
    `HardLimitExceededError` before any LLM call once a hard_limit is over)
    for any manager whose `hard_limit` is set and already exhausted. Returns
    `(spent, limit)` for the first one found, else None. No explicit
    reference to __main__'s `run_cost_manager` needs threading through the
    bot instance -- the ContextVar is visible from any coroutine started
    under the same `with MonetaryCostManager(...)` block, including nested
    `asyncio.gather` tasks.

    This must be called from inside `FableForecastBot.run_research`'s
    `_concurrency_limiter` (`_max_concurrent_questions = 1`), not only at
    the top of `_run_individual_question` before any `await`. Reasoning
    (verified against real asyncio scheduling, see
    tests/test_budget.py's test_run_research_stops_question_after_run_budget_exhausted_mid_batch):
    `forecast_questions()` dispatches all of a tournament's question-tasks
    via one `asyncio.gather` up front. Each task runs its own synchronous
    prefix -- everything before its first real suspension -- before control
    returns to the event loop. A check placed at the very top of
    `_run_individual_question`, before any `await`, is itself part of that
    synchronous prefix, so all N tasks pass it while `current_usage` is
    still whatever it was when the batch started -- none of them has
    finished a real LLM call yet to update it. `_concurrency_limiter` is
    the one place questions are actually serialized one at a time
    (`async with` only lets the next task in after the previous one's
    `run_research` call -- including its cost-incurring LLM/search call --
    has returned), so a check made right after acquiring it is the one that
    reads an up-to-date budget and can actually stop question N+1.
    `_run_individual_question` still calls this too, both because it is
    also correct there (it catches the budget already being exhausted
    before this wave of dispatch even starts -- e.g. between the seasonal
    and MiniBench forecast_on_tournament calls) and to fail fast before any
    of the cheaper non-serialized setup work (notepad init, etc.) runs.
    """
    for manager in HardLimitManager.get_active_cost_managers():
        if manager.hard_limit and manager.amount_left <= 0:
            return manager.current_usage, manager.hard_limit
    return None


def format_bot_cost_over_question_cap_line(
    question_id: int | str | None, usd: float, cap: float
) -> str:
    """Warning-only companion to format_bot_cost_line: one question's cost
    exceeded METAC_MAX_USD_PER_QUESTION. Never causes the question to be
    skipped (wave7 policy 8) -- logged after the fact."""
    return (
        f"event=bot_cost_over_question_cap question_id={question_id} "
        f"usd={usd:.4f} cap={cap:.2f}"
    )


def run_tournament_mode(
    template_bot: "FableForecastBot",
    client: MetaculusClient,
    run_cost_manager: MonetaryCostManager,
) -> list:
    """
    Tournament-mode dispatch: two sequential forecast_on_tournament calls
    (seasonal AI competition, then MiniBench).

    Gate G0 spend guard (project-backlog#618): if the first call already
    exhausted run_cost_manager's hard_limit, the second is skipped entirely
    (logged as event=bot_budget_exhausted) rather than dispatched -- this is
    a coarse, whole-tournament-early belt on top of the real per-question
    stop, which lives in FableForecastBot.run_research (round 1): each
    question's research call only starts after acquiring
    `_concurrency_limiter` (`_max_concurrent_questions = 1`), the one point
    questions are genuinely serialized, so a budget check made there does
    stop question N+1 mid-batch, within a single forecast_on_tournament
    call -- see run_research and _active_run_budget_exhausted's docstrings.

    Extracted from __main__ (not just inline) so it's callable directly from
    tests/test_budget.py without spawning a subprocess or exercising the
    argparse/check_environment scaffolding above it.
    """
    seasonal_tournament_reports = asyncio.run(
        template_bot.forecast_on_tournament(
            client.CURRENT_AI_COMPETITION_ID, return_exceptions=True
        )
    )
    if run_cost_manager.hard_limit and run_cost_manager.amount_left <= 0:
        logger.warning(
            format_bot_budget_exhausted_line(
                spent=run_cost_manager.current_usage,
                limit=run_cost_manager.hard_limit,
            )
        )
        minibench_reports = []
    else:
        minibench_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
        )
    return seasonal_tournament_reports + minibench_reports


def format_parse_path_line(
    question_id: int | str | None,
    question_type: Literal["binary", "mc", "numeric", "date"],
    path: Literal["deterministic", "llm_fallback", "failed"],
) -> str:
    """One grep-able line per question (project-backlog#613): which path
    produced the structured forecast -- the deterministic parser in
    parsing.py, the structure_output LLM fallback, or (rare) the fallback
    itself raising."""
    return f"event=parse_path question_id={question_id} type={question_type} path={path}"


class _TokenUsageTracker:
    """
    Per-question fallback token counter, populated only to cover the case
    where MonetaryCostManager can't price a model (usd stays 0 -- e.g. a
    model litellm has no cost entry for). Scoped per-asyncio-task via a
    ContextVar, the same isolation trick forecasting_tools' own
    HardLimitManager/MonetaryCostManager uses: forecast_questions() runs
    every question concurrently via asyncio.gather, and each gathered
    coroutine gets its own copy of the context, so concurrent questions
    never share (or race on) a counter -- no locking or sleep needed.
    """

    _active: ContextVar[list["_TokenUsageTracker"]] = ContextVar(
        "_active_token_trackers", default=[]
    )

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0

    def __enter__(self) -> "_TokenUsageTracker":
        trackers = self._active.get().copy()
        trackers.append(self)
        self._active.set(trackers)
        _TokenUsageCallback.initialize()
        return self

    def __exit__(self, exc_type, exc_value, tb) -> None:
        trackers = self._active.get().copy()
        trackers.remove(self)
        self._active.set(trackers)

    @classmethod
    def _record(cls, prompt_tokens: int, completion_tokens: int) -> None:
        for tracker in cls._active.get():
            tracker.input_tokens += prompt_tokens
            tracker.output_tokens += completion_tokens


class _TokenUsageCallback(LitellmCustomLogger):
    """litellm success-callback companion to _TokenUsageTracker."""

    _initialized = False

    @staticmethod
    def initialize() -> None:
        if _TokenUsageCallback._initialized:
            return
        already_registered = any(
            isinstance(handler, _TokenUsageCallback) for handler in litellm.callbacks
        )
        if not already_registered:
            litellm.callbacks.append(_TokenUsageCallback())
        _TokenUsageCallback._initialized = True

    def log_success_event(self, kwargs, response_obj, start_time, end_time):  # NOSONAR
        self._track(response_obj)

    async def async_log_success_event(
        self, kwargs, response_obj, start_time, end_time
    ):  # NOSONAR
        self._track(response_obj)

    @staticmethod
    def _track(response_obj) -> None:
        usage = getattr(response_obj, "usage", None)
        if usage is None:
            return
        _TokenUsageTracker._record(
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
        )


class FableForecastBot(ForecastBot):
    """
    This is the template bot for Summer 2026 Metaculus AI Tournament.
    This is a copy of what is used by Metaculus to run the Metac Bots in our benchmark, provided as a template for new bot makers.
    This template is given as-is, and is use-at-your-own-risk.
    We have covered most test cases in forecasting-tools it may be worth double checking key components locally.
    So far our track record has been 1 mentionable bug per season (affecting forecasts for 1-2% of total questions)

    Main changes since Fall:
    - Additional prompting has been added to numeric questions to emphasize putting pecentile values in the correct order.
    - Support for conditional and date questions has been added
    - Note: Summer AIB will not use date/conditional questions, so these are only for forecasting on the main site as you wish.

    The main entry point of this bot is `bot.forecast_on_tournament(tournament_id)` in the parent class.
    See the script at the bottom of the file for more details on how to run the bot.
    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Alternatively, you can use the MetaculusClient to make a custom filter of questions to forecast on
    and forecast them with `bot.forecast_questions(questions)`

    Only the research and forecast functions need to be implemented in ForecastBot subclasses,
    though you may want to override other ForecastBot functions.
    In this example, you can change the prompts to be whatever you want since,
    structure_output uses an LLM to intelligently reformat the output into the needed structure.

    By default (i.e. 'tournament' mode), when you run this script, it will forecast on any open questions in the
    primary bot tournament and MiniBench. If you want to forecast on only one or the other, you can remove one
    of them from the 'tournament' mode code at the bottom of the file.

    You can experiment with what models work best with your bot by using the `llms` parameter when initializing the bot.
    You can initialize the bot with any number of models. For example,
    ```python
    my_bot = MyBot(
        ...
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": GeneralLlm(
                model="openrouter/openai/gpt-4o", # "anthropic/claude-sonnet-4-20250514", etc (see docs for litellm)
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": "openai/gpt-4o-mini",
            "researcher": "asknews/news-summaries",
            "parser": "openai/gpt-4o-mini",
        },
    )
    ```

    Then you can access the model in custom functions like this:
    ```python
    research_strategy = self.get_llm("researcher", "model_name"
    if research_strategy == "asknews/news-summaries":
        ...
    # OR
    summarizer = await self.get_llm("summarizer", "llm").invoke(prompt)
    # OR
    reasoning = await self.get_llm("default", "llm").invoke(prompt)
    ```

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```python
    from forecasting_tools import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```
    Additionally OpenRouter has large rate limits immediately on account creation
    """

    _max_concurrent_questions = (
        1  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 2

    def __init__(self, *args, max_questions: int | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._question_costs_usd: list[float] = []
        # 0 = unlimited (project-backlog#673).
        self._max_questions = (
            max_questions if max_questions is not None else get_max_questions()
        )

    async def forecast_questions(self, questions, return_exceptions: bool = False):
        """Keep only the first `max_questions` questions (input order) so a
        first keyed smoke run is cheap. 0 = unlimited (default)."""
        questions = list(questions)
        if self._max_questions > 0 and len(questions) > self._max_questions:
            kept = questions[: self._max_questions]
            logger.info(
                f"event=bot_question_cap kept={len(kept)} "
                f"dropped={len(questions) - len(kept)}"
            )
            questions = kept
        return await super().forecast_questions(
            questions, return_exceptions=return_exceptions
        )

    ##################################### COST LOGGING #####################################

    async def _run_individual_question(self, question: MetaculusQuestion):
        # Gate G0 spend guard (project-backlog#618 round 1): fail fast,
        # before any of this question's setup work runs, if a prior wave of
        # dispatch already exhausted the run budget (e.g. the seasonal
        # tournament's own questions, before MiniBench's forecast_on_tournament
        # is even called). This alone does NOT stop question N+1 WITHIN one
        # forecast_on_tournament's batch -- see _active_run_budget_exhausted's
        # docstring and the matching check in run_research below, which is
        # the one that does.
        exhausted = _active_run_budget_exhausted()
        if exhausted is not None:
            spent, limit = exhausted
            logger.warning(format_bot_budget_exhausted_line(spent=spent, limit=limit))
            question_id = question.id_of_question or question.id_of_post
            raise HardLimitExceededError(
                f"event=bot_budget_exhausted question_id={question_id} "
                f"spent={spent:.4f} limit={limit:.2f} -- run budget already "
                "exhausted, question not dispatched"
            )
        # forecasting_tools' own ForecastBot._run_individual_question already
        # wraps each question's research+forecast in `MonetaryCostManager` and
        # records the total on the returned report as `price_estimate` -- so
        # rather than nesting a second MonetaryCostManager here (which would
        # just re-total the same litellm callbacks), this reuses that value.
        # The per-question token counter below only backstops the case where
        # litellm has no price for a pinned model and price_estimate is 0.
        with _TokenUsageTracker() as tokens:
            report = await super()._run_individual_question(question)
        usd = report.price_estimate or 0.0
        self._question_costs_usd.append(usd)
        question_id = question.id_of_question or question.id_of_post
        researcher_name = self.get_llm("researcher", "string_name")
        # get_llm(..., "string_name") logs a warning when the llm is a
        # GeneralLlm (it is, for "default" -- see the llms= block below);
        # read .model directly to avoid a warning on every single question.
        default_llm = self.get_llm("default")
        default_name = (
            default_llm.model if isinstance(default_llm, GeneralLlm) else default_llm
        )
        logger.info(
            format_bot_cost_line(
                question_id=question_id,
                url=question.page_url,
                usd=usd,
                researcher=researcher_name,
                default_model=default_name,
                input_tokens=tokens.input_tokens,
                output_tokens=tokens.output_tokens,
            )
        )
        question_max_usd = get_max_usd_per_question()
        if question_max_usd and usd > question_max_usd:
            # Warning only (wave7 policy 8): the question already ran and is
            # never skipped for score reasons -- this just flags gate G0
            # overruns after the fact.
            logger.warning(
                format_bot_cost_over_question_cap_line(
                    question_id=question_id, usd=usd, cap=question_max_usd
                )
            )
        return report

    ##################################### RESEARCH #####################################

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            # Gate G0 spend guard (project-backlog#618 round 1): this is the
            # actual per-question stop point. `_concurrency_limiter`
            # (`_max_concurrent_questions = 1`) is the one place questions
            # are genuinely serialized -- the next question only enters this
            # block after the previous one's research call (its main cost)
            # has returned and updated current_usage -- so a check made
            # right here, not only at the top of _run_individual_question,
            # is what stops question N+1. See _active_run_budget_exhausted's
            # docstring for the full reasoning.
            exhausted = _active_run_budget_exhausted()
            if exhausted is not None:
                spent, limit = exhausted
                logger.warning(
                    format_bot_budget_exhausted_line(spent=spent, limit=limit)
                )
                question_id = question.id_of_question or question.id_of_post
                raise HardLimitExceededError(
                    f"event=bot_budget_exhausted question_id={question_id} "
                    f"spent={spent:.4f} limit={limit:.2f} -- run budget "
                    "already exhausted, question not dispatched"
                )
            research = ""
            researcher = self.get_llm("researcher")

            prompt = clean_indents(
                f"""
                You are an assistant to a superforecaster.
                The superforecaster will give you a question they intend to forecast on.
                To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
                You do not produce forecasts yourself.

                Question:
                {question.question_text}

                This question's outcome will be determined by the specific criteria below:
                {question.resolution_criteria}

                {question.fine_print}
                """
            )

            if isinstance(researcher, GeneralLlm):
                research = await researcher.invoke(prompt)
            elif (
                researcher == "asknews/news-summaries"
                or researcher == "asknews/deep-research/low-depth"
                or researcher == "asknews/deep-research/medium-depth"
                or researcher == "asknews/deep-research/high-depth"
            ):
                research = await AskNewsSearcher().call_preconfigured_version(
                    researcher, prompt
                )
            elif researcher.startswith("smart-searcher"):
                model_name = researcher.removeprefix("smart-searcher/")
                searcher = SmartSearcher(
                    model=model_name,
                    temperature=0,
                    num_searches_to_run=2,
                    num_sites_per_search=10,
                    use_advanced_filters=False,
                )
                research = await searcher.invoke(prompt)
            elif not researcher or researcher == "None" or researcher == "no_research":
                research = ""
            else:
                research = await self.get_llm("researcher", "llm").invoke(prompt)
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")
            return research

    ##################################### BINARY QUESTIONS #####################################

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.
            {self._get_conditional_disclaimer_if_necessary(question)}

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )

        return await self._binary_prompt_to_forecast(question, prompt)

    async def _binary_prompt_to_forecast(
        self,
        question: BinaryQuestion,
        prompt: str,
    ) -> ReasonedPrediction[float]:
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        question_id = question.id_of_question or question.id_of_post
        binary_prediction = parse_binary(reasoning)
        if binary_prediction is not None:
            logger.info(format_parse_path_line(question_id, "binary", "deterministic"))
        else:
            try:
                binary_prediction = await structure_output(
                    reasoning,
                    BinaryPrediction,
                    model=self.get_llm("parser", "llm"),
                    num_validation_samples=self._structure_output_validation_samples,
                )
            except Exception:
                logger.info(format_parse_path_line(question_id, "binary", "failed"))
                raise
            logger.info(format_parse_path_line(question_id, "binary", "llm_fallback"))
        decimal_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {decimal_pred}."
        )
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    ##################################### MULTIPLE CHOICE QUESTIONS #####################################

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of an scenario that results in an unexpected outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        return await self._multiple_choice_prompt_to_forecast(question, prompt)

    async def _multiple_choice_prompt_to_forecast(
        self,
        question: MultipleChoiceQuestion,
        prompt: str,
    ) -> ReasonedPrediction[PredictedOptionList]:
        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}

            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            Additionally, you may sometimes need to parse a 0% probability. Please do not skip options with 0% but rather make it an entry in your final list with 0% probability.
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        question_id = question.id_of_question or question.id_of_post
        predicted_option_list = parse_multiple_choice(reasoning, question.options)
        if predicted_option_list is not None:
            logger.info(format_parse_path_line(question_id, "mc", "deterministic"))
        else:
            try:
                predicted_option_list = await structure_output(
                    text_to_structure=reasoning,
                    output_type=PredictedOptionList,
                    model=self.get_llm("parser", "llm"),
                    num_validation_samples=self._structure_output_validation_samples,
                    additional_instructions=parsing_instructions,
                )
            except Exception:
                logger.info(format_parse_path_line(question_id, "mc", "failed"))
                raise
            logger.info(format_parse_path_line(question_id, "mc", "llm_fallback"))

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}."
        )
        return ReasonedPrediction(
            prediction_value=predicted_option_list, reasoning=reasoning
        )

    ##################################### NUMERIC QUESTIONS #####################################

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested and give your answer in these units (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there. The value for percentile 10 should always be less than the value for percentile 20, and so on.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX (lowest number value)
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX (highest number value)
            "
            """
        )
        return await self._numeric_prompt_to_forecast(question, prompt)

    async def _numeric_prompt_to_forecast(
        self,
        question: NumericQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a numeric question.
            - This text is trying to answer the numeric question: "{question.question_text}".
            - When parsing the text, please make sure to give the values (the ones assigned to percentiles) in terms of the correct units.
            - The units for the forecast are: {question.unit_of_measure}
            - Your work will be shown publicly with these units stated verbatim after the numbers your parse.
            - As an example, someone else guessed that the answer will be between {question.lower_bound} {question.unit_of_measure} and {question.upper_bound} {question.unit_of_measure}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - If the answer doesn't give the answer in the correct units, you should parse it in the right units. For instance if the answer gives numbers as $500,000,000 and units are "B $" then you should parse the answer as 0.5 (since $500,000,000 is $0.5 billion).
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            - Turn any values that are in scientific notation into regular numbers.
            """
        )
        question_id = question.id_of_question or question.id_of_post
        percentile_list = parse_numeric_percentiles(reasoning)
        if percentile_list is not None and not numeric_percentiles_within_bounds(
            percentile_list,
            question.lower_bound,
            question.upper_bound,
            question.open_lower_bound,
            question.open_upper_bound,
        ):
            # Well-formed six-percentile block, but implausible for the
            # question's bounds -- most likely a unit-scale mismatch the
            # deterministic parser can't see (project-backlog#613 round 3
            # review). Fall back to structure_output, which is unit-aware.
            percentile_list = None
        if percentile_list is not None:
            logger.info(format_parse_path_line(question_id, "numeric", "deterministic"))
        else:
            try:
                percentile_list = await structure_output(
                    reasoning,
                    list[Percentile],
                    model=self.get_llm("parser", "llm"),
                    additional_instructions=parsing_instructions,
                    num_validation_samples=self._structure_output_validation_samples,
                )
            except Exception:
                logger.info(format_parse_path_line(question_id, "numeric", "failed"))
                raise
            logger.info(format_parse_path_line(question_id, "numeric", "llm_fallback"))
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}."
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    ##################################### DATE QUESTIONS #####################################

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - This is a date question, and as such, the answer must be expressed in terms of dates.
            - The dates must be written in the format of YYYY-MM-DD. If hours matter, please append the date with the hour in UTC and military time: YYYY-MM-DDTHH:MM:SSZ.No other formatting is allowed.
            - Always start with a lower date chronologically and then increase from there.
            - Do NOT forget this. The dates must be written in chronological order starting at the earliest time at percentile 10 and increasing from there.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: YYYY-MM-DD (oldest date)
            Percentile 20: YYYY-MM-DD
            Percentile 40: YYYY-MM-DD
            Percentile 60: YYYY-MM-DD
            Percentile 80: YYYY-MM-DD
            Percentile 90: YYYY-MM-DD (newest date)
            "
            """
        )
        forecast = await self._date_prompt_to_forecast(question, prompt)
        return forecast

    async def _date_prompt_to_forecast(
        self,
        question: DateQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a date question.
            - This text is trying to answer the question: "{question.question_text}".
            - As an example, someone else guessed that the answer will be between {question.lower_bound} and {question.upper_bound}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - The output is given as dates/times please format it into a valid datetime parsable string. Assume midnight UTC if no hour is given.
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            """
        )
        question_id = question.id_of_question or question.id_of_post
        date_percentile_list = parse_date_percentiles(reasoning)
        if date_percentile_list is not None:
            logger.info(format_parse_path_line(question_id, "date", "deterministic"))
        else:
            try:
                date_percentile_list = await structure_output(
                    reasoning,
                    list[DatePercentile],
                    model=self.get_llm("parser", "llm"),
                    additional_instructions=parsing_instructions,
                    num_validation_samples=self._structure_output_validation_samples,
                )
            except Exception:
                logger.info(format_parse_path_line(question_id, "date", "failed"))
                raise
            logger.info(format_parse_path_line(question_id, "date", "llm_fallback"))

        percentile_list = [
            Percentile(
                percentile=percentile.percentile,
                value=percentile.value.timestamp(),
            )
            for percentile in date_percentile_list
        ]
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}."
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion | DateQuestion
    ) -> tuple[str, str]:
        if isinstance(question, NumericQuestion):
            if question.nominal_upper_bound is not None:
                upper_bound_number = question.nominal_upper_bound
            else:
                upper_bound_number = question.upper_bound
            if question.nominal_lower_bound is not None:
                lower_bound_number = question.nominal_lower_bound
            else:
                lower_bound_number = question.lower_bound
            unit_of_measure = question.unit_of_measure
        elif isinstance(question, DateQuestion):
            upper_bound_number = question.upper_bound.date().isoformat()
            lower_bound_number = question.lower_bound.date().isoformat()
            unit_of_measure = ""
        else:
            raise ValueError()

        if question.open_upper_bound:
            upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number} {unit_of_measure}."
        else:
            upper_bound_message = f"The outcome can not be higher than {upper_bound_number} {unit_of_measure}."

        if question.open_lower_bound:
            lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number} {unit_of_measure}."
        else:
            lower_bound_message = f"The outcome can not be lower than {lower_bound_number} {unit_of_measure}."
        return upper_bound_message, lower_bound_message

    ##################################### CONDITIONAL QUESTIONS #####################################

    async def _run_forecast_on_conditional(
        self, question: ConditionalQuestion, research: str
    ) -> ReasonedPrediction[ConditionalPrediction]:
        parent_info, full_research = await self._get_question_prediction_info(
            question.parent, research, "parent"
        )
        child_info, full_research = await self._get_question_prediction_info(
            question.child, research, "child"
        )
        yes_info, full_research = await self._get_question_prediction_info(
            question.question_yes, full_research, "yes"
        )
        no_info, full_research = await self._get_question_prediction_info(
            question.question_no, full_research, "no"
        )
        full_reasoning = clean_indents(
            f"""
            ## Parent Question Reasoning
            {parent_info.reasoning}
            ## Child Question Reasoning
            {child_info.reasoning}
            ## Yes Question Reasoning
            {yes_info.reasoning}
            ## No Question Reasoning
            {no_info.reasoning}
        """
        )
        full_prediction = ConditionalPrediction(
            parent=parent_info.prediction_value,  # type: ignore
            child=child_info.prediction_value,  # type: ignore
            prediction_yes=yes_info.prediction_value,  # type: ignore
            prediction_no=no_info.prediction_value,  # type: ignore
        )
        return ReasonedPrediction(
            reasoning=full_reasoning, prediction_value=full_prediction
        )

    async def _get_question_prediction_info(
        self, question: MetaculusQuestion, research: str, question_type: str
    ) -> tuple[ReasonedPrediction[PredictionTypes | PredictionAffirmed], str]:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        previous_forecasts = question.previous_forecasts
        if (
            question_type in ["parent", "child"]
            and previous_forecasts
            and question_type not in self.force_reforecast_in_conditional
        ):
            # TODO: add option to not affirm current parent/child forecasts, create new forecast
            previous_forecast = previous_forecasts[-1]
            current_utc_time = datetime.now(timezone.utc)
            if (
                previous_forecast.timestamp_end is None
                or previous_forecast.timestamp_end > current_utc_time
            ):
                pretty_value = DataOrganizer.get_readable_prediction(previous_forecast)  # type: ignore
                prediction = ReasonedPrediction(
                    prediction_value=PredictionAffirmed(),
                    reasoning=f"Already existing forecast reaffirmed at {pretty_value}.",
                )
                return (prediction, research)  # type: ignore
        info = await self._make_prediction(question, research)
        full_research = self._add_reasoning_to_research(research, info, question_type)
        return info, full_research  # type: ignore

    def _add_reasoning_to_research(
        self,
        research: str,
        reasoning: ReasonedPrediction[PredictionTypes],
        question_type: str,
    ) -> str:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        question_type = question_type.title()
        return clean_indents(
            f"""
            {research}
            ---
            ## {question_type} Question Information
            You have previously forecasted the {question_type} Question to the value: {DataOrganizer.get_readable_prediction(reasoning.prediction_value)}
            This is relevant information for your current forecast, but it is NOT your current forecast, but previous forecasting information that is relevant to your current forecast.
            The reasoning for the {question_type} Question was as such:
            ```
            {reasoning.reasoning}
            ```
            This is absolutely essential: do NOT use this reasoning to re-forecast the {question_type} question.
            """
        )

    def _get_conditional_disclaimer_if_necessary(
        self, question: MetaculusQuestion
    ) -> str:
        if question.conditional_type not in ["yes", "no"]:
            return ""
        return clean_indents(
            """
            As you are given a conditional question with a parent and child, you are to only forecast the **CHILD** question, given the parent question's resolution.
            You never re-forecast the parent question under any circumstances, but you use probabilistic reasoning, strongly considering the parent question's resolution, to forecast the child question.
            """
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run the template forecasting bot")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
        help="What to forecast on (default: tournament)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Forecast at most N questions (0 = unlimited; default: "
        "METAC_MAX_QUESTIONS env, else unlimited)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode

    # Clean keyless skip (project-backlog#618): the cron
    # (run_bot_on_tournament.yaml) fires every 20 minutes in tournament mode.
    # Before an LLM key secret is configured, check_environment(strict=True)
    # below only warns (METACULUS_TOKEN is the only hard requirement there),
    # so the run would otherwise proceed and error on every single question.
    # Exit 0 with one grep-able line instead.
    if run_mode == "tournament" and not has_llm_key():
        print("event=bot_skip reason=no_llm_key")
        sys.exit(0)

    # OpenRouter credit preflight (project-backlog#675): free key lookup
    # before any research/AskNews object is built. Skip only in tournament
    # mode; guard failures never block the run.
    _skip, _or_before = openrouter_guard.preflight()
    if _skip and run_mode == "tournament":
        sys.exit(0)
    atexit.register(openrouter_guard.log_spend, _or_before)

    check_environment(strict=True)
    publish_to_metaculus = True
    print_startup_banner(run_mode, will_publish=publish_to_metaculus)

    # Gate G0 (<=US$0.40/question) is measured from the per-question and
    # run-total `event=bot_cost*` log lines emitted by
    # FableForecastBot._run_individual_question below.
    print(
        f"Tournament ids: CURRENT_AI_COMPETITION_ID={MetaculusClient.CURRENT_AI_COMPETITION_ID} "
        f"CURRENT_MINIBENCH_ID={MetaculusClient.CURRENT_MINIBENCH_ID}"
    )

    # Pinned, cheap-by-design OpenRouter models (project-backlog#599): Sonnet
    # 5 as judge/default, Haiku 4.5 for parsing/summarizing, and AskNews for
    # research when both its env vars are configured, else Kimi K3 as the
    # cheap fallback researcher.
    researcher_model = select_researcher(
        os.getenv("ASKNEWS_CLIENT_ID"), os.getenv("ASKNEWS_SECRET")
    )
    ensure_model_pricing()
    template_bot = FableForecastBot(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        max_questions=args.max_questions,
        publish_reports_to_metaculus=publish_to_metaculus,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        llms={
            "default": GeneralLlm(
                model="openrouter/anthropic/claude-sonnet-5",
                temperature=0.3,
                timeout=60,
                allowed_tries=2,
            ),
            "summarizer": "openrouter/anthropic/claude-haiku-4.5",
            "researcher": researcher_model,
            "parser": "openrouter/anthropic/claude-haiku-4.5",
        },
    )

    # Per-mode tournament URL shown in the summary banner footer. These
    # piggyback on the forecasting_tools SDK constants and need updating
    # whenever those rotate seasons.
    TOURNAMENT_URLS = {
        "tournament": "https://www.metaculus.com/tournament/fall-futureeval-2026/",
        "metaculus_cup": "https://www.metaculus.com/tournament/metaculus-cup-summer-2025/",
        "test_questions": "https://www.metaculus.com/tournament/bot-testing-area/",
    }

    # Dispatch on mode. Each branch produces a list of ForecastReport (or
    # exceptions, since return_exceptions=True) which then flows into the
    # summary printers below.
    #
    # Gate G0 spend guard (project-backlog#618): one MonetaryCostManager
    # wraps the whole run. In tournament mode there are two sequential
    # asyncio.run() calls (seasonal, then MiniBench) -- if the first already
    # exhausted the run budget, the second is skipped entirely rather than
    # dispatched (run_tournament_mode). That is a coarse belt on top of the
    # real per-question stop (round 1): FableForecastBot.run_research checks
    # the same run budget right after acquiring `_concurrency_limiter`
    # (`_max_concurrent_questions = 1`), the one point questions are
    # genuinely serialized, so it stops question N+1 mid-batch within a
    # single forecast_on_tournament call -- see PR body for the full receipt.
    client = MetaculusClient()
    max_usd_per_run = get_max_usd_per_run()
    with MonetaryCostManager(hard_limit=max_usd_per_run) as run_cost_manager:
        if run_mode == "tournament":
            forecast_reports = run_tournament_mode(template_bot, client, run_cost_manager)
        elif run_mode == "metaculus_cup":
            # The Metaculus Cup may be uninitialized near the start of a season
            # (Jan/May/Sep). AXC_2025_TOURNAMENT_ID = 32564 and
            # AI_2027_TOURNAMENT_ID = "ai-2027" are also valid targets here.
            template_bot.skip_previously_forecasted_questions = False
            forecast_reports = asyncio.run(
                template_bot.forecast_on_tournament(
                    client.CURRENT_METACULUS_CUP_ID, return_exceptions=True
                )
            )
            if run_cost_manager.hard_limit and run_cost_manager.amount_left <= 0:
                logger.warning(
                    format_bot_budget_exhausted_line(
                        spent=run_cost_manager.current_usage,
                        limit=run_cost_manager.hard_limit,
                    )
                )
        elif run_mode == "test_questions":
            # The bot-testing-area tournament contains all question types and is
            # the recommended target for smoke-testing your bot.
            # https://www.metaculus.com/tournament/bot-testing-area/
            template_bot.skip_previously_forecasted_questions = False
            forecast_reports = asyncio.run(
                template_bot.forecast_on_tournament(
                    "bot-testing-area", return_exceptions=True
                )
            )
            if run_cost_manager.hard_limit and run_cost_manager.amount_left <= 0:
                logger.warning(
                    format_bot_budget_exhausted_line(
                        spent=run_cost_manager.current_usage,
                        limit=run_cost_manager.hard_limit,
                    )
                )

    logger.info(
        format_bot_cost_total_line(
            questions=len(template_bot._question_costs_usd),
            total_usd=sum(template_bot._question_costs_usd),
        )
    )
    real_reports, budget_stops = split_budget_stops(forecast_reports)
    for stop in budget_stops:
        logger.warning(f"event=bot_budget_exhausted error={stop}")
    questions_failed = sum(isinstance(r, BaseException) for r in real_reports)
    questions_budget_stopped = len(budget_stops)
    spend_line = format_bot_run_spend_line(
        usd=run_cost_manager.current_usage,
        limit=max_usd_per_run,
        questions_ok=len(real_reports) - questions_failed,
        questions_failed=questions_failed,
        questions_budget_stopped=questions_budget_stopped,
    )
    logger.info(spend_line)
    _summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if _summary_path:
        with open(_summary_path, "a") as _f:
            _f.write(spend_line + "\n")
    template_bot.log_report_summary(real_reports)
    print_run_summary_banner(
        forecast_reports,
        will_publish=publish_to_metaculus,
        tournament_url=TOURNAMENT_URLS.get(run_mode),
    )
