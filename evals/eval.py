from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import AlOstaAgent


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.json"
RESULTS_PATH = BASE_DIR / "eval_results.json"
DEFAULT_BATCH_SIZE = 5
DEFAULT_PAUSE_SECONDS = 300
DEFAULT_JUDGE_MODEL = os.getenv("GEMINI_JUDGE_MODEL", "gemini-2.5-flash")

# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------
# Each sample is scored ONLY on the metric that matches its own category.
CATEGORY_TOOL_CALLING = "tool calling"
CATEGORY_CORRECT_ARGS = "correct args"
CATEGORY_RESPONSE_QUALITY = "response quality"
CATEGORY_AMBIGUITY = "ambiguity handling"

# Maps a category to the single metric field used to score it.
CATEGORY_METRIC = {
    CATEGORY_TOOL_CALLING: "tool_calling",
    CATEGORY_CORRECT_ARGS: "correct_args",
    CATEGORY_RESPONSE_QUALITY: "response_quality",
    CATEGORY_AMBIGUITY: "ambiguity_handling",
}

# The LLM judge (Gemini) is used for THIS category only. Everything else is
# scored deterministically (tool calling / correct args) or with a local
# heuristic (ambiguity handling). Hallucination scoring has been removed.
LLM_JUDGE_CATEGORIES = {CATEGORY_RESPONSE_QUALITY}


RESPONSE_QUALITY_JUDGE_SYSTEM_PROMPT = """You are a STRICT evaluator for "Al-Osta", an Egyptian-Arabic public-transit and routing assistant for Alexandria, Egypt. The assistant always replies in Egyptian colloquial Arabic.

Your ONLY job is to score RESPONSE QUALITY: did the assistant's final response fully and correctly satisfy what the user actually asked for, given the tool outputs it had? Do NOT judge hallucination, tone, or tool selection here.

==================================================
INPUT YOU RECEIVE
==================================================
- query: the user's message (Egyptian Arabic, may be messy and may end with GPS coordinates).
- expected_behavior: a description / keywords of what a correct answer should contain.
- tool_outputs: the data the assistant had available when it answered.
- final_response: the assistant reply you must score.

==================================================
HOW TO JUDGE
==================================================
Check that final_response does ALL of the following:

1. Answers EVERY part of the request. Multi-part questions ("is there transport AND which is fastest") must have every part answered.

2. Honors the user's explicit intent and constraints. Common cases:
   - "ارخص / اقل تكلفة / ادفع قليل"  (cheapest)        -> must discuss cost and surface the cheapest option, not just dump routes.
   - "اسرع / اقصر وقت"               (fastest)         -> must discuss time and surface the fastest option.
   - "قارن / مقارنة"                 (compare)         -> must actually compare the named options on the requested dimensions AND state a clear conclusion.
   - "مواصلة واحدة / من غير ما اغير / من غير ما انزل واطلع" (single ride / no transfers) -> options must respect that.
   - "اقل مشي / شنطة تقيلة / مش عايز امشي" (minimal walking) -> must reflect low walking.
   - "هل فيه مواصلات؟"              (availability)     -> must explicitly answer yes/no, then give detail.
   - "متوسط السعر"                  (average price)    -> must give an average / price figure, not only a route list.
   - landmark / "علامة مميزة" / "معلومات عن مكان"      -> must give the requested info, not a generic route.

3. Is grounded in tool_outputs and is genuinely useful: specific, relevant routing guidance, not a vague or generic reply.

A response that merely LISTS routes while the user asked a specific question (cheapest, fastest, average price, compare, yes/no) is INCOMPLETE and must NOT get full marks unless it also directly answers that specific question.

If final_response is an error / apology with no useful content (e.g. a connection-error message like "عذرا، في مشكلة في الاتصال"), score 0.

==================================================
SCORING (0 to 1)
==================================================
- 1.0 = fully answers the request and honors every stated constraint.
- 0.5 = partially answers (e.g. gives valid routes but ignores the cheapest/compare/yes-no/average ask, or answers only some parts).
- 0.0 = wrong, irrelevant, empty, or an error message.

Set "passed" to true ONLY when the score is 1.0.

==================================================
OUTPUT FORMAT
==================================================
Return ONLY valid JSON, no markdown, no text outside the object:

{
  "response_quality_score": 1.0,
  "passed": true,
  "reasoning": "One or two sentences explaining the score, referencing the specific user ask and whether it was met."
}
"""


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Deterministic scorers (tool calling / correct args)
# ---------------------------------------------------------------------------

def extract_tool_names(Reasoner_output: list[dict[str, Any]]) -> list[str]:
    return [step["tool"] for step in Reasoner_output if isinstance(step, dict) and "tool" in step]


def _expected_tool_name(expected_tool: dict[str, Any]) -> str | None:
    """Expected-tool specs are inconsistent: some use 'name', some use 'tool'."""
    return expected_tool.get("name") or expected_tool.get("tool")


_MISSING = object()


def score_tool_calling(expected_tools: list[dict[str, Any]], actual_tools: list[str]) -> dict[str, Any]:
    """Partial-credit score for tool SELECTION, count-aware and explained.

    Uses an F1 over the multiset of tool names so it penalizes BOTH directions:
      - recall  drops when an expected call is missing (e.g. expected 2 get_routes, got 1),
      - precision drops when the agent calls an unnecessary tool.
    Order is intentionally ignored here (sequencing/args are scored separately).
    """
    expected_names = [_expected_tool_name(tool) for tool in expected_tools]
    actual_names = list(actual_tools)
    expected_counts = Counter(expected_names)
    actual_counts = Counter(actual_names)

    true_positives = sum((expected_counts & actual_counts).values())
    expected_total = len(expected_names)
    actual_total = len(actual_names)

    if expected_total == 0 and actual_total == 0:
        precision = recall = score = 1.0
    else:
        precision = true_positives / actual_total if actual_total else 0.0
        recall = true_positives / expected_total if expected_total else 0.0
        score = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) else 0.0

    missing = expected_counts - actual_counts  # under-called (recall misses)
    extra = actual_counts - expected_counts    # over-called (precision misses)

    def _fmt(counter: Counter) -> list[str]:
        return [name if n == 1 else f"{name} x{n}" for name, n in sorted(counter.items())]

    missing_list = _fmt(missing)
    extra_list = _fmt(extra)

    if score == 1.0:
        reasoning = f"Called exactly the expected tools: {_fmt(expected_counts) or 'none'}."
    else:
        bits = []
        if missing_list:
            bits.append(f"missing {missing_list}")
        if extra_list:
            bits.append(f"unexpected {extra_list}")
        reasoning = (
            f"Tool-call score {score:.2f} (precision {precision:.2f}, recall {recall:.2f}): "
            + "; ".join(bits)
            + f". Expected {expected_names}, got {actual_names}."
        )

    return {
        "metric": "tool_calling",
        "score": score,
        "source": "deterministic",
        "reasoning": reasoning,
        "details": {
            "expected": expected_names,
            "actual": actual_names,
            "matched": true_positives,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "missing": missing_list,
            "extra": extra_list,
        },
    }


def _match_expected_to_plan(
    expected_tools: list[dict[str, Any]],
    actual_plan: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    """Pair each expected tool with an actual step of the same name (greedy, in order).

    Matching by tool name rather than position means the score survives the agent
    reordering steps or inserting/dropping a tool.
    """
    used = [False] * len(actual_plan)
    pairs: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    for expected_tool in expected_tools:
        name = _expected_tool_name(expected_tool)
        match = None
        for i, step in enumerate(actual_plan):
            if used[i] or not isinstance(step, dict):
                continue
            if step.get("tool") == name:
                used[i] = True
                match = step
                break
        pairs.append((expected_tool, match))
    return pairs


def score_correct_args(
    expected_tools: list[dict[str, Any]],
    actual_plan: list[dict[str, Any]],
) -> dict[str, Any]:
    """Partial-credit score: fraction of expected STATIC args the agent got right.

    Dynamic references (e.g. "$destination.lat") can't be checked statically and
    are skipped (reported, not scored). A missing tool counts its static args as
    failed (or one unit if it had none), so dropping a required call lowers the score.
    """
    pairs = _match_expected_to_plan(expected_tools, actual_plan)

    total = 0
    matched = 0
    arg_details: list[dict[str, Any]] = []
    missing_tools: list[str] = []

    for expected_tool, step in pairs:
        name = _expected_tool_name(expected_tool)
        expected_args = expected_tool.get("args", {})
        static_keys = [k for k, v in expected_args.items() if not (isinstance(v, str) and v.startswith("$"))]

        if step is None:
            missing_tools.append(name)
            total += max(1, len(static_keys))  # penalize the dropped call
            for key in static_keys:
                arg_details.append(
                    {"tool": name, "arg": key, "expected": expected_args[key], "actual": "<tool not called>", "status": "missing_tool"}
                )
            if not static_keys:
                arg_details.append({"tool": name, "arg": "<call>", "expected": name, "actual": "<tool not called>", "status": "missing_tool"})
            continue

        actual_args = step.get("args", {})
        for key, expected_value in expected_args.items():
            if isinstance(expected_value, str) and expected_value.startswith("$"):
                arg_details.append(
                    {"tool": name, "arg": key, "expected": expected_value, "actual": actual_args.get(key), "status": "dynamic_skipped"}
                )
                continue

            total += 1
            actual_value = actual_args.get(key, _MISSING)
            if actual_value == expected_value:
                matched += 1
                status = "match"
            else:
                status = "missing_arg" if actual_value is _MISSING else "wrong_value"
            arg_details.append(
                {
                    "tool": name,
                    "arg": key,
                    "expected": expected_value,
                    "actual": None if actual_value is _MISSING else actual_value,
                    "status": status,
                }
            )

    score = round(matched / total, 4) if total else 1.0

    problems = [
        f"{d['tool']}.{d['arg']} (expected {d['expected']!r}, got {d['actual']!r})"
        for d in arg_details
        if d["status"] in ("wrong_value", "missing_arg", "missing_tool")
    ]
    if total == 0:
        reasoning = "No static args to verify (all expected args are dynamic or none expected)."
    elif not problems:
        reasoning = f"All {matched} expected static args correct (score {score:.2f})."
    else:
        reasoning = f"{matched}/{total} expected static args correct (score {score:.2f}). Issues: " + "; ".join(problems)

    return {
        "metric": "correct_args",
        "score": score,
        "source": "deterministic",
        "reasoning": reasoning,
        "details": {
            "matched": matched,
            "total_static_args": total,
            "missing_tools": missing_tools,
            "args": arg_details,
        },
    }


# ---------------------------------------------------------------------------
# Ambiguity handling (local heuristic, no LLM judge)
# ---------------------------------------------------------------------------

# Phrases that signal the assistant is asking the user for more detail rather
# than answering. Covers Egyptian colloquial + MSA clarification phrasings.
CLARIFICATION_CUES = (
    "؟",
    "?",
    "ممكن تقول",
    "ممكن تقولي",
    "ممكن توضح",
    "ممكن تحدد",
    "ممكن تبعت",
    "تقصد",
    "اقصد",
    "تحب",
    "وضحلي",
    "وضح لي",
    "حددلي",
    "حدد لي",
    "محتاج اعرف",
    "محتاج أعرف",
    "معنديش معلومات كافية",
    "لم أستطع العثور",
    "ماقدرتش أحدد",
    "نقطة البداية",
    "الاسم الكامل",
    "العنوان الكامل",
    "اسم المكان",
    "أي مكان",
    "انهي",
    "انهى",
    "أنهي",
    "من فين",
    "لفين",
    "منين",
    "فين",
    "ابعتلي",
    "ابعت الاسم",
)

# If the response already delivered concrete routing data, it is an answer, not
# a clarification request, regardless of any trailing question mark.
ANSWER_MARKERS = (
    "رحلة 1",
    "رحلة ١",
    "الوقت:",
    "التكلفة:",
    "route_number",
    "وتركب",
)


def _asked_for_clarification(final_response: str) -> bool:
    """True only when the reply asks the user for more info instead of answering."""
    if not final_response:
        return False
    text = final_response.strip()

    # A response that lists routes / costs / times is an answer, not a question.
    if any(marker in text for marker in ANSWER_MARKERS):
        return False

    return any(cue in text for cue in CLARIFICATION_CUES)


def evaluate_ambiguity_local(
    expected_behavior: dict[str, Any],
    Reasoner_output: list[dict[str, Any]],
    final_response: str,
) -> dict[str, Any]:
    should_ask = expected_behavior.get("should_ask_clarification", False)
    asked = _asked_for_clarification(final_response)
    no_tools = len(Reasoner_output) == 0

    if should_ask:
        passed = asked and no_tools
        if passed:
            reasoning = "Correctly asked for clarification without calling any tools."
        elif not no_tools:
            reasoning = "Clarification was expected, but the agent called tools instead of asking first."
        else:
            reasoning = "Clarification was expected, but the agent did not ask the user for the missing detail."
    else:
        # Request was clear: the agent should act, not stall with a question.
        passed = not (asked and no_tools)
        reasoning = (
            "Request was clear and the agent proceeded instead of stalling."
            if passed
            else "Request was clear, but the agent asked for clarification instead of acting."
        )

    return {
        "ambiguity_score": 1.0 if passed else 0.0,
        "passed": passed,
        "reasoning": reasoning,
        "source": "local",
    }


# ---------------------------------------------------------------------------
# Response quality (Gemini judge, with a local keyword fallback)
# ---------------------------------------------------------------------------

def evaluate_response_quality_local(final_response: str, expected_behavior: dict[str, Any]) -> dict[str, Any]:
    expected_keywords = expected_behavior.get("response_should_include", [])
    if expected_keywords:
        passed = all(keyword in final_response for keyword in expected_keywords)
    else:
        quality_keywords = ["الوقت", "التكلفة", "مواصلات", "اركب", "امشي"]
        passed = sum(1 for keyword in quality_keywords if keyword in final_response) >= 2

    return {
        "response_quality_score": 1.0 if passed else 0.0,
        "passed": passed,
        "reasoning": "Local keyword fallback used because the Gemini judge was unavailable or disabled.",
        "source": "local_fallback",
    }


def _extract_json_object(text: str) -> dict[str, Any]:
    clean_text = text.strip()
    if clean_text.startswith("```"):
        clean_text = clean_text.strip("`")
        if clean_text.startswith("json"):
            clean_text = clean_text[4:]
        clean_text = clean_text.strip()

    start_index = clean_text.find("{")
    end_index = clean_text.rfind("}")
    if start_index != -1 and end_index != -1 and end_index >= start_index:
        clean_text = clean_text[start_index : end_index + 1]

    parsed = json.loads(clean_text)
    if not isinstance(parsed, dict):
        raise ValueError("Judge response must be a JSON object")
    return parsed


JUDGE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Human-readable hints for the status codes the Gemini API returns.
_JUDGE_STATUS_HINTS = {
    400: "Bad request: usually an invalid/expired GEMINI_API_KEY (API_KEY_INVALID). Get one at https://aistudio.google.com/apikey.",
    401: "Unauthorized: the GEMINI_API_KEY is missing or invalid.",
    403: "Permission denied: the key can't access this model, or the API isn't enabled for the project.",
    404: f"Model not found: '{{model}}' is not a valid Gemini model. Try GEMINI_JUDGE_MODEL={DEFAULT_JUDGE_MODEL}.",
    429: "Rate limited (free-tier quota). Slow down with --pause-seconds, or use a lighter model like gemini-2.5-flash-lite.",
}


def _describe_judge_error(exc: Exception, model: str) -> str:
    """Turn an opaque API exception into an actionable, status-aware message."""
    status = getattr(exc, "status_code", None)
    body = ""
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            body = response.text
        except Exception:
            body = ""
    hint = _JUDGE_STATUS_HINTS.get(status, "")
    if hint:
        hint = hint.format(model=model)
    parts = [f"Gemini judge HTTP {status}" if status else f"Gemini judge error: {type(exc).__name__}"]
    if hint:
        parts.append(hint)
    detail = (body or str(exc)).strip()
    if detail:
        parts.append(f"Detail: {detail[:300]}")
    return " | ".join(parts)


def _make_judge_client(api_key: str):
    # Gemini speaks the OpenAI protocol, so we reuse the openai SDK and just
    # point it at the Gemini compatibility endpoint.
    openai_module = importlib.import_module("openai")
    return openai_module.OpenAI(api_key=api_key, base_url=JUDGE_BASE_URL)


def verify_judge(api_key: str | None, model: str | None = None) -> tuple[bool, str]:
    """Cheap preflight: one tiny call so failures surface BEFORE running the agent.

    Returns (ok, message). On failure the message is the actionable error string.
    """
    model = model or DEFAULT_JUDGE_MODEL
    if not api_key:
        return False, "No Gemini key for the judge; response quality will use the local fallback."
    try:
        client = _make_judge_client(api_key)
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0,
        )
        return True, f"Gemini judge OK (model={model})."
    except Exception as exc:
        return False, _describe_judge_error(exc, model)


def _call_judge(system_prompt: str, user_prompt: str, api_key: str, model: str | None = None) -> dict[str, Any]:
    model = model or DEFAULT_JUDGE_MODEL
    client = _make_judge_client(api_key)
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
    except Exception as exc:
        raise RuntimeError(_describe_judge_error(exc, model)) from exc

    content = completion.choices[0].message.content if completion.choices else ""
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Gemini judge returned empty content")

    result = _extract_json_object(content)
    result["raw_content"] = content
    result["usage"] = completion.usage.model_dump() if getattr(completion, "usage", None) else {}
    result["source"] = "gemini"
    return result


def evaluate_response_quality(
    query: str,
    expected_behavior: dict[str, Any],
    Reasoner_output: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
    final_response: str,
    judge_api_key: str | None,
    judge_model: str | None = None,
    use_llm_judge: bool = False,
) -> dict[str, Any]:
    if not use_llm_judge or not judge_api_key:
        return evaluate_response_quality_local(final_response, expected_behavior)

    user_prompt = json.dumps(
        {
            "query": query,
            "expected_behavior": expected_behavior,
            "tool_outputs": tool_results,
            "final_response": final_response,
        },
        ensure_ascii=False,
        indent=2,
    )
    try:
        return _call_judge(RESPONSE_QUALITY_JUDGE_SYSTEM_PROMPT, user_prompt, judge_api_key, judge_model)
    except Exception as exc:
        fallback = evaluate_response_quality_local(final_response, expected_behavior)
        fallback["judge_error"] = str(exc)
        return fallback


# ---------------------------------------------------------------------------
# Tokens + agent runner
# ---------------------------------------------------------------------------

def summarize_token_usage(token_usage: dict[str, Any]) -> int:
    if not isinstance(token_usage, dict):
        return 0

    total_tokens = token_usage.get("total_tokens")
    if isinstance(total_tokens, int):
        return total_tokens

    total = 0
    for usage in token_usage.values():
        if not isinstance(usage, dict):
            continue
        for key in ("total_token_count", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int):
                total += value
                break
    return total


def run_agent(query: str, api_key: str, Reasoner_model: str | None = None, synthesizer_model: str | None = None) -> dict[str, Any]:
    agent = AlOstaAgent(api_key, Reasoner_model=Reasoner_model, synthesizer_model=synthesizer_model)
    return agent.process_query_with_trace(query)


# Substrings that mark a temporary, retryable API failure (Gemini overload, quota).
_TRANSIENT_MARKERS = (
    "503", "unavailable", "high demand", "overloaded",
    "429", "resource_exhausted", "rate limit", "quota",
    "timeout", "timed out", "deadline",
)


def _is_transient(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def run_agent_with_retry(
    query: str,
    api_key: str,
    Reasoner_model: str | None = None,
    synthesizer_model: str | None = None,
    max_retries: int = 4,
    base_delay: int = 10,
) -> dict[str, Any]:
    """Run the agent, retrying transient failures with linear backoff.

    If it still fails (or fails for a non-transient reason), returns an empty
    trace tagged with ``agent_error`` so the eval loop continues instead of
    crashing and losing the whole batch's progress.
    """
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return run_agent(query, api_key, Reasoner_model, synthesizer_model)
        except Exception as exc:  # noqa: BLE001 - we want to classify and maybe retry
            last_error = exc
            if attempt >= max_retries or not _is_transient(exc):
                break
            delay = base_delay * attempt
            print(f"  transient agent error (attempt {attempt}/{max_retries}): {exc}. retrying in {delay}s...")
            time.sleep(delay)

    print(f"  agent failed permanently: {last_error}")
    return {
        "Reasoner_output": [],
        "tool_results": [],
        "final_response": "",
        "token_usage": {},
        "agent_error": str(last_error),
    }


# ---------------------------------------------------------------------------
# Per-sample scoring (each sample scored ONLY on its own category)
# ---------------------------------------------------------------------------

def _blank_result(query: str, category: str, agent_result: dict[str, Any]) -> dict[str, Any]:
    Reasoner_output = agent_result.get("Reasoner_output", [])
    tool_results = agent_result.get("tool_results", [])
    final_response = agent_result.get("final_response", "")
    tokens = summarize_token_usage(agent_result.get("token_usage", {}))

    return {
        "query": query,
        "category": category,
        "tool_calling": None,
        "correct_args": None,
        "response_quality": None,
        "ambiguity_handling": None,
        "tokens": tokens,
        "Reasoner_output": Reasoner_output,
        "tool_results": tool_results,
        "final_response": final_response,
        "judgment": None,
        "reasoning": None,
        "agent_error": agent_result.get("agent_error"),
    }


def evaluate_sample(
    sample: dict[str, Any],
    agent_result: dict[str, Any],
    judge_api_key: str | None = None,
    judge_model: str | None = None,
) -> dict[str, Any]:
    query = sample["input"]
    expected_behavior = sample.get("expected_behavior", {})
    category = sample.get("category", "")

    result = _blank_result(query, category, agent_result)
    Reasoner_output = result["Reasoner_output"]
    tool_results = result["tool_results"]
    final_response = result["final_response"]
    expected_tools = expected_behavior.get("expected_tools", [])

    if category == CATEGORY_TOOL_CALLING:
        judgment = score_tool_calling(expected_tools, extract_tool_names(Reasoner_output))
        result["tool_calling"] = judgment["score"]
        result["judgment"] = judgment

    elif category == CATEGORY_CORRECT_ARGS:
        judgment = score_correct_args(expected_tools, Reasoner_output)
        result["correct_args"] = judgment["score"]
        # tool_calling kept as informational context only (not the scored metric).
        result["tool_calling"] = score_tool_calling(expected_tools, extract_tool_names(Reasoner_output))["score"]
        result["judgment"] = judgment

    elif category == CATEGORY_RESPONSE_QUALITY:
        judgment = evaluate_response_quality(
            query,
            expected_behavior,
            Reasoner_output,
            tool_results,
            final_response,
            judge_api_key,
            judge_model,
            use_llm_judge=category in LLM_JUDGE_CATEGORIES,
        )
        score = float(judgment.get("response_quality_score", 0.0))
        result["response_quality"] = score
        result["judgment"] = judgment

    elif category == CATEGORY_AMBIGUITY:
        judgment = evaluate_ambiguity_local(expected_behavior, Reasoner_output, final_response)
        score = float(judgment.get("ambiguity_score", 0.0))
        result["ambiguity_handling"] = score
        result["judgment"] = judgment

    else:
        result["judgment"] = {"metric": "unknown", "score": None, "source": "skipped", "reasoning": "Unknown category."}

    # Surface a flat, human-readable explanation on every result so you can see
    # WHY a score landed where it did without digging into the judgment object.
    judgment = result["judgment"] or {}
    reasoning = judgment.get("reasoning")
    if isinstance(reasoning, dict):  # response-quality fallback sometimes nests it
        reasoning = "; ".join(f"{k}: {v}" for k, v in reasoning.items())
    if result.get("agent_error"):
        reasoning = f"AGENT ERROR (no trace produced): {result['agent_error']}. " + (reasoning or "")
    result["reasoning"] = reasoning

    return result


# ---------------------------------------------------------------------------
# Evaluation loop + reporting
# ---------------------------------------------------------------------------

def evaluate_dataset(
    dataset: list[dict[str, Any]],
    api_key: str,
    judge_api_key: str | None,
    Reasoner_model: str | None = None,
    synthesizer_model: str | None = None,
    judge_model: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    pause_seconds: int = DEFAULT_PAUSE_SECONDS,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for index, sample in enumerate(dataset):
        if index > 0 and index % batch_size == 0:
            print("\n" + "=" * 80)
            print(f"Batch limit reached. Sleeping for {pause_seconds} seconds before continuing...")
            time.sleep(pause_seconds)

        query = sample["input"]
        print("=" * 80)
        print(f"RUNNING TEST {index + 1}/{len(dataset)} [{sample.get('category', '')}]: {query}")

        agent_result = run_agent_with_retry(
            query,
            api_key=api_key,
            Reasoner_model=Reasoner_model,
            synthesizer_model=synthesizer_model,
        )

        result = evaluate_sample(sample, agent_result, judge_api_key, judge_model)
        results.append(result)
        print(result)

    return results


def build_final_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, list[float]] = defaultdict(list)
    total_tokens = 0

    for result in results:
        total_tokens += result.get("tokens", 0) or 0
        category = result.get("category", "")
        metric = CATEGORY_METRIC.get(category)
        if metric and result.get(metric) is not None:
            by_category[category].append(float(result[metric]))

    per_category = {
        category: {
            "score": sum(scores) / len(scores) if scores else 0.0,
            "count": len(scores),
        }
        for category, scores in by_category.items()
    }

    total_tests = len(results)
    return {
        "tool_calling_accuracy": per_category.get(CATEGORY_TOOL_CALLING, {}).get("score", 0.0),
        "correct_args_accuracy": per_category.get(CATEGORY_CORRECT_ARGS, {}).get("score", 0.0),
        "response_quality_score": per_category.get(CATEGORY_RESPONSE_QUALITY, {}).get("score", 0.0),
        "ambiguity_handling_score": per_category.get(CATEGORY_AMBIGUITY, {}).get("score", 0.0),
        "per_category": per_category,
        "average_tokens": total_tokens / total_tests if total_tests else 0.0,
        "total_tests": total_tests,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Alexandria evals against evals/dataset.json")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Number of queries to run before pausing")
    parser.add_argument("--pause-seconds", type=int, default=DEFAULT_PAUSE_SECONDS, help="How long to sleep between batches")
    parser.add_argument("--output", default=str(RESULTS_PATH), help="Path to save detailed JSON results")
    parser.add_argument("--Reasoner-model", default=os.getenv("Reasoner_MODEL"), help="Override the Reasoner model name")
    parser.add_argument("--synthesizer-model", default=os.getenv("SLM_MODEL"), help="Override the synthesizer model name")
    parser.add_argument("--judge-model", default=os.getenv("GEMINI_JUDGE_MODEL", DEFAULT_JUDGE_MODEL), help="Override the Gemini judge model name")
    parser.add_argument("--ignore-judge-errors", action="store_true", help="Continue with the local fallback even if the Gemini judge preflight fails")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    api_key = os.getenv("GEMINI_API_KEY")
    # The judge runs on Gemini too. Allow a separate key, but default to the agent's.
    judge_api_key = os.getenv("GEMINI_JUDGE_API_KEY") or api_key

    if not api_key:
        raise SystemExit("GEMINI_API_KEY is required to run evals")

    ok, message = verify_judge(judge_api_key, args.judge_model)
    print(("OK: " if ok else "WARNING: ") + message)
    if not ok and judge_api_key:
        # A key is set but the judge is broken (bad key / wrong model / quota).
        # Stop now instead of burning quota on 30 agent runs the judge can't score.
        if not args.ignore_judge_errors:
            raise SystemExit(
                "Gemini judge preflight failed (see above). Fix the key/model at "
                "https://aistudio.google.com/apikey, or pass --ignore-judge-errors "
                "to continue using the local fallback."
            )

    dataset = load_dataset()
    results = evaluate_dataset(
        dataset,
        api_key=api_key,
        judge_api_key=judge_api_key,
        Reasoner_model=args.Reasoner_model,
        synthesizer_model=args.synthesizer_model,
        judge_model=args.judge_model,
        batch_size=args.batch_size,
        pause_seconds=args.pause_seconds,
    )

    report = build_final_report(results)

    print("\n")
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    output_path = Path(args.output)
    output_path.write_text(
        json.dumps({"summary": report, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved detailed results to {output_path}")


if __name__ == "__main__":
    main()