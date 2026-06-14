from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
import importlib
import sys

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
DEFAULT_GROK_MODEL = os.getenv("GROK_MODEL", "grok-4-latest")
JUDGE_CATEGORIES = {"response quality", "ambiguity handling"}

RESPONSE_QUALITY_JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for an AI transportation and routing agent.

Your task is to evaluate whether the agent response correctly satisfies the user's request based on:

1. User intent
2. Tool outputs
3. Expected behavior
4. Factual consistency

You MUST evaluate the response carefully and strictly.

==================================================
EVALUATION DIMENSIONS
==================================================

You will evaluate TWO things only:

1. response_quality
2. hallucination

--------------------------------------------------
1. RESPONSE QUALITY
--------------------------------------------------

Determine whether the agent fully answered the user's request.

Check whether the response:

- addresses ALL important parts of the user's query
- respects user constraints
- provides useful and relevant routing guidance
- answers comparison questions correctly
- gives complete information when needed
- avoids vague or generic replies

Examples:
- If the user asked for cheapest route, the response should mention cost.
- If the user asked for fastest route, the response should mention time.
- If the user asked for comparison, the response should compare options clearly.
- If the user asked for minimal walking, the response should reflect that.
- If the user asked whether transportation is available, the response should explicitly answer that.

A response should FAIL if:
- it ignores important parts of the question
- gives generic routing without addressing constraints
- avoids comparison when comparison is requested
- misses key information requested by the user

--------------------------------------------------
2. HALLUCINATION
--------------------------------------------------

Determine whether the response contains information NOT supported by the tool outputs.

You MUST compare the final response against the provided tool outputs.

Hallucination includes:
- inventing routes
- inventing transportation methods
- inventing traffic conditions
- inventing prices
- inventing travel times
- inventing landmarks
- making unsupported claims

The response should FAIL hallucination if it includes factual claims unsupported by tool outputs.

==================================================
SCORING
==================================================

Return scores from 0 to 1.

- 1.0 = perfect
- 0.5 = partially correct
- 0.0 = incorrect

==================================================
OUTPUT FORMAT
==================================================

Return ONLY valid JSON.

Example:

{
	"response_quality_score": 1.0,
	"hallucination_score": 1.0,
	"passed": true,
	"reasoning": {
		"response_quality": "The response correctly compared both Mahmoudia and Abou Qir in terms of time and cost.",
		"hallucination": "All claims were grounded in tool outputs."
	}
}

Do not include markdown.
Do not include explanations outside JSON.
"""

AMIBGUITY_JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for an AI transportation and routing agent.

Your task is to evaluate whether the agent correctly handled ambiguity or requested clarification when needed.

You MUST evaluate strictly based on the user request, the expected behavior, and the agent response.

==================================================
EVALUATION DIMENSIONS
==================================================

You will evaluate ONE thing only:

1. ambiguity_handling

--------------------------------------------------
AMBIGUITY HANDLING
--------------------------------------------------

Determine whether the agent response matches the expected clarification behavior.

Check whether the response:

- asks for clarification when the request is underspecified or ambiguous
- avoids asking unnecessary clarification when the request is already clear
- does not call tools when the expected behavior says to clarify first
- remains aligned with the user's intent

A response should FAIL if:
- it gives a route when clarification is required
- it asks a clarification question when the request is already clear
- it ignores the ambiguity-related expected behavior

==================================================
SCORING
==================================================

Return scores from 0 to 1.

- 1.0 = perfect
- 0.5 = partially correct
- 0.0 = incorrect

==================================================
OUTPUT FORMAT
==================================================

Return ONLY valid JSON.

Example:

{
	"ambiguity_score": 1.0,
	"passed": true,
	"reasoning": "The agent correctly asked for clarification without calling tools."
}

Do not include markdown.
Do not include explanations outside JSON.
"""


# =========================
# LOAD DATASET
# =========================

with open(DATASET_PATH, "r", encoding="utf-8") as f:
	dataset = json.load(f)


# =========================
# HELPER FUNCTIONS
# =========================

def extract_tool_names(planner_output: list[dict[str, Any]]) -> list[str]:
	return [step["tool"] for step in planner_output if isinstance(step, dict) and "tool" in step]


def compare_tools(expected_tools: list[dict[str, Any]], actual_tools: list[str]) -> bool:
	expected_names = [tool["name"] for tool in expected_tools]
	return set(expected_names) == set(actual_tools)


def compare_args(expected_tools: list[dict[str, Any]], actual_plan: list[dict[str, Any]]) -> bool:
	"""
	checks only static args
	ignores dynamic refs like $start.lat
	"""

	for expected_tool, actual_tool in zip(expected_tools, actual_plan):
		expected_args = expected_tool.get("args", {})
		actual_args = actual_tool.get("args", {})

		for key, expected_value in expected_args.items():
			if isinstance(expected_value, str) and expected_value.startswith("$"):
				continue

			actual_value = actual_args.get(key)
			if actual_value != expected_value:
				return False

	return True


def _asked_for_clarification(final_response: str) -> bool:
	return any(
		cue in final_response
		for cue in ("؟", "انهي", "تقصد", "فين", "توضح", "حدد")
	)


def evaluate_response_quality_local(final_response: str, expected_behavior: dict[str, Any]) -> dict[str, Any]:
	expected_keywords = expected_behavior.get("response_should_include", [])
	if expected_keywords:
		passed = all(keyword in final_response for keyword in expected_keywords)
		return {
			"response_quality_score": 1.0 if passed else 0.0,
			"passed": passed,
			"reasoning": {
				"response_quality": "Used local keyword match fallback because Grok judge was unavailable or disabled.",
				"hallucination": "Not evaluated by the Grok judge in this category.",
			},
			"source": "local_fallback",
		}

	quality_keywords = ["الوقت", "التكلفة", "مواصلات", "اركب", "امشي"]
	score = sum(1 for keyword in quality_keywords if keyword in final_response)
	passed = score >= 2
	return {
		"response_quality_score": 1.0 if passed else 0.0,
		"passed": passed,
		"reasoning": {
			"response_quality": "Used local keyword fallback because Grok judge was unavailable or disabled.",
			"hallucination": "Not evaluated by the Grok judge in this category.",
		},
		"source": "local_fallback",
	}


def evaluate_ambiguity_local(expected_behavior: dict[str, Any], planner_output: list[dict[str, Any]], final_response: str) -> dict[str, Any]:
	should_ask = expected_behavior.get("should_ask_clarification", False)
	asked_question = _asked_for_clarification(final_response)
	no_tools_called = len(planner_output) == 0
	passed = (asked_question and no_tools_called) if should_ask else True
	return {
		"ambiguity_score": 1.0 if passed else 0.0,
		"passed": passed,
		"reasoning": "Used local clarification check fallback because Grok judge was unavailable or disabled.",
		"source": "local_fallback",
	}


def evaluate_hallucination_local(planner_output: list[dict[str, Any]], final_response: str) -> dict[str, Any]:
	planner_text = json.dumps(planner_output, ensure_ascii=False)
	hallucinated_words = ["ترام", "مترو", "اتوبيس"]
	hallucinations = [word for word in hallucinated_words if word in final_response and word not in planner_text]
	passed = len(hallucinations) == 0
	return {
		"hallucination_score": 1.0 if passed else 0.0,
		"passed": passed,
		"reasoning": "Used local hallucination check fallback because the Grok judge is reserved for response quality and ambiguity only.",
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


def _call_grok_judge(system_prompt: str, user_prompt: str, api_key: str, model: str | None = None) -> dict[str, Any]:
	openai_module = importlib.import_module("openai")
	client = openai_module.OpenAI(
		api_key=api_key,
		base_url="https://api.x.ai/v1",
	)
	completion = client.chat.completions.create(
		model=model or DEFAULT_GROK_MODEL,
		messages=[
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_prompt},
		],
		temperature=0,
	)

	content = completion.choices[0].message.content if completion.choices else ""
	if not isinstance(content, str) or not content.strip():
		raise RuntimeError("Grok judge returned empty content")

	result = _extract_json_object(content)
	result["raw_content"] = content
	result["usage"] = completion.usage.model_dump() if getattr(completion, "usage", None) else {}
	return result


def evaluate_response_quality(
	query: str,
	expected_behavior: dict[str, Any],
	planner_output: list[dict[str, Any]],
	tool_results: list[dict[str, Any]],
	final_response: str,
	judge_api_key: str,
	judge_model: str | None = None,
	use_grok_judge: bool = False,
) -> dict[str, Any]:
	if not use_grok_judge:
		return evaluate_response_quality_local(final_response, expected_behavior)

	user_prompt = json.dumps(
		{
			"query": query,
			"expected_behavior": expected_behavior,
			"planner_output": planner_output,
			"tool_outputs": tool_results,
			"final_response": final_response,
		},
		ensure_ascii=False,
		indent=2,
	)
	try:
		return _call_grok_judge(RESPONSE_QUALITY_JUDGE_SYSTEM_PROMPT, user_prompt, judge_api_key, judge_model)
	except Exception as exc:
		fallback = evaluate_response_quality_local(final_response, expected_behavior)
		fallback["judge_error"] = str(exc)
		return fallback


def evaluate_ambiguity(
	query: str,
	expected_behavior: dict[str, Any],
	planner_output: list[dict[str, Any]],
	final_response: str,
	judge_api_key: str,
	judge_model: str | None = None,
	use_grok_judge: bool = False,
) -> dict[str, Any]:
	if not use_grok_judge:
		return evaluate_ambiguity_local(expected_behavior, planner_output, final_response)

	user_prompt = json.dumps(
		{
			"query": query,
			"expected_behavior": expected_behavior,
			"planner_output": planner_output,
			"final_response": final_response,
		},
		ensure_ascii=False,
		indent=2,
	)
	try:
		return _call_grok_judge(AMIBGUITY_JUDGE_SYSTEM_PROMPT, user_prompt, judge_api_key, judge_model)
	except Exception as exc:
		fallback = evaluate_ambiguity_local(expected_behavior, planner_output, final_response)
		fallback["judge_error"] = str(exc)
		return fallback


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


# =========================
# AGENT RUNNER
# =========================

def run_agent(query: str, api_key: str, planner_model: str | None = None, synthesizer_model: str | None = None) -> dict[str, Any]:
	agent = AlOstaAgent(
		api_key,
		planner_model=planner_model,
		synthesizer_model=synthesizer_model,
	)
	return agent.process_query_with_trace(query)


# =========================
# EVALUATION LOOP
# =========================

def evaluate_dataset(api_key: str, judge_api_key: str, planner_model: str | None = None, synthesizer_model: str | None = None, judge_model: str | None = None, batch_size: int = DEFAULT_BATCH_SIZE, pause_seconds: int = DEFAULT_PAUSE_SECONDS) -> tuple[list[dict[str, Any]], dict[str, float], int]:
	results: list[dict[str, Any]] = []
	summary = defaultdict(float)
	total_tokens = 0

	for index, sample in enumerate(dataset):
		if index > 0 and index % batch_size == 0:
			print("\n" + "=" * 80)
			print(f"Batch limit reached. Sleeping for {pause_seconds} seconds before continuing...")
			time.sleep(pause_seconds)

		query = sample["input"]
		expected_behavior = sample["expected_behavior"]
		category = sample.get("category", "")

		print("=" * 80)
		print(f"RUNNING TEST {index + 1}/{len(dataset)}: {query}")

		agent_result = run_agent(
			query,
			api_key=api_key,
			planner_model=planner_model,
			synthesizer_model=synthesizer_model,
		)

		planner_output = agent_result.get("planner_output", [])
		tool_results = agent_result.get("tool_results", [])
		final_response = agent_result.get("final_response", "")
		token_usage = agent_result.get("token_usage", {})
		tokens = summarize_token_usage(token_usage)
		total_tokens += tokens

		actual_tools = extract_tool_names(planner_output)
		tool_calling_score = compare_tools(
			expected_behavior.get("expected_tools", []),
			actual_tools,
		)

		args_score = compare_args(
			expected_behavior.get("expected_tools", []),
			planner_output,
		)

		use_response_quality_judge = category in JUDGE_CATEGORIES and category == "response quality"
		use_ambiguity_judge = category in JUDGE_CATEGORIES and category == "ambiguity handling"

		response_quality_judgment = evaluate_response_quality(
			query,
			expected_behavior,
			planner_output,
			tool_results,
			final_response,
			judge_api_key,
			judge_model,
			use_grok_judge=use_response_quality_judge,
		)
		response_quality_score = float(response_quality_judgment.get("response_quality_score", 0.0))

		ambiguity_judgment = evaluate_ambiguity(
			query,
			expected_behavior,
			planner_output,
			final_response,
			judge_api_key,
			judge_model,
			use_grok_judge=use_ambiguity_judge,
		)
		ambiguity_score = float(ambiguity_judgment.get("ambiguity_score", 0.0))

		hallucination_judgment = evaluate_hallucination_local(planner_output, final_response)
		hallucination_score = float(hallucination_judgment.get("hallucination_score", 0.0))

		test_result = {
			"query": query,
			"tool_calling": tool_calling_score,
			"correct_args": args_score,
			"response_quality": response_quality_score,
			"ambiguity_handling": ambiguity_score,
			"hallucination_free": hallucination_score,
			"tokens": tokens,
			"planner_output": planner_output,
			"tool_results": tool_results,
			"final_response": final_response,
			"category": category,
			"response_quality_judgment": response_quality_judgment,
			"ambiguity_judgment": ambiguity_judgment,
			"hallucination_judgment": hallucination_judgment,
		}

		results.append(test_result)

		summary["tool_calling"] += float(tool_calling_score)
		summary["correct_args"] += float(args_score)
		summary["response_quality"] += response_quality_score
		summary["ambiguity_handling"] += ambiguity_score
		summary["hallucination_free"] += hallucination_score

		print(test_result)

	return results, summary, total_tokens


def build_final_report(results: list[dict[str, Any]], summary: dict[str, float], total_tokens: int) -> dict[str, Any]:
	total_tests = len(results)
	report = {
		"tool_calling_accuracy": summary["tool_calling"] / total_tests if total_tests else 0.0,
		"correct_args_accuracy": summary["correct_args"] / total_tests if total_tests else 0.0,
		"response_quality_score": summary["response_quality"] / total_tests if total_tests else 0.0,
		"ambiguity_handling_score": summary["ambiguity_handling"] / total_tests if total_tests else 0.0,
		"hallucination_free_rate": summary["hallucination_free"] / total_tests if total_tests else 0.0,
		"average_tokens": total_tokens / total_tests if total_tests else 0.0,
		"total_tests": total_tests,
	}
	return report


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run Alexandria evals against evals/dataset.json")
	parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Number of queries to run before pausing")
	parser.add_argument("--pause-seconds", type=int, default=DEFAULT_PAUSE_SECONDS, help="How long to sleep between batches")
	parser.add_argument("--output", default=str(RESULTS_PATH), help="Path to save detailed JSON results")
	parser.add_argument("--planner-model", default=os.getenv("PLANNER_MODEL"), help="Override the planner model name")
	parser.add_argument("--synthesizer-model", default=os.getenv("SLM_MODEL"), help="Override the synthesizer model name")
	parser.add_argument("--judge-model", default=os.getenv("GROK_MODEL", DEFAULT_GROK_MODEL), help="Override the Grok judge model name")
	return parser.parse_args()


def main() -> None:
	load_dotenv()
	args = parse_args()
	api_key = os.getenv("GEMINI_API_KEY")
	judge_api_key = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")

	if not api_key:
		raise SystemExit("GEMINI_API_KEY is required to run evals")
	if not judge_api_key:
		print("Warning: XAI_API_KEY/GROK_API_KEY not set, using local fallback judges for all samples.")

	results, summary, total_tokens = evaluate_dataset(
		api_key=api_key,
		judge_api_key=judge_api_key,
		planner_model=args.planner_model,
		synthesizer_model=args.synthesizer_model,
		judge_model=args.judge_model,
		batch_size=args.batch_size,
		pause_seconds=args.pause_seconds,
	)

	report = build_final_report(results, summary, total_tokens)

	print("\n")
	print("=" * 80)
	print("FINAL RESULTS")
	print("=" * 80)
	print(json.dumps(report, ensure_ascii=False, indent=2))

	output_path = Path(args.output)
	output_path.write_text(
		json.dumps(
			{
				"summary": report,
				"results": results,
			},
			ensure_ascii=False,
			indent=2,
		),
		encoding="utf-8",
	)

	print(f"\nSaved detailed results to {output_path}")


if __name__ == "__main__":
	main()
