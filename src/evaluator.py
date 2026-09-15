"""Handle execution and scoring.

This file will focus on Exact Match scoring for the `category`. LLM-as-a-judge scoring can be added in later for summaries, however, first priority is binary accuracy becuase it is the most critical metric for routing.

Architecture/Flow:
1. We wrote our own classifier function.
2. We run each of our golden data set test cases through the classifier function.
3. We check to see if our classify email function returns the right classification based on the ground truth of the test case.
"""

import json
import asyncio
import time
from pathlib import Path
from typing import List, Dict

import os
from openai import AsyncOpenAI, APIStatusError
from src.classifier import classify_email_async
from src.models import ScoredSummaryRelevance

# client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client = AsyncOpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")

MAX_RETRIES = 5
# Keep concurrency modest so we don't get rate-limited fighting ourselves.
MAX_CONCURRENT_REQUESTS = 5

def load_dataset(filepath: str) -> List[Dict]:
    with open(filepath, 'r') as f:
        return json.load(f)

async def score_summary_relevance(expected_summary: str, scored_summary: str) -> int:
    """LLM-as-a-judge: Rates summary relevance from 1 to 5."""
    prompt = f"""
    Compare the scored summary to the expected ground-truth summary.
    Expected: {expected_summary}
    Scored: {scored_summary}

    Rate the semantic similarity and factual accuracy of the Scored summary on a scale of 1 to 5.
    5 = Perfect match in meaning, 1 = Completely irrelevant or contradictory.
    Output ONLY the integer (1, 2, 3, 4, or 5).
    """

    try:
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.beta.chat.completions.parse(
                    # model="gpt-4o-mini",
                    model="openai/gpt-oss-20b",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    response_format=ScoredSummaryRelevance
                )
            except APIStatusError as e:
                if attempt == MAX_RETRIES - 1 or e.status_code not in (429, 500, 502, 503):
                    raise
                await asyncio.sleep(2 ** attempt)
                continue
            return int(response.choices[0].message.parsed.relevance_score)
    except Exception as e:
        return 0
        # return e

async def evaluate_single_case(case: dict, prompt_path: str, semaphore: asyncio.Semaphore) -> dict:
    """Run a single test case through the classifier & LLM judge."""
    async with semaphore:
        return await _evaluate_single_case(case, prompt_path)

async def _evaluate_single_case(case: dict, prompt_path: str) -> dict:
    # 1. Classify the email
    response = await classify_email_async(case['email_text'], prompt_path)
    scored_category = response["output"].category
    scored_summary = response["output"].summary

    # 2. LLM-as-judge for Summary Relevance Score
    summary_relevance_score = await score_summary_relevance(case['expected_summary'], scored_summary)

    # 3. Now we have the scored category + summary + summary relevance score
    is_category_passed = scored_category == case['expected_category']
    result = {
        "id": case["id"],
        "difficulty": case["difficulty"],
        "expected_category": case["expected_category"],
        "scored_category": scored_category,
        "is_category_passed": is_category_passed,
        "expected_summary": case["expected_summary"],
        "scored_summary": scored_summary,
        "summary_score_1_to_5": summary_relevance_score,
        "latency_seconds": round(response["latency_seconds"], 2),
        "total_tokens": response["total_tokens"]
    }

    return result



async def run_async_evaluation(golden_dataset_path: str, prompt_path: str, output_path: str):
    dataset = load_dataset(golden_dataset_path)
    total_cases = len(dataset)
    print(f"Starting async evaluation for {total_cases} cases using {prompt_path}...")
    start_time = time.time()

    # Run all cases concurrently, bounded so we don't overwhelm the free-tier model
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = [evaluate_single_case(case, prompt_path, semaphore) for case in dataset]
    detailed_results = await asyncio.gather(*tasks)


    execution_time = time.time() - start_time

    # Aggregate metrics
    correct_categories = sum(1 for r in detailed_results if r["is_category_passed"])
    avg_summary_score = sum(r["summary_score_1_to_5"] for r in detailed_results) / len(detailed_results)
    avg_latency = sum(r["latency_seconds"] for r in detailed_results) / len(detailed_results)
    total_tokens = sum(r["total_tokens"] for r in detailed_results)
    accuracy = (correct_categories / len(dataset)) * 100

    print("\n" + "="*40)
    print("Async evaluation complete")
    print("="*40)
    print(f"Category Accuracy: {accuracy:.1f}% ({correct_categories}/{len(dataset)})")
    print(f"Avg Summary Score: {avg_summary_score:.2f} / 5.0")
    print(f"Avg Latency:       {avg_latency:.2f} seconds/req")
    print(f"Total Tokens:      {total_tokens}")
    print(f"Total Wall Time:   {execution_time:.2f} seconds (Async Speedup!)")

    run_snapshot = {
            "timestamp": time.time(),
            "prompt_file": prompt_path,
            "metrics": {
                "accuracy": accuracy,
                "avg_summary_score": avg_summary_score,
                "avg_latency_seconds": avg_latency,
                "total_tokens": total_tokens
            },
            "execution_time_seconds": execution_time,
            "total_cases": total_cases,
            "correct_cases": correct_categories,
            "results": detailed_results
        }

    with open(output_path, "w") as f:
        json.dump(run_snapshot, f, indent=2)
    print(f"\n Results saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(run_async_evaluation(
            "data/golden_dataset.json",
            prompt_path="prompts/support_v1.yaml",
            output_path="data/eval_runs/async_run_sep15.json"
    ))
