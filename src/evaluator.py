"""Handle execution and scoring.

This file will focus on Exact Match scoring for the `category`. LLM-as-a-judge scoring can be added in later for summaries, however, first priority is binary accuracy becuase it is the most critical metric for routing.

Architecture/Flow:
1. We wrote our own classifier function. 
2. We run each of our golden data set test cases through the classifier function. 
3. We check to see if our classify email function returns the right classification based on the ground truth of the test case.
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Dict

from src.classifier import classify_email_async

def load_dataset(filepath: str) -> List[Dict]:
    with open(filepath, 'r') as f:
        return json.load(f)

def run_evaluation(golden_dataset_path: str, prompt_path: str, output_path: str):
    dataset = load_dataset(golden_dataset_path)
    total_cases = len(dataset)
    correct_categories = 0
    detailed_results = []
    
    print(f"Starting evaluation using {prompt_path} on {total_cases} cases...\n")
    start_time = time.time()
    
    for idx, case in enumerate(dataset):
        print(f"[{idx+1}/{total_cases}] Evaluating {case['id']}...", end=" ", flush=True)
        
        # 1. Generate prediction
        output = classify_email_async(case['email_text'], prompt_path)
        scored_summary = output.summary
        scored_category = output.category
        
        # 2. Score Category Match
        expected_category = case['expected_category']
        is_category_passed = (scored_category == expected_category)
        if is_category_passed:
            correct_categories += 1
            print("Pass!")
        else:
            print(f"FAIL (Expected: {expected_category}, Got: {scored_category})")
        
        # 3. Record details for this test case
        detailed_results.append({
            "id": case["id"],
            "difficulty": case["difficulty"],
            "expected_category": case["expected_category"],
            "scored_category": scored_category,
            "is_category_passed": is_category_passed,
            "expected_summary": case["expected_summary"],
            "scored_summary": scored_summary
        })
        
    execution_time = time.time() - start_time
    accuracy = (correct_categories / total_cases) * 100
    
    print("\n" + "="*30)
    print("Evaluation Complete")
    print("="*30)
    print(f"Accuracy: {accuracy:.1f}% ({correct_categories}/{total_cases})")
    print(f"Time:      {execution_time:.2f} seconds")
    
    run_snapshot = {
            "timestamp": time.time(),
            "prompt_file": prompt_path,
            "execution_time": execution_time,
            "accuracy": accuracy,
            "total_cases": total_cases,
            "correct_cases": correct_categories,
            "results": detailed_results
        }
    
    with open(output_path, "w") as f:
        json.dump(run_snapshot, f, indent=2)
    print(f"\n Results saved to {output_path}")
    
if __name__ == "__main__":
    run_evaluation(
        "data/golden_dataset.json",
        prompt_path="prompts/support_v1.yaml",
        output_path="data/eval_runs/latest_run.json"
    )