"""
Evaluate QA results using Gemini API for semantic comparison.

Reads results.jsonl and uses Gemini to assess if the model answer matches
the expected answer. Outputs detailed evaluation with scores.
"""

import json
import csv
import os
import argparse
import sys
import time
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

try:
    from google.api_core import exceptions as google_exceptions
except ImportError:
    google_exceptions = None

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def setup_gemini():
    """Initialize Gemini LLM."""
    if not GEMINI_API_KEY:
        print("[ERROR] GEMINI_API_KEY not set. Please set environment variable or add to .env")
        sys.exit(1)
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=GEMINI_API_KEY,
        temperature=0.3,
    )
    return llm


def create_evaluation_prompt():
    """Create prompt template for answer evaluation."""
    prompt = ChatPromptTemplate.from_template("""You are an expert evaluator for a University chatbot QA system.

Your task: Compare the model's answer with the expected answer and provide:
1. A SCORE (0-100): How well the model answer matches the expected answer
2. An ASSESSMENT (CORRECT/PARTIALLY_CORRECT/INCORRECT/HALLUCINATED)
3. A BRIEF EXPLANATION (1-2 sentences)

Guidelines:
- CORRECT (90-100): Model answer contains the expected answer or equivalent information
- PARTIALLY_CORRECT (50-89): Model answer contains some correct information but missing key details
- INCORRECT (20-49): Model answer is on topic but wrong or incomplete
- HALLUCINATED (0-19): Model answer is misleading, contradicts expected answer, or completely off-topic

Question: {question}

Expected Answer: {expected}

Model Answer: {model_answer}

Respond in this exact format:
SCORE: <0-100>
ASSESSMENT: <CORRECT/PARTIALLY_CORRECT/INCORRECT/HALLUCINATED>
EXPLANATION: <1-2 sentences>""")
    
    return prompt


def parse_gemini_response(response_text: str) -> dict:
    """Parse Gemini's structured response."""
    lines = response_text.strip().split('\n')
    result = {
        "score": 0,
        "assessment": "UNKNOWN",
        "explanation": ""
    }
    
    for line in lines:
        if line.startswith("SCORE:"):
            try:
                result["score"] = int(line.replace("SCORE:", "").strip())
            except:
                pass
        elif line.startswith("ASSESSMENT:"):
            result["assessment"] = line.replace("ASSESSMENT:", "").strip()
        elif line.startswith("EXPLANATION:"):
            result["explanation"] = line.replace("EXPLANATION:", "").strip()
    
    return result


def load_results(jsonl_path: str) -> list:
    """Load results from JSONL file."""
    results = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def parse_range(range_str: str, total_questions: int) -> list:
    """Parse range string and return list of 1-based question indices.
    
    Formats:
    - '1-5': questions 1 to 5 (inclusive)
    - '3,5,7': specific questions
    - 'all': all questions
    - None: all questions
    """
    if not range_str or range_str.lower() == 'all':
        return list(range(1, total_questions + 1))
    
    indices = []
    parts = range_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            # Range format: '1-5'
            try:
                start, end = part.split('-')
                start = int(start.strip())
                end = int(end.strip())
                if start < 1 or end > total_questions or start > end:
                    print(f"[ERROR] Invalid range: {part}. Must be 1-based and within 1-{total_questions}")
                    sys.exit(1)
                indices.extend(range(start, end + 1))
            except ValueError:
                print(f"[ERROR] Invalid range format: {part}")
                sys.exit(1)
        else:
            # Single index
            try:
                idx = int(part)
                if idx < 1 or idx > total_questions:
                    print(f"[ERROR] Index {idx} out of range. Must be between 1 and {total_questions}")
                    sys.exit(1)
                indices.append(idx)
            except ValueError:
                print(f"[ERROR] Invalid index: {part}")
                sys.exit(1)
    
    # Remove duplicates and sort
    return sorted(list(set(indices)))


def is_gemini_limit_error(error: Exception) -> bool:
    """Return True when the Gemini API reports a quota or rate-limit failure."""
    error_name = type(error).__name__
    message = str(error).lower()

    if google_exceptions is not None:
        limit_exceptions = (
            getattr(google_exceptions, "ResourceExhausted", tuple()),
            getattr(google_exceptions, "TooManyRequests", tuple()),
        )
        if isinstance(error, limit_exceptions):
            return True

    return error_name in {"ResourceExhausted", "TooManyRequests"} or "quota" in message or "rate limit" in message


def write_evaluations(evaluations: list, output_jsonl: str, output_csv: str, append_mode: bool = True):
    """Write evaluation results to JSONL (append mode) and CSV files.
    
    Args:
        evaluations: List of evaluation results
        output_jsonl: Path to output JSONL file
        output_csv: Path to output CSV file
        append_mode: If True, append to JSONL file; if False, overwrite
    """
    # Write JSONL (append mode)
    mode = 'a' if append_mode else 'w'
    print(f"\n[INFO] {'Appending to' if append_mode else 'Writing to'} JSONL: {output_jsonl}")
    with open(output_jsonl, mode, encoding='utf-8') as f:
        for e in evaluations:
            f.write(json.dumps(e) + '\n')
    
    # Regenerate CSV from all JSONL entries
    print(f"[INFO] Regenerating CSV from all results: {output_csv}")
    all_results = []
    if os.path.exists(output_jsonl):
        with open(output_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    all_results.append(json.loads(line))
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Q#", "Question", "Expected Answer", "Model Answer",
            "Gemini Score", "Assessment", "Explanation", "Retrieved Sources"
        ])
        for e in all_results:
            sources_str = "; ".join([s["source"] for s in e.get("retrieved_sources", [])])
            writer.writerow([
                e.get("q_num", ""),
                e.get("question", "")[:80],
                e.get("expected_answer", "")[:80],
                e.get("model_answer", "")[:80],
                e.get("gemini_score", "N/A"),
                e.get("gemini_assessment", ""),
                e.get("gemini_explanation", "")[:100],
                sources_str
            ])


def evaluate_with_gemini(results: list, llm, prompt_template, output_jsonl: str, output_csv: str, range_indices: list = None):
    """Evaluate results using Gemini and save outputs.
    
    Args:
        results: List of all results
        llm: Gemini LLM instance
        prompt_template: Evaluation prompt template
        output_jsonl: Output JSONL file path
        output_csv: Output CSV file path
        range_indices: List of 1-based indices to evaluate; None means all
    """
    
    chain = prompt_template | llm | StrOutputParser()
    
    # Filter results based on range if specified
    if range_indices is not None:
        filtered_results = [results[i-1] for i in range_indices if i <= len(results)]
        total_results = len(results)
        evaluated_count = len(filtered_results)
        range_str = f"{min(range_indices)}-{max(range_indices)}" if len(range_indices) > 0 else "none"
        print(f"\n[INFO] Evaluating questions {range_str} ({evaluated_count} of {total_results} total)...\n")
    else:
        filtered_results = results
        evaluated_count = len(results)
        print(f"\n[INFO] Evaluating {evaluated_count} questions with Gemini...\n")
    
    evaluations = []
    
    for idx, result in enumerate(filtered_results, start=1):
        q_num = result.get("q_num", "")
        question = result.get("question", "")
        expected = result.get("expected_answer", "")
        model_answer = result.get("model_answer", "")
        
        try:
            # Call Gemini for evaluation
            response = chain.invoke({
                "question": question,
                "expected": expected,
                "model_answer": model_answer
            })
            
            # Parse response
            evaluation = parse_gemini_response(response)
            
            # Combine with original result
            combined = {
                **result,
                "gemini_score": evaluation["score"],
                "gemini_assessment": evaluation["assessment"],
                "gemini_explanation": evaluation["explanation"],
                "gemini_raw_response": response
            }
            evaluations.append(combined)
            
            print(f"[{idx}/{evaluated_count}] {q_num}: {evaluation['assessment']} ({evaluation['score']}/100)")
            
        except Exception as e:
            if is_gemini_limit_error(e):
                print(f"\n[CRITICAL] Gemini API limit exceeded at question [{idx}/{evaluated_count}] {q_num}")
                print(f"  {type(e).__name__}: {e}")
                print("[INFO] Saving partial results before terminating...")
                write_evaluations(evaluations, output_jsonl, output_csv, append_mode=True)
                sys.exit(1)

            print(f"[{idx}/{evaluated_count}] {q_num}: ERROR")
            print(f"  {type(e).__name__}: {e}")
            combined = {
                **result,
                "gemini_score": 0,
                "gemini_assessment": "ERROR",
                "gemini_explanation": str(e),
                "error": True
            }
            evaluations.append(combined)
        
        # Add delay to avoid rate limiting
        if idx < evaluated_count:
            time.sleep(0.5)
    
    write_evaluations(evaluations, output_jsonl, output_csv, append_mode=True)
    
    # Compute statistics
    scores = [e.get("gemini_score", 0) for e in evaluations if "error" not in e]
    assessments = [e.get("gemini_assessment", "") for e in evaluations if "error" not in e]
    
    correct = sum(1 for a in assessments if a == "CORRECT")
    partial = sum(1 for a in assessments if a == "PARTIALLY_CORRECT")
    incorrect = sum(1 for a in assessments if a == "INCORRECT")
    hallucinated = sum(1 for a in assessments if a == "HALLUCINATED")
    errors = sum(1 for e in evaluations if "error" in e)
    
    avg_score = sum(scores) / len(scores) if scores else 0
    
    # Summary
    print(f"\n{'='*70}")
    print(f"GEMINI EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total questions: {len(evaluations)}")
    print(f"Average score: {avg_score:.1f}/100")
    print(f"\nAssessment breakdown:")
    print(f"  CORRECT:             {correct:2d} ({correct*100//len(evaluations):2d}%)")
    print(f"  PARTIALLY_CORRECT:   {partial:2d} ({partial*100//len(evaluations):2d}%)")
    print(f"  INCORRECT:           {incorrect:2d} ({incorrect*100//len(evaluations):2d}%)")
    print(f"  HALLUCINATED:        {hallucinated:2d} ({hallucinated*100//len(evaluations):2d}%)")
    print(f"  ERRORS:              {errors:2d}")
    print(f"\nResults saved to:")
    print(f"  - {output_jsonl}")
    print(f"  - {output_csv}")
    
    return evaluations


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate QA results using Gemini API"
    )
    parser.add_argument(
        "--input",
        default="tests_Q&A/results.jsonl",
        help="Input JSONL file from eval_sample_questions.py (default: tests_Q&A/results.jsonl)"
    )
    parser.add_argument(
        "--output-jsonl",
        default="tests_Q&A/analysis_gemini_results.jsonl",
        help="Output JSONL with Gemini evaluations (default: tests_Q&A/analysis_gemini_results.jsonl)"
    )
    parser.add_argument(
        "--output-csv",
        default="tests_Q&A/analysis_gemini_results.csv",
        help="Output CSV with Gemini evaluations (default: tests_Q&A/analysis_gemini_results.csv)"
    )
    parser.add_argument(
        "--range",
        default=None,
        help="Question range to run (e.g. '1-5', '3,5,7', or 'all'). 1-based indices."
    )
    
    args = parser.parse_args()
    
    # Check input file
    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        print(f"[INFO] Run 'python eval_sample_questions.py' first to generate results.jsonl")
        sys.exit(1)
    
    # Setup Gemini
    print("[INFO] Setting up Gemini...")
    llm = setup_gemini()
    
    # Create prompt template
    prompt_template = create_evaluation_prompt()
    
    # Load results
    print(f"[INFO] Loading {args.input}")
    results = load_results(args.input)
    print(f"[INFO] Loaded {len(results)} results")
    
    # Parse range if specified
    range_indices = None
    if args.range:
        range_indices = parse_range(args.range, len(results))
        print(f"[INFO] Parsed range: {args.range} -> {len(range_indices)} questions")
    
    # Evaluate
    evaluations = evaluate_with_gemini(
        results,
        llm,
        prompt_template,
        args.output_jsonl,
        args.output_csv,
        range_indices=range_indices
    )


if __name__ == "__main__":
    main()
