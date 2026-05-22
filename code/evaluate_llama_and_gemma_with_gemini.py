"""
evaluate_llama_and_gemma_with_gemini.py — Gemini evaluation for Llama and Gemma models.

  - Token-overlap F1  (lexical)
  - Gemini-as-judge   (semantic score 0-100 + CORRECT/PARTIALLY_CORRECT/…)

Input:
    evaluation/results_llama.csv
    evaluation/results_gemma3.csv

Output:
    evaluation/gemini_evaluation_llama_results.csv   — per-question results
    evaluation/gemini_evaluation_gemma_results.csv   — per-question results
"""

import csv
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

try:
    from google.api_core import exceptions as google_exceptions
except ImportError:
    google_exceptions = None

load_dotenv()


# ── Config ────────────────────────────────────────────────────────────────────

LLAMA_INPUT   = "evaluation/results_llama.csv"
GEMMA_INPUT   = "evaluation/results_gemma3.csv"
LLAMA_OUTPUT  = "evaluation/gemini_evaluation_llama_results.csv"
GEMMA_OUTPUT  = "evaluation/gemini_evaluation_gemma_results.csv"
GEMINI_MODEL  = "gemini-2.5-pro"

EVAL_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert evaluator for a University chatbot QA system.

Compare the model answer with the expected answer and respond in this EXACT format:
SCORE: <0-100>
ASSESSMENT: <CORRECT/PARTIALLY_CORRECT/INCORRECT/HALLUCINATED>
EXPLANATION: <1-2 sentences>

Guidelines:
- CORRECT (90-100): answer contains expected information or a valid equivalent
- PARTIALLY_CORRECT (50-89): some correct info but missing key details
- INCORRECT (20-49): on-topic but wrong or significantly incomplete
- HALLUCINATED (0-19): misleading, contradicts expected answer, or off-topic

Question: {question}
Expected Answer: {expected}
Model Answer: {model_answer}"""
)


# ── Metrics ───────────────────────────────────────────────────────────────────

def token_overlap_f1(prediction: str, reference: str) -> float:
    pred_tokens = set(prediction.lower().split())
    ref_tokens  = set(reference.lower().split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def contains_key_info(prediction: str, reference: str) -> bool:
    ref_words  = {w.lower().strip(".,;:()[]") for w in reference.split() if len(w) > 4}
    pred_lower = prediction.lower()
    matches    = sum(1 for w in ref_words if w in pred_lower)
    return matches >= max(1, len(ref_words) // 4)


# ── Gemini helpers ────────────────────────────────────────────────────────────

def is_rate_limit_error(exc: Exception) -> bool:
    name    = type(exc).__name__
    message = str(exc).lower()
    if google_exceptions is not None:
        limit_types = (
            getattr(google_exceptions, "ResourceExhausted", ()),
            getattr(google_exceptions, "TooManyRequests", ()),
        )
        if isinstance(exc, limit_types):
            return True
    return name in {"ResourceExhausted", "TooManyRequests"} or "quota" in message or "rate limit" in message


def make_llm(model_name: str) -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY not set in environment or .env file.")
        sys.exit(1)
    return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.2)


def parse_eval_response(text: str) -> dict:
    result = {"score": 0, "assessment": "UNKNOWN", "explanation": ""}
    for line in text.strip().splitlines():
        if line.startswith("SCORE:"):
            try:
                result["score"] = int(line.replace("SCORE:", "").strip())
            except ValueError:
                pass
        elif line.startswith("ASSESSMENT:"):
            result["assessment"] = line.replace("ASSESSMENT:", "").strip()
        elif line.startswith("EXPLANATION:"):
            result["explanation"] = line.replace("EXPLANATION:", "").strip()
    return result


# ── CSV loader ───────────────────────────────────────────────────────────────

def load_csv_pairs(csv_file: str, prediction_col: str, limit: int | None = None) -> list[dict]:
    """Read QA pairs from a CSV file produced by a model evaluation run."""
    path = Path(csv_file)
    if not path.exists():
        print(f"[ERROR] CSV file not found: {csv_file}")
        sys.exit(1)
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "id":         row.get("id", ""),
                "question":   row["question"],
                "expected":   row["expected"],
                "prediction": row[prediction_col],
            })
    if limit is not None:
        rows = rows[:limit]
    return rows


# ── Core logic ────────────────────────────────────────────────────────────────

def judge(question: str, expected: str, prediction: str, eval_chain) -> dict:
    """Use Gemini to semantically score the prediction against the expected answer."""
    raw = eval_chain.invoke({
        "question":     question,
        "expected":     expected,
        "model_answer": prediction,
    })
    return parse_eval_response(raw)


# ── Output helpers ────────────────────────────────────────────────────────────

def save_csv(results: list, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[INFO] Results saved → '{path}'")


def print_summary(results: list, f1_scores: list, gemini_scores: list, model_name: str, output: str) -> None:
    n             = len(results)
    avg_f1        = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    avg_gemini    = sum(gemini_scores) / len(gemini_scores) if gemini_scores else 0.0
    key_info_rate = sum(1 for r in results if r["has_key_info"]) / n if n else 0.0

    assessments  = [r["gemini_assessment"] for r in results]
    correct      = sum(1 for a in assessments if a == "CORRECT")
    partial      = sum(1 for a in assessments if a == "PARTIALLY_CORRECT")
    incorrect    = sum(1 for a in assessments if a == "INCORRECT")
    hallucinated = sum(1 for a in assessments if a == "HALLUCINATED")
    pct          = lambda x: f"{x * 100 // n:2d}" if n else " 0"

    summary = (
        f"\n{'='*65}\n"
        f"  GEMINI BASELINE EVALUATION SUMMARY\n"
        f"{'='*65}\n"
        f"  Model              : {model_name}\n"
        f"  Questions          : {n}\n"
        f"  Avg F1 score       : {avg_f1:.4f}  ({avg_f1*100:.1f}%)\n"
        f"  Avg Gemini score   : {avg_gemini:.1f}/100\n"
        f"  Key info hit       : {key_info_rate*100:.1f}%\n"
        f"\n"
        f"  Assessment breakdown:\n"
        f"    CORRECT           : {correct:3d} ({pct(correct)}%)\n"
        f"    PARTIALLY_CORRECT : {partial:3d} ({pct(partial)}%)\n"
        f"    INCORRECT         : {incorrect:3d} ({pct(incorrect)}%)\n"
        f"    HALLUCINATED      : {hallucinated:3d} ({pct(hallucinated)}%)\n"
        f"{'='*65}\n"
    )
    print(summary)

    summary_path = Path(output).parent / "gemini_evaluation_baseline_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"[INFO] Summary saved → '{summary_path}'")


# ── Main evaluation loop ──────────────────────────────────────────────────────

def evaluate(csv_file: str, prediction_col: str, output: str,
             limit: int | None, model_label: str) -> None:
    qa_pairs = load_csv_pairs(csv_file, prediction_col, limit)

    print(f"\n{'='*65}")
    print(f"  Gemini Judge Evaluation — {model_label}")
    print(f"  Questions : {len(qa_pairs)}  |  source: '{csv_file}'")
    print(f"{'='*65}\n")

    eval_chain = EVAL_PROMPT | make_llm(GEMINI_MODEL) | StrOutputParser()

    results       = []
    f1_scores     = []
    gemini_scores = []

    print(f"{'='*65}")
    print(f"  Running ({len(qa_pairs)} questions)...")
    print(f"{'='*65}\n")

    for i, item in enumerate(qa_pairs, 1):
        qid        = item.get("id", f"Q{i}")
        question   = item["question"]
        expected   = item["expected"]
        prediction = item["prediction"]

        print(f"[{i:3}/{len(qa_pairs)}] {qid}: {question[:65]}...")

        # Step 1 — Lexical metrics
        t0      = time.time()
        f1      = token_overlap_f1(prediction, expected)
        has_key = contains_key_info(prediction, expected)
        f1_scores.append(f1)

        # Step 2 — Gemini-as-judge
        gemini_score, gemini_assessment, gemini_explanation = 0, "N/A", ""
        if prediction and not prediction.startswith("ERROR:"):
            try:
                scored             = judge(question, expected, prediction, eval_chain)
                gemini_score       = scored["score"]
                gemini_assessment  = scored["assessment"]
                gemini_explanation = scored["explanation"]
                gemini_scores.append(gemini_score)
            except Exception as exc:
                if is_rate_limit_error(exc):
                    print(f"\n[CRITICAL] Rate limit during judge call — saving partial results.")
                    results.append(_row(qid, question, expected, prediction, f1, has_key,
                                        gemini_score, gemini_assessment, gemini_explanation,
                                        round(time.time() - t0, 2)))
                    save_csv(results, output)
                    sys.exit(1)
                gemini_assessment = f"EVAL_ERROR: {exc}"

        elapsed = time.time() - t0
        print(f"        F1={f1:.2f}  score={gemini_score}/100  [{gemini_assessment}]  ({elapsed:.1f}s)")

        results.append(_row(qid, question, expected, prediction, f1, has_key,
                            gemini_score, gemini_assessment, gemini_explanation, elapsed))

        if i < len(qa_pairs):
            time.sleep(0.4)

    save_csv(results, output)
    print_summary(results, f1_scores, gemini_scores, model_label, output)


def _row(qid, question, expected, prediction, f1, has_key,
         gemini_score, gemini_assessment, gemini_explanation, elapsed) -> dict:
    return {
        "id":                 qid,
        "question":           question,
        "expected":           expected,
        "model_prediction":   prediction,
        "f1":                 round(f1, 4),
        "has_key_info":       has_key,
        "gemini_score":       gemini_score,
        "gemini_assessment":  gemini_assessment,
        "gemini_explanation": gemini_explanation,
        "time_s":             round(elapsed, 2),
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("[INFO] Evaluating Llama predictions (first 115 questions)...")
    evaluate(
        csv_file=LLAMA_INPUT,
        prediction_col="prediction",
        output=LLAMA_OUTPUT,
        limit=LLAMA_LIMIT,
        model_label="Llama",
    )

    print("[INFO] Evaluating Gemma3 predictions (all questions)...")
    evaluate(
        csv_file=GEMMA_INPUT,
        prediction_col="prediction_gemma",
        output=GEMMA_OUTPUT,
        limit=None,
        model_label="Gemma3",
    )


if __name__ == "__main__":
    main()
