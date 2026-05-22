"""
Evaluate the Gemini baseline answers in evaluation/baseline_results.csv
using BERTScore, comparing the 'expected' (reference) column against the
'gemini_answer' (hypothesis) column.

Input:
    evaluation/baseline_results.csv   — columns: id, question, expected, gemini_answer

Output:
    evaluation/baseline_results_with_bert_score.csv          — per-question P/R/F1
    evaluation/baseline_results_with_bert_score_summary.txt  — aggregate stats
"""

import argparse
import csv
import sys
from pathlib import Path

from bert_score import score as bert_score_fn


# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_INPUT  = "evaluation/baseline_results.csv"
BERT_MODEL     = "microsoft/deberta-v3-xsmall"   # fast; swap for "roberta-large" for higher accuracy
DEFAULT_OUTPUT = f"evaluation/baseline_results_with_bert_score_{BERT_MODEL.replace('/', '_')}.csv"
DEVICE         = None                         # None = auto (GPU if available, else CPU)


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_baseline(path: str) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def save_csv(results: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"[INFO] Results saved  → '{path}'")


def save_summary(summary: str, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(summary, encoding="utf-8")
    print(f"[INFO] Summary saved  → '{path}'")


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(input_csv: str, output_csv: str, limit: int | None, model: str) -> None:
    rows = load_baseline(input_csv)
    if limit:
        rows = rows[:limit]

    print(f"\n{'='*65}")
    print(f"  BERTScore Baseline Evaluation")
    print(f"  Model   : {model}")
    print(f"  Rows    : {len(rows)}  |  source: '{input_csv}'")
    print(f"{'='*65}\n")

    references   = [r["expected"]      for r in rows]
    hypotheses   = [r["gemini_answer"] for r in rows]

    print("[INFO] Running BERTScore (this may take a moment)...")
    P, R, F1 = bert_score_fn(
        cands=hypotheses,
        refs=references,
        model_type=model,
        device=DEVICE,
        verbose=True,
        use_fast_tokenizer=False,
    )

    results = []
    for i, row in enumerate(rows):
        results.append({
            "id":            row["id"],
            "question":      row["question"],
            "expected":      row["expected"],
            "gemini_answer": row["gemini_answer"],
            "bert_P":        round(P[i].item(), 4),
            "bert_R":        round(R[i].item(), 4),
            "bert_F1":       round(F1[i].item(), 4),
        })

    save_csv(results, output_csv)

    n       = len(results)
    avg_p   = sum(r["bert_P"]  for r in results) / n
    avg_r   = sum(r["bert_R"]  for r in results) / n
    avg_f1  = sum(r["bert_F1"] for r in results) / n

    summary = (
        f"\n{'='*65}\n"
        f"  BERT SCORE BASELINE EVALUATION SUMMARY\n"
        f"{'='*65}\n"
        f"  BERTScore model    : {model}\n"
        f"  Questions          : {n}\n"
        f"  Avg Precision      : {avg_p:.4f}  ({avg_p*100:.1f}%)\n"
        f"  Avg Recall         : {avg_r:.4f}  ({avg_r*100:.1f}%)\n"
        f"  Avg F1             : {avg_f1:.4f}  ({avg_f1*100:.1f}%)\n"
        f"{'='*65}\n"
    )
    print(summary)

    summary_path = str(Path(output_csv).parent / f"baseline_results_with_bert_score_{BERT_MODEL.replace('/', '_')}_summary.txt")
    save_summary(summary, summary_path)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BERTScore evaluation for Erasmus QA baseline.")
    parser.add_argument("--input",  default=DEFAULT_INPUT,  help="Path to baseline_results.csv")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument("--model",  default=BERT_MODEL,     help="HuggingFace model for BERTScore")
    parser.add_argument("--limit",  type=int, default=None, help="Evaluate first N rows only")
    args = parser.parse_args()

    evaluate(input_csv=args.input, output_csv=args.output, limit=args.limit, model=args.model)


if __name__ == "__main__":
    main()
