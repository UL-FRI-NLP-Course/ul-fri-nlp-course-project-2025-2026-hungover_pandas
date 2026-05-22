"""
evaluate_llama_and_gemma_with_bert_score.py — BERTScore evaluation for Llama and Gemma models.

Input:
    evaluation/results_llama.csv
    evaluation/results_llama_tuned.csv
    evaluation/results_gemma3.csv
    evaluation/results_gemma3_tuned.csv

Output (per model):
    evaluation/llama_not_tuned_results_with_bert_score.csv
    evaluation/llama_tuned_results_with_bert_score.csv
    evaluation/gemma_not_tuned_results_with_bert_score.csv
    evaluation/gemma_tuned_results_with_bert_score.csv
    evaluation/llama_and_gemma_bert_score_summary.txt
"""

import csv
import sys
from pathlib import Path

from bert_score import score as bert_score_fn


# ── Config ────────────────────────────────────────────────────────────────────

BERT_MODEL = "microsoft/deberta-v3-xsmall"   # fast; swap for "roberta-large" for higher accuracy
DEVICE     = None                         # None = auto (GPU if available, else CPU)

MODELS = [
    {
        "label":          "Llama (not tuned)",
        "input":          "evaluation/no tune results/llama_results.csv",
        "output":         f"evaluation/llama_not_tuned_results_with_bert_score_{BERT_MODEL.replace('/', '_')}.csv",
        "prediction_col": "prediction",
    },
    {
        "label":          "Llama (tuned)",
        "input":          "evaluation/tuned results/results_llama_tuned.csv",
        "output":         f"evaluation/llama_tuned_results_with_bert_score_{BERT_MODEL.replace('/', '_')}.csv",
        "prediction_col": "prediction",
    },
    {
        "label":          "Gemma3 (not tuned)",
        "input":          "evaluation/no tune results/results_gemma3.csv",
        "output":         f"evaluation/gemma_not_tuned_results_with_bert_score_{BERT_MODEL.replace('/', '_')}.csv",
        "prediction_col": "prediction",
    },
    {
        "label":          "Gemma3 (tuned)",
        "input":          "evaluation/tuned results/results_gemma3_tuned.csv",
        "output":         f"evaluation/gemma_tuned_results_with_bert_score_{BERT_MODEL.replace('/', '_')}.csv",
        "prediction_col": "prediction",
    },
]


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_csv(path: str, prediction_col: str, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if limit:
        rows = rows[:limit]
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

def evaluate(input_csv: str, output_csv: str, prediction_col: str,
             model_label: str, bert_model: str,
             limit: int | None = None) -> str:
    path = Path(input_csv)
    if not path.exists():
        print(f"[ERROR] CSV file not found: {input_csv}")
        sys.exit(1)

    rows = load_csv(input_csv, prediction_col, limit)

    print(f"\n{'='*65}")
    print(f"  BERTScore Evaluation — {model_label}")
    print(f"  BERTScore model : {bert_model}")
    print(f"  Rows    : {len(rows)}  |  source: '{input_csv}'")
    print(f"{'='*65}\n")

    references  = [r["expected"]        for r in rows]
    hypotheses  = [r[prediction_col]    for r in rows]

    print("[INFO] Running BERTScore (this may take a moment)...")

    P, R, F1 = bert_score_fn(
        cands=hypotheses,
        refs=references,
        model_type=bert_model,
        device=DEVICE,
        verbose=True,
    )

    results = []
    for i, row in enumerate(rows):
        results.append({
            "id":               row.get("id", ""),
            "question":         row["question"],
            "expected":         row["expected"],
            "model_prediction": row[prediction_col],
            "bert_P":           round(P[i].item(), 4),
            "bert_R":           round(R[i].item(), 4),
            "bert_F1":          round(F1[i].item(), 4),
        })

    save_csv(results, output_csv)

    n      = len(results)
    avg_p  = sum(r["bert_P"]  for r in results) / n
    avg_r  = sum(r["bert_R"]  for r in results) / n
    avg_f1 = sum(r["bert_F1"] for r in results) / n

    summary_block = (
        f"\n{'='*65}\n"
        f"  BERT SCORE EVALUATION SUMMARY — {model_label}\n"
        f"{'='*65}\n"
        f"  BERTScore model    : {bert_model}\n"
        f"  Questions          : {n}\n"
        f"  Avg Precision      : {avg_p:.4f}  ({avg_p*100:.1f}%)\n"
        f"  Avg Recall         : {avg_r:.4f}  ({avg_r*100:.1f}%)\n"
        f"  Avg F1             : {avg_f1:.4f}  ({avg_f1*100:.1f}%)\n"
        f"{'='*65}\n"
    )
    print(summary_block)
    return summary_block


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    all_summaries = []
    for cfg in MODELS:
        block = evaluate(
            input_csv=cfg["input"],
            output_csv=cfg["output"],
            prediction_col=cfg["prediction_col"],
            model_label=cfg["label"],
            bert_model=BERT_MODEL,
        )
        all_summaries.append(block)

    combined_summary = "".join(all_summaries)
    summary_path = f"evaluation/llama_and_gemma_bert_score_{BERT_MODEL.replace('/', '_')}_summary.txt"
    save_summary(combined_summary, summary_path)


if __name__ == "__main__":
    main()
