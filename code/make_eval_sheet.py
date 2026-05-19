"""
make_human_eval_sheet.py — Generate a human evaluation spreadsheet.

Merges test_questions.json with model outputs from:
  - evaluation/results.csv          (Llama 3.1, column: prediction)
  - evaluation/results_gemma3.csv   (Gemma 3, column: prediction_gemma)

Output:
  evaluation/human_eval_sheet.csv   — open in Google Sheets or Excel

Usage:
    python code/make_human_eval_sheet.py
"""

import csv
import json
from pathlib import Path

QA_FILE    = "evaluation/test_questions.json"
LLAMA_CSV  = "evaluation/results.csv"
GEMMA_CSV  = "evaluation/results_gemma3.csv"
OUTPUT     = "evaluation/human_eval_sheet.csv"


def load_answers(path: str, answer_col: str) -> dict:
    """Load {question_id: answer} from a results CSV. Tries multiple column names."""
    answers = {}
    fallbacks = [answer_col, "prediction", "prediction_gemma", "gemini_answer"]
    p = Path(path)
    if not p.exists():
        print(f"  WARNING: {path} not found — answers will be blank for this model")
        return answers
    with open(p, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        col = next((c for c in fallbacks if c in reader.fieldnames), None)
        if col is None:
            print(f"  WARNING: No answer column found in {path}. Columns: {reader.fieldnames}")
            return answers
        f.seek(0)
        reader = csv.DictReader(f)
        for row in reader:
            answers[row["id"]] = row.get(col, "").strip()
    print(f"  Loaded {len(answers)} answers from '{path}' (column: '{col}')")
    return answers


def main():
    with open(QA_FILE, encoding="utf-8") as f:
        qa_pairs = json.load(f)
    print(f"Loaded {len(qa_pairs)} QA pairs from '{QA_FILE}'")

    llama_answers = load_answers(LLAMA_CSV,  "prediction")
    gemma_answers = load_answers(GEMMA_CSV,  "prediction_gemma")

    covered_llama = sum(1 for qa in qa_pairs if qa["id"] in llama_answers)
    covered_gemma = sum(1 for qa in qa_pairs if qa["id"] in gemma_answers)
    print(f"\n  Llama answers available: {covered_llama}/{len(qa_pairs)}")
    print(f"  Gemma answers available: {covered_gemma}/{len(qa_pairs)}")
    if covered_llama < len(qa_pairs) or covered_gemma < len(qa_pairs):
        missing_llama = len(qa_pairs) - covered_llama
        missing_gemma = len(qa_pairs) - covered_gemma
        if missing_llama:
            print(f"  ⚠  {missing_llama} Llama answers missing — run evaluate.py first")
        if missing_gemma:
            print(f"  ⚠  {missing_gemma} Gemma answers missing — run evaluate.py with --output results_gemma3.csv first")

    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ID",
            "Question",
            "Expected Answer",
            "Llama 3.1 — Answer",
            "Llama — Score (3/2/1/0)",
            "Llama — Label",
            "Llama — Notes",
            "Gemma 3 — Answer",
            "Gemma — Score (3/2/1/0)",
            "Gemma — Label",
            "Gemma — Notes",
        ])
        for qa in qa_pairs:
            qid = qa["id"]
            writer.writerow([
                qid,
                qa["question"],
                qa["answer"],
                llama_answers.get(qid, ""),
                "",  # reviewer fills score
                "",  # reviewer fills label
                "",  # optional note
                gemma_answers.get(qid, ""),
                "",
                "",
                "",
            ])

    print(f"\n  Saved → '{OUTPUT}'  ({len(qa_pairs)} rows)")
    print()
    print("  Scoring guide for your reviewer:")
    print("    3 — CORRECT      : answer matches expected, even if phrased differently")
    print("    2 — PARTIAL      : some correct info but missing key details")
    print("    1 — INCORRECT    : wrong, off-topic, or significantly incomplete")
    print("    0 — HALLUCINATED : contradicts expected or invents false facts")
    print()

if __name__ == "__main__":
    main()