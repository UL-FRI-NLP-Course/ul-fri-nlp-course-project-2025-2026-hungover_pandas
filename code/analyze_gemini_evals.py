"""analyze_gemini_evals.py — Analysis of Gemini-as-judge evaluations across all models.

Focuses on gemini_assessment (CORRECT/PARTIALLY_CORRECT/INCORRECT/HALLUCINATED)
and gemini_score (0-100).  F1 is ignored.

Input:
    evaluation/all_gemini_evaluations/*.csv

Output (all in evaluation/gemini_evals_analysis/):
    per_model_summary.csv       — assessment counts, percentages, score stats per model
    tuning_delta.csv            — paired tuned vs untuned comparison (Llama & Gemma)
    hardest_questions.csv       — questions with lowest avg gemini_score across models
    report.md                   — full markdown report
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT     = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "evaluation" / "all_gemini_evaluations"
OUT_DIR  = ROOT / "evaluation" / "gemini_evals_analysis"

ASSESSMENT_ORDER  = ["CORRECT", "PARTIALLY_CORRECT", "INCORRECT", "HALLUCINATED"]
VALID_ASSESSMENTS = set(ASSESSMENT_ORDER)

FILES = {
    "Gemini baseline": "gemini_evaluation_baseline_results.csv",
    "Gemma not tuned": "gemini_evaluation_gemma_results.csv",
    "Gemma tuned":     "gemini_evaluation_gemma_tuned_results.csv",
    "Llama not tuned": "gemini_evaluation_llama_results.csv",
    "Llama tuned":     "gemini_evaluation_llama_tuned_results.csv",
}

MODEL_ORDER = [
    "Gemini baseline",
    "Llama tuned",
    "Gemma tuned",
    "Llama not tuned",
    "Gemma not tuned",
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for model_name, filename in FILES.items():
        path = EVAL_DIR / filename
        df   = pd.read_csv(path)
        df["model"]            = model_name
        df["valid_assessment"] = df["gemini_assessment"].isin(VALID_ASSESSMENTS)
        frames[model_name] = df
    return frames


# ── Per-model summary ─────────────────────────────────────────────────────────

def per_model_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        df      = frames[model]
        n_total = len(df)
        valid   = df[df["valid_assessment"]]
        n_valid = len(valid)
        scores  = valid["gemini_score"]

        row: dict = {
            "model":        model,
            "n_total":      n_total,
            "n_valid":      n_valid,
            "n_errors":     n_total - n_valid,
            "mean_score":   round(scores.mean(), 2),
            "median_score": round(scores.median(), 1),
            "std_score":    round(scores.std(), 2),
        }
        for a in ASSESSMENT_ORDER:
            count         = int((valid["gemini_assessment"] == a).sum())
            row[f"n_{a}"] = count
            row[f"pct_{a}"] = round(count / n_valid * 100, 1) if n_valid else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


# ── Assessment distribution pivot (% only) ───────────────────────────────────

def assessment_pivot(summary: pd.DataFrame) -> pd.DataFrame:
    cols = ["model"] + [f"pct_{a}" for a in ASSESSMENT_ORDER]
    pct  = summary[cols].copy()
    pct.columns = ["model"] + ASSESSMENT_ORDER
    return pct


# ── Tuning delta (paired by question ID) ─────────────────────────────────────

def tuning_delta(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pairs = [
        ("Gemma", "Gemma not tuned", "Gemma tuned"),
        ("Llama", "Llama not tuned", "Llama tuned"),
    ]
    rows = []
    for base, untuned_name, tuned_name in pairs:
        keep = ["id", "gemini_score", "gemini_assessment"]
        df_u = frames[untuned_name][keep][frames[untuned_name]["valid_assessment"]].copy()
        df_t = frames[tuned_name][keep][frames[tuned_name]["valid_assessment"]].copy()

        paired = df_u.merge(df_t, on="id", suffixes=("_untuned", "_tuned"))
        if paired.empty:
            continue

        delta = paired["gemini_score_tuned"] - paired["gemini_score_untuned"]

        row: dict = {
            "model_family":      base,
            "n_paired":          len(paired),
            "mean_score_untuned": round(paired["gemini_score_untuned"].mean(), 2),
            "mean_score_tuned":   round(paired["gemini_score_tuned"].mean(), 2),
            "mean_score_delta":   round(delta.mean(), 2),
            "median_score_delta": round(delta.median(), 1),
        }
        for a in ASSESSMENT_ORDER:
            n_u = int((paired["gemini_assessment_untuned"] == a).sum())
            n_t = int((paired["gemini_assessment_tuned"] == a).sum())
            row[f"pct_{a}_untuned"] = round(n_u / len(paired) * 100, 1)
            row[f"pct_{a}_tuned"]   = round(n_t / len(paired) * 100, 1)
            row[f"pct_{a}_delta"]   = round(row[f"pct_{a}_tuned"] - row[f"pct_{a}_untuned"], 1)
        rows.append(row)
    return pd.DataFrame(rows)


# ── Hardest questions (lowest avg gemini_score across models) ─────────────────

def hardest_questions(frames: dict[str, pd.DataFrame], n_bottom: int = 20) -> pd.DataFrame:
    dfs = []
    for model, df in frames.items():
        valid = df[df["valid_assessment"]][
            ["id", "question", "gemini_score", "gemini_assessment"]
        ].copy()
        valid["model"] = model
        dfs.append(valid)

    combined   = pd.concat(dfs, ignore_index=True)
    by_question = (
        combined.groupby(["id", "question"])
        .agg(
            n_models        =("gemini_score",      "count"),
            avg_score       =("gemini_score",      "mean"),
            min_score       =("gemini_score",      "min"),
            n_correct       =("gemini_assessment", lambda x: (x == "CORRECT").sum()),
            n_partial       =("gemini_assessment", lambda x: (x == "PARTIALLY_CORRECT").sum()),
            n_incorrect     =("gemini_assessment", lambda x: (x == "INCORRECT").sum()),
            n_hallucinated  =("gemini_assessment", lambda x: (x == "HALLUCINATED").sum()),
        )
        .reset_index()
        .assign(avg_score=lambda d: d["avg_score"].round(1))
        .sort_values("avg_score")
    )
    return by_question.head(n_bottom)


# ── Markdown helpers ──────────────────────────────────────────────────────────

def markdown_table(df: pd.DataFrame, float_fmt: str = ".1f") -> str:
    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda v: f"{v:{float_fmt}}")
    header    = "| " + " | ".join(map(str, formatted.columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(formatted.columns)) + " |"
    rows_md   = [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for row in formatted.to_numpy()
    ]
    return "\n".join([header, separator, *rows_md])


# ── Report builder ────────────────────────────────────────────────────────────

def build_report(
    summary:  pd.DataFrame,
    pivot:    pd.DataFrame,
    delta:    pd.DataFrame,
    hardest:  pd.DataFrame,
    frames:   dict[str, pd.DataFrame],
) -> str:
    n_files = len(FILES)
    lines   = [
        "# Gemini-as-Judge Evaluation Report",
        "",
        f"Evaluated **{n_files} models** using Gemini as an automated judge.",
        "Metric focus: `gemini_assessment` (CORRECT / PARTIALLY_CORRECT / INCORRECT / HALLUCINATED)"
        " and `gemini_score` (0–100).  Token-overlap F1 is excluded from this analysis.",
        "",
        "---",
        "",
        "## 1. Overall model ranking",
        "",
    ]

    ranking_cols = ["model", "n_valid", "mean_score", "median_score", "std_score",
                    "pct_CORRECT", "pct_PARTIALLY_CORRECT", "pct_INCORRECT", "pct_HALLUCINATED"]
    ranking = summary[ranking_cols].sort_values("mean_score", ascending=False)
    lines.append(markdown_table(ranking, float_fmt=".1f"))

    lines += [
        "",
        "---",
        "",
        "## 2. Assessment distribution (%)",
        "",
        markdown_table(pivot),
        "",
        "---",
        "",
        "## 3. Tuning impact (paired comparison on shared question IDs)",
        "",
    ]

    for _, row in delta.iterrows():
        fam   = row["model_family"]
        sign  = "+" if row["mean_score_delta"] >= 0 else ""
        lines += [
            f"### {fam}",
            f"Paired questions: **{int(row['n_paired'])}**",
            "",
            f"| Metric | Not tuned | Tuned | Delta |",
            f"| --- | --- | --- | --- |",
            f"| Mean score | {row['mean_score_untuned']:.1f} | {row['mean_score_tuned']:.1f} | {sign}{row['mean_score_delta']:.1f} |",
        ]
        for a in ASSESSMENT_ORDER:
            u   = row[f"pct_{a}_untuned"]
            t   = row[f"pct_{a}_tuned"]
            d   = row[f"pct_{a}_delta"]
            ds  = f"+{d:.1f}" if d >= 0 else f"{d:.1f}"
            lines.append(f"| {a} % | {u:.1f} | {t:.1f} | {ds} |")
        lines.append("")

    lines += [
        "---",
        "",
        f"## 4. Hardest questions (bottom 20 by avg Gemini score across all models)",
        "",
        markdown_table(
            hardest[["id", "question", "avg_score", "n_models",
                      "n_correct", "n_partial", "n_incorrect", "n_hallucinated"]],
            float_fmt=".1f",
        ),
        "",
        "---",
        "",
        "## 5. Key observations",
        "",
    ]

    best_model   = summary.sort_values("mean_score", ascending=False).iloc[0]
    worst_model  = summary.sort_values("mean_score").iloc[0]
    best_correct = summary.sort_values("pct_CORRECT", ascending=False).iloc[0]

    lines.append(
        f"- **Best overall model** by mean Gemini score: **{best_model['model']}**"
        f" with {best_model['mean_score']:.1f}/100 and {best_model['pct_CORRECT']:.1f}% CORRECT."
    )
    lines.append(
        f"- **Weakest model** by mean score: **{worst_model['model']}**"
        f" with {worst_model['mean_score']:.1f}/100."
    )
    if best_correct["model"] != best_model["model"]:
        lines.append(
            f"- Highest CORRECT rate: **{best_correct['model']}** ({best_correct['pct_CORRECT']:.1f}%)."
        )

    for _, row in delta.iterrows():
        fam  = row["model_family"]
        sign = "+" if row["mean_score_delta"] >= 0 else ""
        direction = "improved" if row["mean_score_delta"] >= 0 else "degraded"
        lines.append(
            f"- **{fam} tuning** {direction} mean score by"
            f" {sign}{row['mean_score_delta']:.1f} points"
            f" ({row['mean_score_untuned']:.1f} → {row['mean_score_tuned']:.1f})."
            f"  CORRECT rate: {row['pct_CORRECT_untuned']:.1f}% → {row['pct_CORRECT_tuned']:.1f}%"
            f" ({'+' if row['pct_CORRECT_delta'] >= 0 else ''}{row['pct_CORRECT_delta']:.1f} pp)."
        )

    hallucination_worst = summary.sort_values("pct_HALLUCINATED", ascending=False).iloc[0]
    lines.append(
        f"- Highest hallucination rate: **{hallucination_worst['model']}**"
        f" ({hallucination_worst['pct_HALLUCINATED']:.1f}% HALLUCINATED)."
    )

    hardest_q = hardest.iloc[0]
    lines.append(
        f"- Hardest question across all models: **\"{hardest_q['question'][:80]}...\"**"
        f" (avg score {hardest_q['avg_score']:.1f}/100,"
        f" {int(hardest_q['n_hallucinated'])} hallucinated out of {int(hardest_q['n_models'])} models)."
    )

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading CSV files...")
    frames = load_all()

    print("[INFO] Computing per-model summary...")
    summary = per_model_summary(frames)
    summary.to_csv(OUT_DIR / "per_model_summary.csv", index=False)

    pivot = assessment_pivot(summary)

    print("[INFO] Computing tuning deltas...")
    delta = tuning_delta(frames)
    delta.to_csv(OUT_DIR / "tuning_delta.csv", index=False)

    print("[INFO] Finding hardest questions...")
    hardest = hardest_questions(frames)
    hardest.to_csv(OUT_DIR / "hardest_questions.csv", index=False)

    print("[INFO] Building report...")
    report = build_report(summary, pivot, delta, hardest, frames)
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[INFO] All outputs saved to '{OUT_DIR}'")


if __name__ == "__main__":
    main()
