from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "evaluation" / "csv_bert_scores_for_all_models"
OUT_DIR = ROOT / "evaluation" / "bert_scores_analysis"
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 42
BOOTSTRAP_CI = 0.95


def parse_csv_name(path: Path) -> tuple[str, str]:
    name = path.stem
    baseline_marker = "baseline_results_with_bert_score_"
    if name.startswith(baseline_marker):
        return "Gemini baseline", normalize_scorer_name(name.replace(baseline_marker, ""))

    model_markers = [
        ("llama_not_tuned", "Llama not tuned"),
        ("llama_tuned", "Llama tuned"),
        ("gemma_not_tuned", "Gemma not tuned"),
        ("gemma_tuned", "Gemma tuned"),
    ]
    for marker_key, label in model_markers:
        marker = f"{marker_key}_results_with_bert_score_"
        if name.startswith(marker):
            return label, normalize_scorer_name(name.replace(marker, ""))

    return "Unknown", normalize_scorer_name(name)


def normalize_scorer_name(name: str | None) -> str | None:
    if name is None:
        return None
    return {"roberta_large": "roberta-large"}.get(name, name)


def bootstrap_mean_ci(
    values: pd.Series,
    rng: np.random.Generator,
    iterations: int = BOOTSTRAP_ITERATIONS,
    ci: float = BOOTSTRAP_CI,
) -> tuple[float, float, float]:
    array = values.dropna().to_numpy(dtype=float)
    if len(array) == 0:
        return np.nan, np.nan, np.nan
    samples = rng.choice(array, size=(iterations, len(array)), replace=True)
    means = samples.mean(axis=1)
    alpha = (1 - ci) / 2
    return float(means.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def load_csv_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    frames = []
    agg_rows = []
    for path in sorted(CSV_DIR.glob("*.csv")):
        evaluated_model, scorer_model = parse_csv_name(path)
        df = pd.read_csv(path)
        df["evaluated_model"] = evaluated_model
        df["scorer_model"] = scorer_model
        df["source_file"] = path.name
        pred_col = "gemini_answer" if "gemini_answer" in df.columns else "model_prediction"
        df["prediction_text"] = df[pred_col]
        frames.append(df)
        bootstrap_mean, bootstrap_low, bootstrap_high = bootstrap_mean_ci(df["bert_F1"], rng)
        agg_rows.append(
            {
                "evaluated_model": evaluated_model,
                "scorer_model": scorer_model,
                "source_file": path.name,
                "n": len(df),
                "mean_P": df["bert_P"].mean(),
                "mean_R": df["bert_R"].mean(),
                "mean_F1": df["bert_F1"].mean(),
                "median_F1": df["bert_F1"].median(),
                "std_F1": df["bert_F1"].std(),
                "min_F1": df["bert_F1"].min(),
                "max_F1": df["bert_F1"].max(),
                "bootstrap_mean_F1": bootstrap_mean,
                "bootstrap_ci95_low_F1": bootstrap_low,
                "bootstrap_ci95_high_F1": bootstrap_high,
            }
        )

    all_rows = pd.concat(frames, ignore_index=True)
    aggregate = pd.DataFrame(agg_rows).sort_values(
        ["scorer_model", "mean_F1"], ascending=[True, False]
    )
    return all_rows, aggregate


def bootstrap_tuning_deltas(all_rows: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    rows = []
    for base in ["Gemma", "Llama"]:
        tuned = f"{base} tuned"
        untuned = f"{base} not tuned"
        for scorer_model, scorer_rows in all_rows.groupby("scorer_model"):
            tuned_rows = scorer_rows[scorer_rows["evaluated_model"] == tuned][["id", "bert_F1"]]
            untuned_rows = scorer_rows[scorer_rows["evaluated_model"] == untuned][["id", "bert_F1"]]
            paired = tuned_rows.merge(
                untuned_rows,
                on="id",
                suffixes=("_tuned", "_not_tuned"),
            )
            if paired.empty:
                continue
            diff = paired["bert_F1_tuned"] - paired["bert_F1_not_tuned"]
            bootstrap_mean, bootstrap_low, bootstrap_high = bootstrap_mean_ci(diff, rng)
            rows.append(
                {
                    "model_family": base,
                    "scorer_model": scorer_model,
                    "n_paired_questions": len(paired),
                    "mean_delta_F1": diff.mean(),
                    "bootstrap_mean_delta_F1": bootstrap_mean,
                    "bootstrap_ci95_low_delta_F1": bootstrap_low,
                    "bootstrap_ci95_high_delta_F1": bootstrap_high,
                    "ci_excludes_zero": bootstrap_low > 0 or bootstrap_high < 0,
                }
            )
    return pd.DataFrame(rows).sort_values(["model_family", "mean_delta_F1"], ascending=[True, False])


def markdown_table(df: pd.DataFrame) -> str:
    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda value: f"{value:.4f}")
    header = "| " + " | ".join(map(str, formatted.columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(formatted.columns)) + " |"
    rows = [
        "| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |"
        for row in formatted.to_numpy()
    ]
    return "\n".join([header, separator, *rows])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows, aggregate = load_csv_results()
    bootstrap_deltas = bootstrap_tuning_deltas(all_rows)

    aggregate.to_csv(OUT_DIR / "aggregate_by_evaluated_and_scorer_model.csv", index=False)
    all_rows.to_csv(OUT_DIR / "all_bert_score_rows_long.csv", index=False)
    bootstrap_deltas.to_csv(OUT_DIR / "bootstrap_tuning_deltas_by_scorer.csv", index=False)

    model_overall = (
        aggregate.groupby("evaluated_model")
        .agg(
            scorer_models=("scorer_model", "nunique"),
            avg_mean_P=("mean_P", "mean"),
            avg_mean_R=("mean_R", "mean"),
            avg_mean_F1=("mean_F1", "mean"),
            sd_across_scorers_F1=("mean_F1", "std"),
            best_scorer_F1=("mean_F1", "max"),
            worst_scorer_F1=("mean_F1", "min"),
            avg_bootstrap_ci95_low_F1=("bootstrap_ci95_low_F1", "mean"),
            avg_bootstrap_ci95_high_F1=("bootstrap_ci95_high_F1", "mean"),
        )
        .reset_index()
        .sort_values("avg_mean_F1", ascending=False)
    )
    model_overall.to_csv(OUT_DIR / "overall_model_ranking_across_scorers.csv", index=False)

    scorer_sensitivity = (
        aggregate.groupby("scorer_model")
        .agg(
            evaluated_models=("evaluated_model", "nunique"),
            avg_F1_across_evaluated_models=("mean_F1", "mean"),
            spread_between_evaluated_models=("mean_F1", lambda s: s.max() - s.min()),
            mean_std_within_files=("std_F1", "mean"),
        )
        .reset_index()
        .sort_values("avg_F1_across_evaluated_models", ascending=False)
    )
    scorer_sensitivity.to_csv(OUT_DIR / "scorer_model_sensitivity.csv", index=False)

    pivot = aggregate.pivot(index="scorer_model", columns="evaluated_model", values="mean_F1").reset_index()
    for base in ["Llama", "Gemma"]:
        tuned = f"{base} tuned"
        untuned = f"{base} not tuned"
        if tuned in pivot and untuned in pivot:
            pivot[f"{base} tuned_minus_not_tuned"] = pivot[tuned] - pivot[untuned]
    pivot.to_csv(OUT_DIR / "mean_f1_pivot_by_scorer.csv", index=False)

    question_consensus = (
        all_rows.groupby(["id", "question", "expected"])
        .agg(
            avg_F1=("bert_F1", "mean"),
            sd_F1=("bert_F1", "std"),
            min_F1=("bert_F1", "min"),
            max_F1=("bert_F1", "max"),
            observations=("bert_F1", "count"),
        )
        .reset_index()
        .sort_values("avg_F1")
    )
    question_consensus.head(25).to_csv(OUT_DIR / "lowest_scoring_questions_overall.csv", index=False)
    question_consensus.tail(25).sort_values("avg_F1", ascending=False).to_csv(
        OUT_DIR / "highest_scoring_questions_overall.csv", index=False
    )

    cols = [
        c
        for c in [
            "scorer_model",
            "Gemma not tuned",
            "Gemma tuned",
            "Gemma tuned_minus_not_tuned",
            "Llama not tuned",
            "Llama tuned",
            "Llama tuned_minus_not_tuned",
            "Gemini baseline",
        ]
        if c in pivot.columns
    ]

    lines = [
        "# BERTScore Results Analysis",
        "",
        f"Analyzed {len(list(CSV_DIR.glob('*.csv')))} CSV files.",
        f"Bootstrap settings: {BOOTSTRAP_ITERATIONS:,} resamples, {int(BOOTSTRAP_CI * 100)}% percentile confidence intervals, seed {BOOTSTRAP_SEED}.",
        "",
        "## Overall ranking across scorer models",
        markdown_table(model_overall),
        "",
        "## Tuned vs not tuned mean F1 by scorer",
        markdown_table(pivot[cols]),
        "",
        "## Scorer model sensitivity",
        markdown_table(scorer_sensitivity),
        "",
        "## Bootstrap tuning deltas by scorer",
        markdown_table(bootstrap_deltas),
        "",
        "## Key observations",
    ]
    best = model_overall.iloc[0]
    lines.append(
        f"- Best average evaluated model across scorer models: **{best.evaluated_model}** with avg mean F1 **{best.avg_mean_F1:.4f}**."
    )
    if "Gemma tuned_minus_not_tuned" in pivot:
        lines.append(
            f"- Gemma tuning delta averaged over scorers: **{pivot['Gemma tuned_minus_not_tuned'].mean():+.4f} F1**."
        )
        gemma_significant = bootstrap_deltas[
            (bootstrap_deltas["model_family"] == "Gemma") & bootstrap_deltas["ci_excludes_zero"]
        ]
        lines.append(
            f"- Gemma bootstrap CIs exclude zero for **{len(gemma_significant)}/{bootstrap_deltas[bootstrap_deltas['model_family'] == 'Gemma'].shape[0]}** scorer models."
        )
    if "Llama tuned_minus_not_tuned" in pivot:
        lines.append(
            f"- Llama tuning delta averaged over scorers: **{pivot['Llama tuned_minus_not_tuned'].mean():+.4f} F1**."
        )
        llama_significant = bootstrap_deltas[
            (bootstrap_deltas["model_family"] == "Llama") & bootstrap_deltas["ci_excludes_zero"]
        ]
        lines.append(
            f"- Llama bootstrap CIs exclude zero for **{len(llama_significant)}/{bootstrap_deltas[bootstrap_deltas['model_family'] == 'Llama'].shape[0]}** scorer models."
        )
    lines.append(
        f"- Highest absolute-F1 scorer: **{scorer_sensitivity.iloc[0].scorer_model}** with mean F1 **{scorer_sensitivity.iloc[0].avg_F1_across_evaluated_models:.4f}**."
    )
    lines.append(
        f"- Lowest absolute-F1 scorer: **{scorer_sensitivity.iloc[-1].scorer_model}** with mean F1 **{scorer_sensitivity.iloc[-1].avg_F1_across_evaluated_models:.4f}**."
    )

    report = "\n".join(lines)
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
