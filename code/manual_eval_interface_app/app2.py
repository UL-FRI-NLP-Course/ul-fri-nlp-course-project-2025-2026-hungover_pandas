import csv
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Tuned RAG Evaluation", layout="wide")

FIRST_CSV_PATH = "eval_sheet_first.csv"
TUNED_CSV_PATH = "eval_sheet_tuned.csv"

EXPECTED_COLUMNS = [
    "ID",
    "Question",
    "Expected Answer",
    "Llama 3.1 — Answer",
    "Llama — Score (3/2/1/0)",
    "Llama — Halucination",
    "Llama — Notes",
    "Gemma 3 — Answer",
    "Gemma — Score (3/2/1/0)",
    "Gemma — Halucination",
    "Gemma — Notes",
    "Retrieval Needed",
]

RETRIEVAL_OPTIONS = [
    "required",
    "helpful",
    "not needed",
    "unavailable",
    "ambiguous",
]

SCORE_OPTIONS = [3, 2, 1, 0]


# -----------------------------
# HELPERS
# -----------------------------
def safe_text(value):
    if pd.isna(value):
        return ""
    return str(value)


def safe_score(value):
    try:
        if pd.isna(value) or value == "":
            return 3
        return int(float(value))
    except Exception:
        return 3


def safe_bool(value):
    if pd.isna(value) or value == "":
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ["true", "1", "yes", "y"]


def load_eval_csv(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        st.error(f"Missing CSV file: {path}")
        st.stop()

    try:
        df = pd.read_csv(
            path,
            engine="python",
            quotechar='"',
            keep_default_na=False,
            on_bad_lines="warn",
        )
    except Exception as e:
        st.error(f"Failed to load {path}: {e}")
        st.stop()

    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df = df[EXPECTED_COLUMNS]

    return df


def save_tuned_csv(df: pd.DataFrame):
    df = df[EXPECTED_COLUMNS].fillna("")

    df.to_csv(
        TUNED_CSV_PATH,
        index=False,
        quoting=csv.QUOTE_ALL,
    )


def answer_box(title: str, text: str, meta: str = ""):
    st.markdown(f"<div class='box-title'>{title}</div>", unsafe_allow_html=True)

    if meta:
        st.markdown(
            f"<div class='meta'>{meta}</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"<div class='answer-box'>{safe_text(text)}</div>",
        unsafe_allow_html=True,
    )


def unresolved(value):
    return safe_text(value).strip() == ""


# -----------------------------
# CSS
# -----------------------------
st.markdown(
    """
<style>
.block-container {
    padding-top: 0.35rem;
    padding-bottom: 0.4rem;
    padding-left: 0.7rem;
    padding-right: 0.7rem;
    max-width: 100%;
}

html, body, [class*="css"] {
    font-size: 12px;
}

h1 {
    font-size: 1.2rem !important;
    margin: 0 !important;
    padding: 0 !important;
}

.question-box {
    background-color: #151515;
    border: 1px solid #444;
    border-radius: 6px;
    padding: 7px;
    margin-bottom: 5px;
    font-size: 13px;
}

.answer-box {
    background-color: #101010;
    border: 1px solid #333;
    border-radius: 6px;
    padding: 6px;
    min-height: 135px;
    max-height: 220px;
    overflow-y: auto;
    white-space: pre-wrap;
    font-size: 11px;
    line-height: 1.18;
}

.expected-box {
    background-color: #0f1a12;
    border: 1px solid #2c5a35;
    border-radius: 6px;
    padding: 6px;
    max-height: 145px;
    overflow-y: auto;
    white-space: pre-wrap;
    font-size: 11px;
    line-height: 1.18;
}

.box-title {
    font-weight: 700;
    font-size: 11px;
    margin-top: 0.15rem;
    margin-bottom: 0.1rem;
    color: #ddd;
}

.meta {
    color: #aaa;
    font-size: 10px;
    margin-bottom: 2px;
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# LOAD DATA
# -----------------------------
first_df = load_eval_csv(FIRST_CSV_PATH)
tuned_df = load_eval_csv(TUNED_CSV_PATH)

if len(first_df) != len(tuned_df):
    st.warning(
        f"Row mismatch: first.csv has {len(first_df)} rows, tuned.csv has {len(tuned_df)} rows."
    )

if len(tuned_df) == 0:
    st.error("No rows loaded.")
    st.stop()

if "idx" not in st.session_state:
    st.session_state.idx = 0

idx = st.session_state.idx
idx = max(0, min(idx, len(tuned_df) - 1))
st.session_state.idx = idx

first_row = first_df.iloc[idx]
tuned_row = tuned_df.iloc[idx]

# -----------------------------
# HEADER
# -----------------------------
st.title("Tuned RAG Evaluation")

top1, top2, top3, top4 = st.columns([0.8, 0.8, 1.2, 1.2])

with top1:
    st.metric("Row", f"{idx + 1}/{len(first_df)}")

with top2:
    st.metric("ID", safe_text(tuned_row["ID"]))

retrieval_default = safe_text(first_row["Retrieval Needed"])
if retrieval_default not in RETRIEVAL_OPTIONS:
    retrieval_default = "required"

with top3:
    retrieval_needed = st.selectbox(
        "Retrieval Needed",
        RETRIEVAL_OPTIONS,
        index=RETRIEVAL_OPTIONS.index(retrieval_default),
    )

with top4:
    unanswered = (
        tuned_df["Llama — Score (3/2/1/0)"].replace("", pd.NA).isna().sum()
        + tuned_df["Gemma — Score (3/2/1/0)"].replace("", pd.NA).isna().sum()
    )
    st.metric("Unanswered", int(unanswered))

st.markdown(
    f"<div class='question-box'><b>Question:</b> {safe_text(tuned_row['Question'])}</div>",
    unsafe_allow_html=True,
)

st.markdown("<div class='box-title'>EXPECTED ANSWER</div>", unsafe_allow_html=True)

st.markdown(
    f"<div class='expected-box'>{safe_text(tuned_row['Expected Answer'])}</div>",
    unsafe_allow_html=True,
)

# -----------------------------
# MODEL UI
# -----------------------------
llama_col, gemma_col = st.columns(2)

with llama_col:
    st.subheader("Llama 3.1")

    first_meta = (
        f"first score: {safe_text(first_row['Llama — Score (3/2/1/0)'])} | "
        f"first hallucination: {safe_text(first_row['Llama — Halucination'])}"
    )

    answer_box(
        "FIRST VERSION ANSWER",
        first_row["Llama 3.1 — Answer"],
        first_meta,
    )

    if safe_text(first_row["Llama — Notes"]):
        st.caption(f"first notes: {safe_text(first_row['Llama — Notes'])}")

    answer_box(
        "TUNED VERSION ANSWER",
        tuned_row["Llama 3.1 — Answer"],
    )

    if unresolved(tuned_row["Llama — Score (3/2/1/0)"]):
        llama_default_score = safe_score(first_row["Llama — Score (3/2/1/0)"])
    else:
        llama_default_score = safe_score(
            tuned_row["Llama — Score (3/2/1/0)"]
        )

    llama_score = st.radio(
        "Tuned score",
        SCORE_OPTIONS,
        horizontal=True,
        index=SCORE_OPTIONS.index(llama_default_score),
        key=f"llama_score_{idx}",
    )

    if unresolved(tuned_row["Llama — Halucination"]):
        llama_default_hall = safe_bool(first_row["Llama — Halucination"])
    else:
        llama_default_hall = safe_bool(
            tuned_row["Llama — Halucination"]
        )

    llama_hall = st.checkbox(
        "Tuned hallucination",
        value=llama_default_hall,
        key=f"llama_hall_{idx}",
    )

    llama_notes = st.text_area(
        "Tuned notes",
        value=safe_text(tuned_row["Llama — Notes"]),
        height=55,
        key=f"llama_notes_{idx}",
    )

with gemma_col:
    st.subheader("Gemma 3")

    first_meta = (
        f"first score: {safe_text(first_row['Gemma — Score (3/2/1/0)'])} | "
        f"first hallucination: {safe_text(first_row['Gemma — Halucination'])}"
    )

    answer_box(
        "FIRST VERSION ANSWER",
        first_row["Gemma 3 — Answer"],
        first_meta,
    )

    if safe_text(first_row["Gemma — Notes"]):
        st.caption(f"first notes: {safe_text(first_row['Gemma — Notes'])}")

    answer_box(
        "TUNED VERSION ANSWER",
        tuned_row["Gemma 3 — Answer"],
    )

    if unresolved(tuned_row["Gemma — Score (3/2/1/0)"]):
        gemma_default_score = safe_score(first_row["Gemma — Score (3/2/1/0)"])
    else:
        gemma_default_score = safe_score(
            tuned_row["Gemma — Score (3/2/1/0)"]
        )

    gemma_score = st.radio(
        "Tuned score",
        SCORE_OPTIONS,
        horizontal=True,
        index=SCORE_OPTIONS.index(gemma_default_score),
        key=f"gemma_score_{idx}",
    )

    if unresolved(tuned_row["Gemma — Halucination"]):
        gemma_default_hall = safe_bool(first_row["Gemma — Halucination"])
    else:
        gemma_default_hall = safe_bool(
            tuned_row["Gemma — Halucination"]
        )

    gemma_hall = st.checkbox(
        "Tuned hallucination",
        value=gemma_default_hall,
        key=f"gemma_hall_{idx}",
    )

    gemma_notes = st.text_area(
        "Tuned notes",
        value=safe_text(tuned_row["Gemma — Notes"]),
        height=55,
        key=f"gemma_notes_{idx}",
    )

# -----------------------------
st.caption("Changes are saved only when you click 💾 Save. Moving between rows does not write to the CSV.")
# NAVIGATION
# -----------------------------
nav1, nav2, nav3, nav4 = st.columns([1, 1, 1, 1])

with nav1:
    if st.button("⬅ Previous", disabled=(idx == 0)):
        st.session_state.idx -= 1
        st.rerun()

with nav2:
    if st.button("💾 Save"):
        tuned_df.loc[idx, "Llama — Score (3/2/1/0)"] = llama_score
        tuned_df.loc[idx, "Llama — Halucination"] = llama_hall
        tuned_df.loc[idx, "Llama — Notes"] = llama_notes

        tuned_df.loc[idx, "Gemma — Score (3/2/1/0)"] = gemma_score
        tuned_df.loc[idx, "Gemma — Halucination"] = gemma_hall
        tuned_df.loc[idx, "Gemma — Notes"] = gemma_notes

        tuned_df.loc[idx, "Retrieval Needed"] = retrieval_needed

        save_tuned_csv(tuned_df)
        st.success("Saved")

with nav3:
    jump_to = st.number_input(
        "Jump to row",
        min_value=1,
        max_value=len(first_df),
        value=idx + 1,
        step=1,
    )

with nav4:
    if st.button("Next ➡", disabled=(idx >= len(first_df) - 1)):
        st.session_state.idx += 1
        st.rerun()

if jump_to != idx + 1:
    st.session_state.idx = int(jump_to) - 1
    st.rerun()

print(len(first_df))
print(len(tuned_df))