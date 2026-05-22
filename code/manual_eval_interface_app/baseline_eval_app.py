import streamlit as st
import pandas as pd
import csv
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Baseline Evaluation Tool",
    layout="wide"
)

CSV_PATH = "gemini_evaluation_baseline_results.csv"

EXPECTED_COLUMNS = [
    "ID",
    "Question",
    "Expected Answer",
    "Baseline — Answer",
    "Baseline — Score (3/2/1/0)",
    "Baseline — Halucination",
    "Baseline — Notes",
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
def safe_score(value):
    try:
        if pd.isna(value) or value == "":
            return 3
        score = int(float(value))
        return score if score in SCORE_OPTIONS else 3
    except Exception:
        return 3


def safe_bool(value):
    if pd.isna(value) or value == "":
        return False

    if isinstance(value, bool):
        return value

    value = str(value).strip().lower()
    return value in ["true", "1", "yes", "y"]


def safe_text(value):
    if pd.isna(value):
        return ""
    return str(value)


def load_csv(path):
    rows = []

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)

        try:
            next(reader)
        except StopIteration:
            return pd.DataFrame(columns=EXPECTED_COLUMNS)

        for row in reader:
            if len(row) < len(EXPECTED_COLUMNS):
                row += [""] * (len(EXPECTED_COLUMNS) - len(row))

            if len(row) > len(EXPECTED_COLUMNS):
                row = row[:len(EXPECTED_COLUMNS)]

            rows.append(row)

    df = pd.DataFrame(rows, columns=EXPECTED_COLUMNS)

    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    return df


def save_current():
    df.loc[idx, "Baseline — Score (3/2/1/0)"] = baseline_score
    df.loc[idx, "Baseline — Halucination"] = baseline_hall
    df.loc[idx, "Baseline — Notes"] = baseline_notes
    df.loc[idx, "Retrieval Needed"] = retrieval_needed

    df.to_csv(
        CSV_PATH,
        index=False,
        quoting=csv.QUOTE_ALL
    )


# -----------------------------
# CSS
# -----------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 0.45rem;
    padding-bottom: 0.4rem;
    padding-left: 0.8rem;
    padding-right: 0.8rem;
    max-width: 100%;
}

html, body, [class*="css"] {
    font-size: 12px;
}

h1 {
    font-size: 1.35rem !important;
    margin-bottom: 0.2rem !important;
}

h2, h3 {
    margin-top: 0.25rem !important;
    margin-bottom: 0.15rem !important;
}

div[data-testid="stTextArea"] textarea {
    font-size: 12px !important;
    line-height: 1.25 !important;
    padding: 7px !important;
}

div[data-testid="stRadio"] label,
div[data-testid="stCheckbox"] label,
div[data-testid="stSelectbox"] label {
    font-size: 12px !important;
}

.small-label {
    font-size: 11px;
    font-weight: 700;
    margin-bottom: 0.15rem;
}

.answer-box {
    background-color: #111;
    border: 1px solid #333;
    border-radius: 6px;
    padding: 7px;
    min-height: 170px;
    max-height: 250px;
    font-size: 11px;
    line-height: 1.22;
    overflow-y: auto;
    white-space: pre-wrap;
}

.question-box {
    background-color: #161616;
    border: 1px solid #444;
    border-radius: 7px;
    padding: 8px;
    margin-bottom: 6px;
}

.stButton > button {
    width: 100%;
    height: 2rem;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# LOAD CSV
# -----------------------------
if not Path(CSV_PATH).exists():
    st.error(f"CSV file not found: {CSV_PATH}")
    st.stop()

try:
    df = load_csv(CSV_PATH)
except Exception as e:
    st.error(f"Failed to load CSV: {e}")
    st.stop()

if len(df) == 0:
    st.error("CSV has no rows.")
    st.stop()


# -----------------------------
# SESSION STATE
# -----------------------------
if "index" not in st.session_state:
    st.session_state.index = 0

idx = st.session_state.index
idx = max(0, min(idx, len(df) - 1))
st.session_state.index = idx

row = df.iloc[idx]


# -----------------------------
# HEADER
# -----------------------------
st.title("Baseline Evaluation Tool")

top1, top2, top3, top4 = st.columns([1, 1, 2, 1])

with top1:
    st.metric("Question", f"{idx + 1}/{len(df)}")

with top2:
    progress = int(((idx + 1) / len(df)) * 100)
    st.metric("Progress", f"{progress}%")

current_retrieval = safe_text(row["Retrieval Needed"])
if current_retrieval not in RETRIEVAL_OPTIONS:
    current_retrieval = "required"

with top3:
    retrieval_needed = st.selectbox(
        "Retrieval Needed",
        RETRIEVAL_OPTIONS,
        index=RETRIEVAL_OPTIONS.index(current_retrieval),
        key=f"retrieval_needed_{idx}"
    )

with top4:
    st.metric("ID", safe_text(row["ID"]))


# -----------------------------
# QUESTION
# -----------------------------
st.markdown(f"""
<div class="question-box">
<div class="small-label">QUESTION</div>
<div style="font-size:14px; font-weight:600;">
{safe_text(row["Question"])}
</div>
</div>
""", unsafe_allow_html=True)


# -----------------------------
# ANSWERS
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="small-label">EXPECTED ANSWER</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="answer-box">{safe_text(row["Expected Answer"])}</div>',
        unsafe_allow_html=True
    )

with col2:
    st.markdown('<div class="small-label">BASELINE ANSWER</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="answer-box">{safe_text(row["Baseline — Answer"])}</div>',
        unsafe_allow_html=True
    )


# -----------------------------
# EVALUATION CONTROLS
# -----------------------------
eval1, eval2, eval3 = st.columns([1.2, 1, 3])

with eval1:
    baseline_score = st.radio(
        "Score",
        SCORE_OPTIONS,
        horizontal=True,
        index=SCORE_OPTIONS.index(
            safe_score(row["Baseline — Score (3/2/1/0)"])
        ),
        key=f"baseline_score_{idx}"
    )

with eval2:
    baseline_hall = st.checkbox(
        "Hallucination",
        value=safe_bool(row["Baseline — Halucination"]),
        key=f"baseline_hall_{idx}"
    )

with eval3:
    baseline_notes = st.text_area(
        "Notes",
        value=safe_text(row["Baseline — Notes"]),
        height=65,
        key=f"baseline_notes_{idx}"
    )


# -----------------------------
# NAVIGATION
# -----------------------------
st.markdown("<br>", unsafe_allow_html=True)

nav1, nav2, nav3, nav4 = st.columns([1, 1, 1, 1])

with nav1:
    if st.button("⬅ Previous", disabled=(idx == 0)):
        save_current()
        st.session_state.index -= 1
        st.rerun()

with nav2:
    if st.button("💾 Save"):
        save_current()
        st.success("Saved")

with nav3:
    jump_to = st.number_input(
        "Jump",
        min_value=1,
        max_value=len(df),
        value=idx + 1,
        step=1,
        key=f"jump_{idx}"
    )

with nav4:
    if st.button("Next ➡", disabled=(idx >= len(df) - 1)):
        save_current()
        st.session_state.index += 1
        st.rerun()

if jump_to != idx + 1:
    save_current()
    st.session_state.index = int(jump_to) - 1
    st.rerun()