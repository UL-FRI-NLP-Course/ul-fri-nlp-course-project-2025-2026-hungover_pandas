import streamlit as st
import pandas as pd
import csv
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="RAG Evaluation Tool",
    layout="wide"
)

CSV_PATH = "eval_sheet_tuned.csv"

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

def safe_score(value):
    try:
        if pd.isna(value) or value == "":
            return 3
        return int(float(value))
    except Exception:
        return 3


def safe_bool(value):
    if pd.isna(value):
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

# -----------------------------
# COMPACT UI CSS
# -----------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 0.6rem;
    padding-bottom: 0.5rem;
    padding-left: 1rem;
    padding-right: 1rem;
    max-width: 100%;
}

html, body, [class*="css"] {
    font-size: 13px;
}

h1 {
    font-size: 1.5rem !important;
    margin-bottom: 0.3rem !important;
}

h2, h3 {
    margin-top: 0.3rem !important;
    margin-bottom: 0.2rem !important;
}

div[data-testid="stTextArea"] textarea {
    font-size: 12px !important;
    line-height: 1.25 !important;
    padding: 8px !important;
}

div[data-testid="stRadio"] label {
    font-size: 12px !important;
}

div[data-testid="stCheckbox"] label {
    font-size: 12px !important;
}

.stTextInput input {
    font-size: 12px !important;
}

hr {
    margin-top: 0.3rem !important;
    margin-bottom: 0.5rem !important;
}

.small-label {
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 0.15rem;
}

.answer-box {
    background-color: #111;
    border: 1px solid #333;
    border-radius: 6px;
    padding: 6px;
    min-height: 180px;
    max-height: 260px;
    font-size: 11px;
    line-height: 1.2;
    overflow-y: auto;
    white-space: pre-wrap;
}

.question-box {
    background-color: #161616;
    border: 1px solid #444;
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 8px;
}

.stButton > button {
    width: 100%;
    height: 2.2rem;
    font-size: 13px;
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

st.write("Loaded rows:", len(df))

# -----------------------------
# SESSION STATE
# -----------------------------
if "index" not in st.session_state:
    st.session_state.index = 0

idx = st.session_state.index
row = df.iloc[idx]

# -----------------------------
# TITLE
# -----------------------------
st.title("RAG Evaluation Tool")

top1, top2, top3 = st.columns([1,1,2])

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

# -----------------------------
# QUESTION
# -----------------------------
st.markdown(f"""
<div class="question-box">
<div class="small-label">QUESTION</div>
<div style="font-size:15px; font-weight:600;">
{row['Question']}
</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# EXPECTED ANSWER
# -----------------------------
st.markdown('<div class="small-label">GROUND TRUTH</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="answer-box">{row["Expected Answer"]}</div>',
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------
# MODEL COLUMNS
# -----------------------------
col1, col2 = st.columns(2)

# =========================================================
# LLAMA
# =========================================================
with col1:
    st.subheader("Llama 3.1")

    st.markdown(
        f'<div class="answer-box">{row["Llama 3.1 — Answer"]}</div>',
        unsafe_allow_html=True
    )

    default_llama_score = safe_score(
        row["Llama — Score (3/2/1/0)"]
    )

    llama_score = st.radio(
        "Score",
        [3, 2, 1, 0],
        horizontal=True,
        index=[3,2,1,0].index(default_llama_score),
        key=f"llama_score_{idx}"
    )

    default_llama_hall = safe_bool(
        row["Llama — Halucination"]
    )

    llama_hall = st.checkbox(
        "Hallucination",
        value=default_llama_hall,
        key=f"llama_hall_{idx}"
    )


    llama_notes = st.text_area(
        "Notes",
        value=safe_text(row["Llama — Notes"]),
        height=70,
        key=f"llama_notes_{idx}"
    )

# =========================================================
# GEMMA
# =========================================================
with col2:
    st.subheader("Gemma 3")

    st.markdown(
        f'<div class="answer-box">{row["Gemma 3 — Answer"]}</div>',
        unsafe_allow_html=True
    )

    default_gemma_score = safe_score(
        row["Gemma — Score (3/2/1/0)"]
    )

    gemma_score = st.radio(
        "Score",
        [3, 2, 1, 0],
        horizontal=True,
        index=[3,2,1,0].index(default_gemma_score),
        key=f"gemma_score_{idx}"
    )

    default_gemma_hall = safe_bool(
        row["Gemma — Halucination"]
    )

    gemma_hall = st.checkbox(
        "Hallucination",
        value=default_gemma_hall,
        key=f"gemma_hall_{idx}"
    )


    gemma_notes = st.text_area(
        "Notes",
        value=safe_text(row["Gemma — Notes"]),
        height=70,
        key=f"gemma_notes_{idx}"
    )

# -----------------------------
# SAVE FUNCTION
# -----------------------------
def save_current():
    df.loc[idx, "Llama — Score (3/2/1/0)"] = llama_score
    df.loc[idx, "Llama — Halucination"] = llama_hall
    df.loc[idx, "Llama — Notes"] = llama_notes

    df.loc[idx, "Gemma — Score (3/2/1/0)"] = gemma_score
    df.loc[idx, "Gemma — Halucination"] = gemma_hall
    df.loc[idx, "Gemma — Notes"] = gemma_notes

    df.loc[idx, "Retrieval Needed"] = retrieval_needed

    df.to_csv(
        CSV_PATH,
        index=False,
        quoting=csv.QUOTE_ALL
    )

# -----------------------------
# NAVIGATION
# -----------------------------
st.markdown("<br>", unsafe_allow_html=True)

nav1, nav2, nav3 = st.columns([1,2,1])

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
    if st.button("Next ➡", disabled=(idx >= len(df) - 1)):
        save_current()
        st.session_state.index += 1
        st.rerun()