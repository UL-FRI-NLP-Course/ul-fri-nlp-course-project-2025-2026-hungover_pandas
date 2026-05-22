# RAG Evaluation Tool

This folder contains Streamlit apps for manually checking model answers and saving evaluation labels back into CSV files.

## Setup

```bash
pip install -r requirements.txt
```

Run the commands from this folder, or make sure the expected CSV file is in your current working directory.

## Apps

### `app.py`

Use this app for manual evaluation of the tuned RAG results in `eval_sheet_tuned.csv`.

It shows one question at a time with the expected answer, the Llama 3.1 answer, and the Gemma 3 answer. For each model, the evaluator can assign a score from 3 to 0, mark whether the answer contains a hallucination, add notes, and label whether retrieval was required, helpful, not needed, unavailable, or ambiguous.

Run it with:

```bash
streamlit run app.py
```

### `app2.py`

Use this app when comparing the first evaluation sheet against the tuned evaluation sheet.

It loads `eval_sheet_first.csv` and `eval_sheet_tuned.csv`. The app displays the tuned answer for each model while also showing the score, hallucination flag, and notes from the first evaluation when available. This makes it easier to re-evaluate the tuned outputs while keeping the previous evaluation visible as context. Only the tuned CSV is saved when you click Save.

Run it with:

```bash
streamlit run app2.py
```

### `baseline_eval_app.py`

Use this app for manual evaluation of the Gemini-generated baseline results in `gemini_evaluation_baseline_results.csv`.

It shows the question, expected answer, and baseline answer. The evaluator can assign the baseline score, mark hallucination, add notes, and label retrieval need. This app is for evaluating the non-tuned baseline separately from the Llama/Gemma RAG comparison apps.

Run it with:

```bash
streamlit run baseline_eval_app.py
```
