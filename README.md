# Erasmus Chatbot — UL FRI
### NLP Course Project | Group: hungover_pandas

A domain-specific RAG (Retrieval-Augmented Generation) chatbot for UL FRI students interested in Erasmus exchange. The chatbot answers questions about partner universities, deadlines, financial support, visas, Learning Agreements, course options, and related mobility rules using the documents in `data/`.

---

## Repository structure

```text
hungover_pandas/
├── README.md
├── requirements.txt
├── environment.yml
├── Natural language processing 2026.pdf
├── Retrieval_augmented_generation.ipynb
├── Miniconda3-latest-Linux-x86_64.sh
├── code/
│   ├── index.py                                  # Build the FAISS index from data/
│   ├── rag.py                                    # Main RAG chatbot, interactive or test mode
│   ├── llm_test.py                               # Verify local LLM loading and generation
│   ├── evaluate.py                               # Run QA evaluation for one model
│   ├── tune_rag.py                               # Compare RAG retrieval/chunking configs
│   ├── summarize.py                              # Summarize tuned and untuned result CSVs
│   ├── make_human_eval_sheet.py                  # Build a CSV for manual human evaluation
│   ├── evaluate_baseline_with_bert_score.py      # BERTScore for Gemini baseline answers
│   ├── evaluate_llama_and_gemma_with_bert_score.py
│   ├── evaluate_llama_and_gemma_with_gemini.py   # Gemini-as-judge evaluation
│   ├── analyze_bert_scores.py                    # Aggregate BERTScore experiment outputs
│   ├── analyze_gemini_evals.py                   # Aggregate Gemini-as-judge outputs
│   ├── bert_scores_analysis.ipynb
│   ├── gemini_evals_analysis.ipynb
│   └── run_rag.sh                                # SLURM batch job
├── data/
│   ├── Erasmus+ charter.pdf
│   ├── declaration_for_disadvantaged_students.pdf
│   ├── declaration_for_late_enrolment_certificate.pdf
│   ├── guidelines-learning-studies.pdf
│   ├── online-learning-agreement.pdf
│   ├── doctoral_erasmus_mobility.txt
│   ├── erasmus_financial_support_2026_27.txt
│   ├── erasmus_general_info.txt
│   ├── travel_residence_and_visas.txt
│   ├── ul_fri_erasmus_partner_institutions.txt
│   └── partner_university_course_rag_files/
│       ├── README_partner_course_files.txt
│       ├── charles_university_mff_course_info.txt
│       ├── ntnu_course_info.txt
│       ├── other_universities_course_info.txt
│       ├── reykjavik_university_course_info.txt
│       ├── tu_berlin_course_info.txt
│       ├── tu_graz_course_info.txt
│       ├── university_of_padova_course_info.txt
│       ├── university_of_tartu_course_info.txt
│       └── vub_course_info.txt
├── evaluation/
│   ├── test_questions.json                       # QA benchmark used by evaluate.py
│   ├── test_qa_dataset.txt                       # Text version of the QA benchmark
│   ├── baseline_results.csv                      # Gemini baseline answers
│   ├── results*.csv                              # Generated model QA outputs
│   ├── summary*.txt                              # Generated evaluation summaries
│   ├── final_summary.csv
│   ├── final_summary.txt
│   ├── tuning_results.csv
│   ├── tuning_summary.txt
│   ├── all_gemini_evaluations/                   # Gemini-as-judge result CSVs
│   ├── bert_scores_analysis/                     # Aggregated BERTScore analysis outputs
│   ├── csv_bert_scores_for_all_models/           # BERTScore CSVs for scorer/model combinations
│   ├── gemini_evals_analysis/                    # Aggregated Gemini-as-judge reports
│   ├── no tune results/                          # Untuned Llama/Gemma outputs and human sheet
│   ├── summary_bert_scores_for_all_models/       # BERTScore summary text files
│   └── tuned results/                            # Tuned Llama/Gemma outputs and human sheet
├── faiss_index/
│   ├── index.faiss                               # Generated vector index
│   └── index.pkl                                 # Generated LangChain FAISS metadata
├── logs/
│   ├── rag_<JOBID>.out
│   └── rag_<JOBID>.err
└── models/
    ├── llama-3.1-8b-instruct/                    # Default local model
    └── gemma-3-4b-it/                            # Optional comparison model
```

---

## How the RAG pipeline works

```text
INDEXING
  data/**/*.txt + data/**/*.pdf
      -> split into 500-character chunks with 50-character overlap
      -> embed chunks with sentence-transformers/all-MiniLM-L6-v2
      -> save FAISS vector store to faiss_index/

QUERYING
  user question
      -> embed question
      -> retrieve relevant chunks from FAISS
      -> build a grounded prompt with retrieved context
      -> generate an answer with Llama 3.1 or Gemma 3
```

---

## Reproducible setup

Run commands from the repository root unless stated otherwise:

```bash
cd /d/hpc/projects/onj_fri/hungover_pandas
```

### 1. Start an interactive GPU job

```bash
srun --partition=gpu --gpus=1 --mem=20G --time=02:00:00 --pty bash
```

Verify the allocated GPU:

```bash
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
```

### 2. Create or activate the Python environment

On the ARNES HPC setup used by `code/run_rag.sh`:

```bash
source /cvmfs/sling.si/modules/el7/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate nlp_hungover_pandas
cd /d/hpc/projects/onj_fri/hungover_pandas
```

To recreate the Conda environment from the exported specification:

```bash
conda env create -f environment.yml
conda activate nlp_hungover_pandas
```

To update an existing environment from the same file:

```bash
conda env update -f environment.yml --prune
```

### 3. Install required Python packages

If you are not using Conda, or if the `nlp_hungover_pandas` environment is missing pip packages, install them with:

```bash
python -m pip install -r requirements.txt
```

For Gemini-as-judge evaluation, set an API key in the environment or in a local `.env` file:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

Do not commit API keys.

---

## Main RAG commands

### 1. Build or rebuild the FAISS index

Run this after changing files in `data/`:

```bash
python code/index.py --data-dir data/ --index-dir faiss_index/
```

### 2. Test model loading

```bash
python code/llm_test.py
python code/llm_test.py --model models/gemma-3-4b-it
```

### 3. Run the chatbot

Interactive mode:

```bash
python code/rag.py --mode chat
```

Useful chat commands:

- **`debug`:** Toggle retrieved source display.
- **`quit`:** Exit the chatbot.

Predefined smoke-test questions:

```bash
python code/rag.py --mode test
```

Use Gemma instead of the default Llama model:

```bash
python code/rag.py --mode chat --model models/gemma-3-4b-it
```

---

## Evaluation commands

### Run automatic QA evaluation

Quick test:

```bash
python code/evaluate.py --limit 10 --output evaluation/results_llama_quick.csv
```

Full Llama evaluation:

```bash
python code/evaluate.py \
  --model models/llama-3.1-8b-instruct \
  --output evaluation/results_llama.csv \
  --no-skip
```

Full Gemma evaluation:

```bash
python code/evaluate.py \
  --model models/gemma-3-4b-it \
  --output evaluation/results_gemma3.csv \
  --no-skip
```

The default QA file is `evaluation/test_questions.json`.

### Tune retrieval settings

```bash
python code/tune_rag.py --limit 20
python code/tune_rag.py --model models/gemma-3-4b-it --limit 20
```

Expected outputs:

- **`evaluation/tuning_results.csv`:** Per-question tuning results.
- **`evaluation/tuning_summary.txt`:** Comparison table.

### Summarize model result CSVs

```bash
python code/summarize.py
```

Expected outputs:

- **`evaluation/final_summary.csv`:** Machine-readable summary.
- **`evaluation/final_summary.txt`:** Human-readable summary.

### Create a human evaluation sheet

```bash
python code/make_human_eval_sheet.py
```

Expected output:

- **`evaluation/tuned results/human_eval_sheet_tuned.csv`:** Spreadsheet for manual scoring.

---

## BERTScore and Gemini-as-judge commands

### BERTScore for the Gemini baseline

```bash
python code/evaluate_baseline_with_bert_score.py \
  --input evaluation/baseline_results.csv \
  --limit 10
```

Omit `--limit 10` for the full baseline evaluation.

### BERTScore for Llama and Gemma outputs

This script expects these inputs:

- **`evaluation/no tune results/llama_results.csv`**
- **`evaluation/no tune results/results_gemma3.csv`**
- **`evaluation/tuned results/results_llama_tuned.csv`**
- **`evaluation/tuned results/results_gemma3_tuned.csv`**

Run:

```bash
python code/evaluate_llama_and_gemma_with_bert_score.py
```

### Aggregate BERTScore outputs

This script reads `evaluation/csv_bert_scores_for_all_models/*.csv` and writes analysis files to `evaluation/bert_scores_analysis/`.

```bash
python code/analyze_bert_scores.py
```

### Gemini-as-judge evaluation

Requires `GEMINI_API_KEY`.

```bash
export GEMINI_API_KEY="your_api_key_here"
python code/evaluate_llama_and_gemma_with_gemini.py
```

### Aggregate Gemini-as-judge outputs

This script reads `evaluation/all_gemini_evaluations/*.csv` and writes analysis files to `evaluation/gemini_evals_analysis/`.

```bash
python code/analyze_gemini_evals.py
```

---

## SLURM batch execution

Submit the included batch job:

```bash
sbatch code/run_rag.sh
```

Check job status:

```bash
squeue -u "$USER"
```

Inspect logs after replacing `<JOBID>` with the SLURM job ID:

```bash
tail -f logs/rag_<JOBID>.out
tail -f logs/rag_<JOBID>.err
```

`code/run_rag.sh` activates the `nlp_hungover_pandas` conda environment, moves to `/d/hpc/projects/onj_fri/hungover_pandas`, and runs:

```bash
python code/rag.py --mode test
```

---

## Models

| Model | Path | Use |
|---|---|---|
| Llama 3.1 8B Instruct | `models/llama-3.1-8b-instruct` | Default chatbot and evaluation model |
| Gemma 3 4B IT | `models/gemma-3-4b-it` | Faster comparison model |

Model loading automatically chooses a strategy from the available GPU:

- **Compute capability >= 8.0:** 4-bit NF4 quantization.
- **Compute capability >= 7.5:** 8-bit quantization.
- **Older CUDA GPU:** float16.
- **No GPU:** float32 CPU fallback, which is very slow.

---

## Evaluation metrics

| Metric | Description |
|---|---|
| Token F1 | Lexical token overlap between prediction and expected answer. |
| Key info hit rate | Whether important reference terms appear in the prediction. |
| Fallback rate | How often the model says the answer is not in the knowledge base. |
| BERTScore | Semantic similarity between prediction and reference using transformer scorer models. |
| Gemini-as-judge score | LLM-based semantic assessment. |
| Human evaluation score | Shown and explained in the report. |

---

## Contact

FRI Erasmus office: erasmus@fri.uni-lj.si