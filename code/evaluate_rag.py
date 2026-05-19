"""
evaluate.py — Automated evaluation of the Erasmus RAG chatbot.

Runs questions from evaluation/test_questions.json through the RAG pipeline.
Supports resuming — skips questions already answered in the output CSV.

Usage:
    python code/evaluate.py                          # run all, skip existing
    python code/evaluate.py --limit 20               # first 20 only
    python code/evaluate.py --model <path>           # override model path
    python code/evaluate.py --output results.csv     # custom output file
    python code/evaluate.py --no-skip                # re-run everything

Output:
    evaluation/results.csv        — per-question results (Llama)
    evaluation/results_gemma3.csv — per-question results (Gemma)
    evaluation/summary.txt        — overall metrics
"""

import argparse
import csv
import json
import logging
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.generation").setLevel(logging.ERROR)
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

import torch
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ── Config ───────────────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = (
    "/d/hpc/projects/onj_fri/hungover_pandas/models/llama-3.1-8b-instruct"
)
DEFAULT_INDEX_DIR  = "faiss_index/"
DEFAULT_QA_FILE    = "evaluation/test_questions.json"
DEFAULT_OUTPUT     = "evaluation/results.csv"
EMBED_MODEL        = "sentence-transformers/all-MiniLM-L6-v2"

SYSTEM_PROMPT = (
    "You are a friendly and helpful Erasmus exchange assistant for students at "
    "UL FRI (Faculty of Computer and Information Science, University of Ljubljana, Slovenia). "
    "Answer questions using ONLY the context provided below. "
    "If the answer is not in the context, say exactly: "
    "\"I don't have that information in my knowledge base. "
    "Please check the UL FRI website or contact the International Office directly.\" "
    "Be concise, accurate, and always respond in English."
)


# ── Metrics ──────────────────────────────────────────────────────────────────

def token_overlap_f1(prediction: str, reference: str) -> float:
    pred_tokens = set(prediction.lower().split())
    ref_tokens  = set(reference.lower().split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def contains_key_info(prediction: str, reference: str) -> bool:
    ref_words = {w.lower().strip(".,;:()[]") for w in reference.split() if len(w) > 4}
    pred_lower = prediction.lower()
    matches = sum(1 for w in ref_words if w in pred_lower)
    return matches >= max(1, len(ref_words) // 4)


def is_fallback(prediction: str) -> bool:
    return "don't have that information" in prediction.lower()


# ── Resume: load already-answered question IDs ───────────────────────────────

def load_existing_ids(output_path: str) -> set:
    """Return set of question IDs already present in the output CSV."""
    p = Path(output_path)
    if not p.exists():
        return set()
    with open(p, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        ids = {row["id"] for row in reader if row.get("id")}
    if ids:
        print(f"  Resuming — {len(ids)} questions already answered in '{output_path}'")
    return ids


# ── Model loading ─────────────────────────────────────────────────────────────

def get_load_strategy() -> dict:
    if not torch.cuda.is_available():
        return {"dtype": torch.float32}
    major, minor = torch.cuda.get_device_capability(0)
    cc = major + minor / 10
    if cc >= 8.0:
        from transformers import BitsAndBytesConfig
        return {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            "device_map": "auto",
        }
    elif cc >= 7.5:
        from transformers import BitsAndBytesConfig
        return {"quantization_config": BitsAndBytesConfig(load_in_8bit=True), "device_map": "auto"}
    else:
        return {"dtype": torch.float16, "device_map": "auto"}


def load_components(model_path: str, index_dir: str):
    print(f"Loading FAISS index from '{index_dir}'...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.load_local(
        index_dir, embeddings, allow_dangerous_deserialization=True
    )
    print(f"  {vectorstore.index.ntotal} vectors loaded.")
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.6},
    )

    print(f"Loading LLM: {Path(model_path).name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"

    load_kwargs = get_load_strategy()
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    if hasattr(model, "generation_config") and hasattr(model.generation_config, "max_length"):
        model.generation_config.max_length = None
        model.generation_config.max_new_tokens = 512

    terminators = [tokenizer.eos_token_id]
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_id and eot_id != tokenizer.unk_token_id:
        terminators.append(eot_id)

    hf_pipe = pipeline(
        task="text-generation", model=model, tokenizer=tokenizer,
        return_full_text=False, max_new_tokens=512, do_sample=False,
        repetition_penalty=1.1, eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
    )
    llm = HuggingFacePipeline(pipeline=hf_pipe)
    print("  LLM ready.\n")
    return retriever, llm, tokenizer


def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source','unknown')}]\n{d.page_content}"
        for d in docs
    )


def build_chain(retriever, llm, tokenizer):
    def make_prompt(inputs):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{inputs['context']}\n\nQuestion: {inputs['question']}"},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RunnableLambda(make_prompt)
        | llm
        | StrOutputParser()
    )


# ── Main evaluation loop ──────────────────────────────────────────────────────

def evaluate(model_path: str, index_dir: str, qa_file: str, output: str,
             limit: int | None, skip_existing: bool):

    with open(qa_file, encoding="utf-8") as f:
        qa_pairs = json.load(f)
    if limit:
        qa_pairs = qa_pairs[:limit]

    # Determine which questions still need answers
    already_done = load_existing_ids(output) if skip_existing else set()
    todo = [qa for qa in qa_pairs if qa["id"] not in already_done]

    if not todo:
        print("All questions already answered. Nothing to do.")
        print("Run with --no-skip to re-run everything.")
        return

    print(f"\nTotal questions : {len(qa_pairs)}")
    print(f"Already answered: {len(already_done)}")
    print(f"To run now      : {len(todo)}")

    retriever, llm, tokenizer = load_components(model_path, index_dir)
    chain = build_chain(retriever, llm, tokenizer)

    Path(output).parent.mkdir(parents=True, exist_ok=True)

    # Append mode if resuming, write mode if starting fresh
    file_mode = "a" if already_done else "w"
    fieldnames = ["id", "question", "expected", "prediction",
                  "f1", "has_key_info", "is_fallback", "time_s"]

    print(f"\n{'='*65}")
    print(f"  Running ({len(todo)} questions)...")
    print(f"{'='*65}\n")

    f1_scores     = []
    fallback_count = 0

    with open(output, file_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if file_mode == "w":
            writer.writeheader()

        for i, item in enumerate(todo, 1):
            qid      = item["id"]
            question = item["question"]
            expected = item["answer"]

            print(f"[{i:3}/{len(todo)}] {qid}: {question[:70]}...")
            t0 = time.time()

            try:
                prediction = chain.invoke(question).strip()
            except Exception as e:
                prediction = f"ERROR: {e}"

            elapsed  = time.time() - t0
            f1       = token_overlap_f1(prediction, expected)
            has_key  = contains_key_info(prediction, expected)
            fallback = is_fallback(prediction)

            if fallback:
                fallback_count += 1
            f1_scores.append(f1)

            print(f"        F1={f1:.2f}  key={'✓' if has_key else '✗'}  fallback={'yes' if fallback else 'no'}  ({elapsed:.1f}s)")

            row = {
                "id": qid, "question": question, "expected": expected,
                "prediction": prediction, "f1": round(f1, 4),
                "has_key_info": has_key, "is_fallback": fallback,
                "time_s": round(elapsed, 2),
            }
            writer.writerow(row)
            f.flush()  # write immediately so progress is saved even if job times out

    print(f"\nResults saved to '{output}'")

    if f1_scores:
        avg_f1        = sum(f1_scores) / len(f1_scores)
        key_info_rate = sum(1 for r in [True] if r) / max(len(f1_scores), 1)
        fallback_rate = fallback_count / len(f1_scores)
        print(f"\n  This batch — Avg F1: {avg_f1:.3f}  Fallback rate: {fallback_rate*100:.1f}%")

    print(f"\nDone. Run code/make_human_eval_sheet.py to generate annotation file.\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Erasmus RAG chatbot.")
    parser.add_argument("--model",       default=DEFAULT_MODEL_PATH)
    parser.add_argument("--index-dir",   default=DEFAULT_INDEX_DIR)
    parser.add_argument("--qa-file",     default=DEFAULT_QA_FILE)
    parser.add_argument("--output",      default=DEFAULT_OUTPUT)
    parser.add_argument("--limit",       type=int, default=None)
    parser.add_argument("--no-skip",     action="store_true",
                        help="Re-run all questions even if already answered")
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        index_dir=args.index_dir,
        qa_file=args.qa_file,
        output=args.output,
        limit=args.limit,
        skip_existing=not args.no_skip,
    )


if __name__ == "__main__":
    main()