"""
evaluate.py — Automated evaluation of the Erasmus RAG chatbot.

Runs all questions from evaluation/test_questions.json through the RAG pipeline,
compares answers to ground truth, and saves a results CSV + summary statistics.

Usage:
    python code/evaluate.py                          # evaluate all questions
    python code/evaluate.py --limit 20               # first 20 questions only
    python code/evaluate.py --model <path>           # override model path
    python code/evaluate.py --output results.csv     # custom output file

Output:
    evaluation/results.csv   — per-question results
    evaluation/summary.txt   — overall metrics for your report
"""

import argparse
import csv
import json
import logging
import sys
import time
import warnings
from pathlib import Path

# Suppress noisy but harmless warnings from transformers / bitsandbytes
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


# ── Simple lexical metrics (no extra deps needed) ────────────────────────────

def token_overlap_f1(prediction: str, reference: str) -> float:
    """
    Token-level F1 score between prediction and reference.
    Fast proxy for answer quality — used in SQuAD evaluation.
    """
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
    """
    Rough check: does the prediction contain at least one meaningful noun
    from the reference? Useful for catching total misses.
    """
    # Extract "content words" (length > 4) from reference
    ref_words = {w.lower().strip(".,;:()[]") for w in reference.split() if len(w) > 4}
    pred_lower = prediction.lower()
    matches = sum(1 for w in ref_words if w in pred_lower)
    return matches >= max(1, len(ref_words) // 4)


def is_fallback(prediction: str) -> bool:
    return "don't have that information" in prediction.lower()


# ── Model loading (same as rag.py) ───────────────────────────────────────────

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

def evaluate(model_path: str, index_dir: str, qa_file: str, output: str, limit: int | None):
    # Load QA pairs
    qa_path = Path(qa_file)
    if not qa_path.exists():
        print(f"ERROR: QA file not found at '{qa_file}'")
        sys.exit(1)
    with open(qa_path, encoding="utf-8") as f:
        qa_pairs = json.load(f)
    if limit:
        qa_pairs = qa_pairs[:limit]
    print(f"Evaluating {len(qa_pairs)} questions from '{qa_file}'")

    # Load components
    retriever, llm, tokenizer = load_components(model_path, index_dir)
    chain = build_chain(retriever, llm, tokenizer)

    # Run evaluation
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    results = []
    f1_scores = []
    fallback_count = 0

    print(f"\n{'='*65}")
    print(f"  Running evaluation ({len(qa_pairs)} questions)...")
    print(f"{'='*65}\n")

    for i, item in enumerate(qa_pairs, 1):
        qid      = item["id"]
        question = item["question"]
        expected = item["answer"]

        print(f"[{i:3}/{len(qa_pairs)}] {qid}: {question[:70]}...")
        t0 = time.time()

        try:
            prediction = chain.invoke(question).strip()
        except Exception as e:
            prediction = f"ERROR: {e}"

        elapsed = time.time() - t0
        f1      = token_overlap_f1(prediction, expected)
        has_key = contains_key_info(prediction, expected)
        fallback = is_fallback(prediction)

        if fallback:
            fallback_count += 1
        f1_scores.append(f1)

        print(f"        F1={f1:.2f}  key_info={'✓' if has_key else '✗'}  fallback={'yes' if fallback else 'no'}  ({elapsed:.1f}s)")

        results.append({
            "id":          qid,
            "question":    question,
            "expected":    expected,
            "prediction":  prediction,
            "f1":          round(f1, 4),
            "has_key_info": has_key,
            "is_fallback": fallback,
            "time_s":      round(elapsed, 2),
        })

    # Save CSV
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to '{output}'")

    # Summary
    avg_f1        = sum(f1_scores) / len(f1_scores)
    key_info_rate = sum(1 for r in results if r["has_key_info"]) / len(results)
    fallback_rate = fallback_count / len(results)

    summary = (
        f"\n{'='*65}\n"
        f"  EVALUATION SUMMARY\n"
        f"{'='*65}\n"
        f"  Model        : {Path(model_path).name}\n"
        f"  Questions    : {len(results)}\n"
        f"  Avg F1 score : {avg_f1:.4f}  ({avg_f1*100:.1f}%)\n"
        f"  Key info hit : {key_info_rate*100:.1f}%  (answer contains relevant terms)\n"
        f"  Fallback rate: {fallback_rate*100:.1f}%  (\"don't have that information\")\n"
        f"{'='*65}\n"
    )
    print(summary)

    summary_path = Path(output).parent / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Summary saved to '{summary_path}'")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Erasmus RAG chatbot.")
    parser.add_argument("--model",     default=DEFAULT_MODEL_PATH)
    parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    parser.add_argument("--qa-file",   default=DEFAULT_QA_FILE)
    parser.add_argument("--output",    default=DEFAULT_OUTPUT)
    parser.add_argument("--limit",     type=int, default=None,
                        help="Evaluate only first N questions (for quick testing)")
    args = parser.parse_args()

    evaluate(args.model, args.index_dir, args.qa_file, args.output, args.limit)


if __name__ == "__main__":
    main()