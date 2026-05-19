"""
tune_rag.py — Test multiple RAG configurations and find the best one.

Loads the LLM once, then swaps retriever settings between configs.
Runs each config on 50 questions and prints a comparison table.

Usage:
    python code/tune_rag.py                    # test all configs, 50 questions
    python code/tune_rag.py --limit 20         # faster, 20 questions per config
    python code/tune_rag.py --model <path>     # override model

Output:
    evaluation/tuning_results.csv   — full results per config
    evaluation/tuning_summary.txt   — comparison table (copy into report)
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
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ── Paths ─────────────────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = (
    "/d/hpc/projects/onj_fri/hungover_pandas/models/llama-3.1-8b-instruct"
)
DEFAULT_QA_FILE = "evaluation/test_questions.json"
DATA_DIR        = "data/"

SYSTEM_PROMPT = (
    "You are a friendly and helpful Erasmus exchange assistant for students at "
    "UL FRI (Faculty of Computer and Information Science, University of Ljubljana, Slovenia). "
    "Answer questions using ONLY the context provided below. "
    "If the answer is not in the context, say exactly: "
    "\"I don't have that information in my knowledge base. "
    "Please check the UL FRI website or contact the International Office directly.\" "
    "Be concise, accurate, and always respond in English."
)

# ── Configurations to test ────────────────────────────────────────────────────
# Each config changes ONE OR TWO things from the baseline so results are interpretable.
CONFIGS = [
    {
        "name":        "baseline",
        "description": "k=3, chunk=500, MMR, MiniLM",
        "chunk_size":  500,
        "chunk_overlap": 50,
        "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
        "search_type": "mmr",
        "k":           3,
        "fetch_k":     10,
        "lambda_mult": 0.6,
    },
    {
        "name":        "k5",
        "description": "k=5, chunk=500, MMR, MiniLM",
        "chunk_size":  500,
        "chunk_overlap": 50,
        "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
        "search_type": "mmr",
        "k":           5,
        "fetch_k":     15,
        "lambda_mult": 0.6,
    },
    {
        "name":        "k7",
        "description": "k=7, chunk=500, MMR, MiniLM",
        "chunk_size":  500,
        "chunk_overlap": 50,
        "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
        "search_type": "mmr",
        "k":           7,
        "fetch_k":     20,
        "lambda_mult": 0.6,
    },
    {
        "name":        "chunk300",
        "description": "k=3, chunk=300, MMR, MiniLM",
        "chunk_size":  300,
        "chunk_overlap": 30,
        "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
        "search_type": "mmr",
        "k":           3,
        "fetch_k":     10,
        "lambda_mult": 0.6,
    },
    {
        "name":        "chunk700",
        "description": "k=3, chunk=700, MMR, MiniLM",
        "chunk_size":  700,
        "chunk_overlap": 70,
        "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
        "search_type": "mmr",
        "k":           3,
        "fetch_k":     10,
        "lambda_mult": 0.6,
    },
    {
        "name":        "similarity_k5",
        "description": "k=5, chunk=500, similarity, MiniLM",
        "chunk_size":  500,
        "chunk_overlap": 50,
        "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
        "search_type": "similarity",
        "k":           5,
        "fetch_k":     5,   # not used for similarity but kept for consistency
        "lambda_mult": None,
    },
    {
        "name":        "mpnet_k5",
        "description": "k=5, chunk=500, MMR, mpnet",
        "chunk_size":  500,
        "chunk_overlap": 50,
        "embed_model": "sentence-transformers/all-mpnet-base-v2",
        "search_type": "mmr",
        "k":           5,
        "fetch_k":     15,
        "lambda_mult": 0.6,
    },
]


# ── Metrics ───────────────────────────────────────────────────────────────────

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
    ref_words  = {w.lower().strip(".,;:()[]") for w in reference.split() if len(w) > 4}
    pred_lower = prediction.lower()
    matches    = sum(1 for w in ref_words if w in pred_lower)
    return matches >= max(1, len(ref_words) // 4)


def is_fallback(prediction: str) -> bool:
    return "don't have that information" in prediction.lower()


# ── LLM loading (once) ────────────────────────────────────────────────────────

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


def load_llm(model_path: str):
    print(f"\nLoading LLM once: {Path(model_path).name}")
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        cc   = torch.cuda.get_device_capability(0)
        print(f"  GPU: {torch.cuda.get_device_name(0)}  ({vram:.1f} GB, CC {cc[0]}.{cc[1]})")

    load_kwargs = get_load_strategy()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    if hasattr(model, "generation_config") and hasattr(model.generation_config, "max_length"):
        model.generation_config.max_length     = None
        model.generation_config.max_new_tokens = 512
    print(f"  Loaded in {time.time()-t0:.1f}s")

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
    return llm, tokenizer


# ── Index building per config ─────────────────────────────────────────────────

def build_index_for_config(cfg: dict) -> FAISS:
    """Build a fresh FAISS index with config's chunk size and embedding model."""
    print(f"  Building index: chunk={cfg['chunk_size']}, embed={cfg['embed_model'].split('/')[-1]}")

    # Load docs
    all_docs = []
    for loader_cls, glob in [(TextLoader, "**/*.txt"), (PyPDFLoader, "**/*.pdf")]:
        loader = DirectoryLoader(
            DATA_DIR, glob=glob, loader_cls=loader_cls,
            loader_kwargs={"encoding": "utf-8"} if loader_cls == TextLoader else {},
            silent_errors=True,
        )
        all_docs.extend(loader.load())

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(all_docs)
    print(f"    {len(chunks)} chunks from {len(all_docs)} docs")

    # Embed
    embeddings = HuggingFaceEmbeddings(
        model_name=cfg["embed_model"],
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.from_documents(chunks, embeddings)


def build_retriever(vectorstore: FAISS, cfg: dict):
    if cfg["search_type"] == "mmr":
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": cfg["k"], "fetch_k": cfg["fetch_k"], "lambda_mult": cfg["lambda_mult"]},
        )
    else:
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": cfg["k"]},
        )


# ── RAG chain ─────────────────────────────────────────────────────────────────

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


# ── Run one config ────────────────────────────────────────────────────────────

def run_config(cfg: dict, qa_pairs: list, llm, tokenizer) -> dict:
    print(f"\n{'─'*60}")
    print(f"  Config: {cfg['description']}")
    print(f"{'─'*60}")

    vectorstore = build_index_for_config(cfg)
    retriever   = build_retriever(vectorstore, cfg)
    chain       = build_chain(retriever, llm, tokenizer)

    f1_scores, key_hits, fallbacks, times = [], [], [], []

    for i, item in enumerate(qa_pairs, 1):
        t0 = time.time()
        try:
            prediction = chain.invoke(item["question"]).strip()
        except Exception as e:
            prediction = f"ERROR: {e}"

        elapsed = time.time() - t0
        f1      = token_overlap_f1(prediction, item["answer"])
        key     = contains_key_info(prediction, item["answer"])
        fb      = is_fallback(prediction)

        f1_scores.append(f1)
        key_hits.append(key)
        fallbacks.append(fb)
        times.append(elapsed)

        print(f"  [{i:2}/{len(qa_pairs)}] F1={f1:.2f} key={'✓' if key else '✗'} fb={'y' if fb else 'n'} ({elapsed:.1f}s)")

    n = len(qa_pairs)
    return {
        "name":         cfg["name"],
        "description":  cfg["description"],
        "avg_f1":       sum(f1_scores) / n,
        "key_info_pct": sum(key_hits) / n * 100,
        "fallback_pct": sum(fallbacks) / n * 100,
        "avg_time_s":   sum(times) / n,
        "n":            n,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tune RAG configurations.")
    parser.add_argument("--model",  default=DEFAULT_MODEL_PATH)
    parser.add_argument("--qa-file", default=DEFAULT_QA_FILE)
    parser.add_argument("--limit",  type=int, default=50,
                        help="Questions per config (default: 50)")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Run only these config names (e.g. --configs baseline k5 mpnet_k5)")
    args = parser.parse_args()

    # Load QA pairs
    with open(args.qa_file, encoding="utf-8") as f:
        all_qa = json.load(f)
    # Use a fixed sample for fair comparison — same questions for every config
    qa_pairs = all_qa[:args.limit]
    print(f"Using {len(qa_pairs)} questions for tuning.")

    # Filter configs if requested
    configs = CONFIGS
    if args.configs:
        configs = [c for c in CONFIGS if c["name"] in args.configs]
        if not configs:
            print(f"ERROR: No matching configs. Available: {[c['name'] for c in CONFIGS]}")
            sys.exit(1)

    # Load LLM once
    llm, tokenizer = load_llm(args.model)

    # Run each config
    summary_rows = []
    for cfg in configs:
        row = run_config(cfg, qa_pairs, llm, tokenizer)
        summary_rows.append(row)

    # Save detailed CSV
    Path("evaluation").mkdir(exist_ok=True)
    with open("evaluation/tuning_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    # Print comparison table
    table = (
        f"\n{'='*75}\n"
        f"  RAG TUNING RESULTS  ({len(qa_pairs)} questions each)\n"
        f"{'='*75}\n"
        f"  {'Config':<35} {'Avg F1':>7} {'Key Info':>9} {'Fallback':>9} {'Avg(s)':>7}\n"
        f"  {'─'*35} {'─'*7} {'─'*9} {'─'*9} {'─'*7}\n"
    )
    best_f1  = max(r["avg_f1"]       for r in summary_rows)
    best_key = max(r["key_info_pct"] for r in summary_rows)
    for r in summary_rows:
        f1_star  = " ◄" if r["avg_f1"]       == best_f1  else ""
        key_star = " ◄" if r["key_info_pct"] == best_key else ""
        table += (
            f"  {r['description']:<35} "
            f"{r['avg_f1']*100:>6.1f}%{f1_star:<2} "
            f"{r['key_info_pct']:>8.1f}%{key_star:<2} "
            f"{r['fallback_pct']:>8.1f}%  "
            f"{r['avg_time_s']:>6.1f}s\n"
        )
    table += f"{'='*75}\n"
    table += f"  ◄ = best in column\n"
    table += f"\n  Recommended: pick the config with highest Key Info % (most meaningful metric)\n"
    table += f"{'='*75}\n"

    print(table)

    with open("evaluation/tuning_summary.txt", "w", encoding="utf-8") as f:
        f.write(table)
    print("Saved → evaluation/tuning_results.csv")
    print("Saved → evaluation/tuning_summary.txt")

    # Print the winner
    winner = max(summary_rows, key=lambda r: r["key_info_pct"])
    print(f"\n  WINNER: {winner['description']}")
    print(f"  Update rag.py and evaluate.py with these settings:")
    cfg = next(c for c in CONFIGS if c["name"] == winner["name"])
    print(f"    embed_model  = \"{cfg['embed_model']}\"")
    print(f"    chunk_size   = {cfg['chunk_size']}")
    print(f"    search_type  = \"{cfg['search_type']}\"")
    print(f"    k            = {cfg['k']}")
    if cfg["search_type"] == "mmr":
        print(f"    fetch_k      = {cfg['fetch_k']}")
        print(f"    lambda_mult  = {cfg['lambda_mult']}")
    print()


if __name__ == "__main__":
    main()