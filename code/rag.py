"""
rag.py — Erasmus Chatbot for UL FRI students (HPC version).

Uses Llama-3.1-8B-Instruct loaded locally on ARNES HPC with 4-bit quantisation.
FAISS index, embeddings, retriever, and chain structure unchanged from Gemini version.

Usage:
    python code/rag.py                   # Llama 3.1, interactive chat
    python code/rag.py --mode test       # predefined Erasmus test questions
    python code/rag.py --mode chat       # interactive chat (default)
    python code/rag.py --model <path>    # override model path
"""

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

# Suppress noisy but harmless warnings from transformers / bitsandbytes
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.generation").setLevel(logging.ERROR)
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

import torch
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ── Model path ───────────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = (
    "/d/hpc/projects/onj_fri/hungover_pandas/models/llama-3.1-8b-instruct"
)
DEFAULT_INDEX_DIR = "faiss_index/"
EMBED_MODEL       = "sentence-transformers/all-MiniLM-L6-v2"

SYSTEM_PROMPT = (
    "You are a friendly and helpful Erasmus exchange assistant for students at "
    "UL FRI (Faculty of Computer and Information Science, University of Ljubljana, Slovenia). "
    "Answer questions using ONLY the context provided below. "
    "If the answer is not in the context, say exactly: "
    "\"I don't have that information in my knowledge base. "
    "Please check the UL FRI website or contact the International Office directly.\" "
    "Be concise, accurate, and always respond in English."
)

TEST_QUESTIONS = [
    "What is the application deadline for Erasmus exchange?",
    "Which partner universities are available for computer science students?",
    "Do I need a visa to study in an EU country as an Erasmus student?",
    "What financial support is available for Erasmus students?",
    "How do I get my credits recognised after returning from Erasmus?",
]


# ── GPU / loading strategy ────────────────────────────────────────────────────

def get_load_strategy() -> dict:
    if not torch.cuda.is_available():
        print("  WARNING: No GPU — inference will be very slow on CPU.")
        return {"dtype": torch.float32}

    major, minor = torch.cuda.get_device_capability(0)
    cc = major + minor / 10

    if cc >= 8.0:
        print(f"  CC {major}.{minor} → 4-bit NF4 quantisation")
        from transformers import BitsAndBytesConfig
        return {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            "device_map": "auto",
        }
    elif cc >= 7.5:
        print(f"  CC {major}.{minor} → 8-bit quantisation")
        from transformers import BitsAndBytesConfig
        return {
            "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
            "device_map": "auto",
        }
    else:
        print(f"  CC {major}.{minor} (V100) → float16")
        return {"dtype": torch.float16, "device_map": "auto"}


# ── Vector store ──────────────────────────────────────────────────────────────

def load_vectorstore(index_dir: str) -> FAISS:
    if not Path(index_dir).exists():
        print(f"ERROR: FAISS index not found at '{index_dir}'.")
        print("       Run  python code/index.py  first.")
        sys.exit(1)

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
    return vectorstore


# ── LLM ───────────────────────────────────────────────────────────────────────

def load_llm(model_path: str):
    """Load model + tokenizer, return (HuggingFacePipeline, tokenizer)."""
    print(f"\nLoading LLM: {Path(model_path).name}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            vram = torch.cuda.get_device_properties(i).total_memory / 1e9
            cc   = torch.cuda.get_device_capability(i)
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}  ({vram:.1f} GB, CC {cc[0]}.{cc[1]})")

    load_kwargs = get_load_strategy()

    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("  Loading model weights... (~1-3 min on H100)")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    # Remove max_length from generation_config to avoid conflict with max_new_tokens
    if hasattr(model, "generation_config") and hasattr(model.generation_config, "max_length"):
        model.generation_config.max_length = None
        model.generation_config.max_new_tokens = 512
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Llama 3.1 end-of-turn token for clean stopping
    terminators = [tokenizer.eos_token_id]
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_id and eot_id != tokenizer.unk_token_id:
        terminators.append(eot_id)

    hf_pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.1,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
    )

    llm = HuggingFacePipeline(pipeline=hf_pipe)
    print("  LLM ready.\n")
    return llm, tokenizer


# ── RAG chain ─────────────────────────────────────────────────────────────────

def format_docs(docs: list) -> str:
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in docs
    )


def make_prompt(tokenizer, inputs: dict) -> str:
    """Format prompt using Llama 3.1 chat template."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context:\n{inputs['context']}\n\n"
                f"Question: {inputs['question']}"
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_rag_chain(retriever, llm, tokenizer):
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RunnableLambda(lambda inputs: make_prompt(tokenizer, inputs))
        | llm
        | StrOutputParser()
    )


# ── Debug helper ──────────────────────────────────────────────────────────────

def ask_with_sources(question: str, retriever, llm, tokenizer) -> str:
    retrieved = retriever.invoke(question)
    context   = format_docs(retrieved)
    answer = (
        RunnableLambda(lambda inputs: make_prompt(tokenizer, inputs))
        | llm
        | StrOutputParser()
    ).invoke({"context": context, "question": question})

    answer = answer.strip()
    print(f"\nA: {answer}")
    print("\n[Retrieved sources:]")
    for i, doc in enumerate(retrieved):
        src = doc.metadata.get("source", "unknown")
        print(f"  [{i+1}] {src}")
        print(f"       {doc.page_content[:100].strip()}...")
    return answer


# ── Run modes ─────────────────────────────────────────────────────────────────

def run_test_mode(retriever, llm, tokenizer):
    print(f"\n{'='*60}")
    print("  Test Mode — Predefined Erasmus Questions")
    print(f"{'='*60}\n")
    for q in TEST_QUESTIONS:
        print(f"Q: {q}")
        print("-" * 55)
        ask_with_sources(q, retriever, llm, tokenizer)
        print()


def run_chat_mode(rag_chain, retriever, llm, tokenizer):
    print(f"\n{'='*60}")
    print("  Erasmus Chatbot — UL FRI")
    print("  Commands:  'debug' — toggle source display | 'quit' — exit")
    print(f"{'='*60}\n")

    show_sources = False
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if user_input.lower() == "debug":
            show_sources = not show_sources
            print(f"[Source display: {'ON' if show_sources else 'OFF'}]")
            continue

        # Greeting detection — respond friendly without hitting RAG
        greetings = {"hey", "hi", "hello", "sup", "hola", "greetings"}
        if user_input.lower().strip("!.,") in greetings:
            print("Bot: Hi! I'm the UL FRI Erasmus assistant. Ask me anything about Erasmus exchange — deadlines, partner universities, financial support, visas, and more.\n")
            continue

        t0 = time.time()
        if show_sources:
            ask_with_sources(user_input, retriever, llm, tokenizer)
        else:
            answer = rag_chain.invoke(user_input)
            print(f"Bot: {answer.strip()}")
        print(f"     ({time.time()-t0:.1f}s)\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Erasmus RAG Chatbot — HPC version")
    parser.add_argument("--model",     default=DEFAULT_MODEL_PATH, help="Path to model directory")
    parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    parser.add_argument("--mode",      default="chat", choices=["chat", "test"])
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"ERROR: Model not found at: {args.model}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Erasmus RAG Chatbot — UL FRI (HPC)")
    print(f"  Model : {Path(args.model).name}")
    print(f"  Mode  : {args.mode}")
    print(f"{'='*60}")

    vectorstore    = load_vectorstore(args.index_dir)
    retriever      = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.6},
    )
    llm, tokenizer = load_llm(args.model)
    rag_chain      = build_rag_chain(retriever, llm, tokenizer)

    if args.mode == "test":
        run_test_mode(retriever, llm, tokenizer)
    else:
        run_chat_mode(rag_chain, retriever, llm, tokenizer)


if __name__ == "__main__":
    main()