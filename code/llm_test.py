"""
llm_test.py — Verify that an HPC-hosted LLM loads and generates text correctly.

Run this BEFORE rag.py to confirm the model works on your SLURM GPU allocation.

Usage:
    python code/llm_test.py                  # test Llama 3.1 (default)
    python code/llm_test.py --model <path>   # test any model by full path
"""

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.generation").setLevel(logging.ERROR)
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

import torch

# ── Confirmed model paths on ARNES HPC ──────────────────────────────────────
LLAMA31_PATH = (
    "/d/hpc/projects/onj_fri/hungover_pandas/models/llama-3.1-8b-instruct"
)
DEFAULT_MODEL = LLAMA31_PATH

ERASMUS_QUESTIONS = [
    "What documents do I need to apply for an Erasmus exchange?",
    "How do I find partner universities for Erasmus at UL FRI?",
    "What financial support is available for Erasmus students?",
]


def get_load_strategy() -> dict:
    """
    Choose loading strategy based on GPU compute capability.
    - CC >= 8.0  → 4-bit NF4 (H100, A100, ...)
    - CC >= 7.5  → 8-bit     (T4, ...)
    - CC == 7.0  → float16   (V100 — bitsandbytes broken below CC 7.5)
    - No GPU     → float32 on CPU
    """
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


def load_model(model_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n  Path : {model_path}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            vram = torch.cuda.get_device_properties(i).total_memory / 1e9
            cc   = torch.cuda.get_device_capability(i)
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}  ({vram:.1f} GB, CC {cc[0]}.{cc[1]})")

    load_kwargs = get_load_strategy()

    print("\n  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("  Loading model weights... (~1-3 min on H100)")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    if hasattr(model, "generation_config") and hasattr(model.generation_config, "max_length"):
        model.generation_config.max_length = None
    print(f"  Model loaded in {time.time()-t0:.1f}s")
    return model, tokenizer


def make_llama31_prompt(question: str) -> str:
    """
    Llama 3.1 uses the <|begin_of_text|> chat template format.
    apply_chat_template handles this automatically.
    """
    return question  # tokenizer.apply_chat_template handles formatting in run_tests


def run_tests(model_path: str, model, tokenizer) -> bool:
    from transformers import pipeline
    from langchain_huggingface import HuggingFacePipeline
    from langchain_core.runnables import RunnableLambda
    from langchain_core.output_parsers import StrOutputParser

    # Get the end-of-turn token id for clean stopping
    terminators = [tokenizer.eos_token_id]
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_id and eot_id != tokenizer.unk_token_id:
        terminators.append(eot_id)

    hf_pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        max_new_tokens=256,
        do_sample=False,
        repetition_penalty=1.1,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
    )
    llm = HuggingFacePipeline(pipeline=hf_pipe)

    system = (
        "You are a helpful assistant for students at UL FRI "
        "who want to go on Erasmus exchange. Answer concisely in English."
    )

    def make_prompt(question: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": question},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    chain = RunnableLambda(make_prompt) | llm | StrOutputParser()

    print(f"\n{'='*60}")
    print("  LLM Generation Test (no RAG — raw model output)")
    print(f"{'='*60}")

    all_passed = True
    for i, q in enumerate(ERASMUS_QUESTIONS, 1):
        print(f"\n[Q{i}] {q}")
        print("-" * 55)
        try:
            t0 = time.time()
            answer = chain.invoke(q).strip()
            print(f"[A]  {answer}")
            print(f"     ({time.time()-t0:.1f}s)")
            if len(answer) < 10:
                print("  WARNING: Answer too short.")
                all_passed = False
        except Exception as e:
            print(f"  ERROR: {e}")
            all_passed = False

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Test HPC LLM before running RAG chatbot.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Full path to model directory")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"\nERROR: Model not found at: {args.model}")
        sys.exit(1)

    model_label = Path(args.model).name
    print(f"\n{'='*60}")
    print(f"  HPC LLM Test — {model_label}")
    print(f"{'='*60}")

    model, tokenizer = load_model(args.model)
    passed = run_tests(args.model, model, tokenizer)

    print(f"\n{'='*60}")
    if passed:
        print("  RESULT: All tests passed — ready to run rag.py")
        print(f"\n  Next step:")
        print(f"    python code/rag.py --mode test")
    else:
        print("  RESULT: Some tests had issues — check output above.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()