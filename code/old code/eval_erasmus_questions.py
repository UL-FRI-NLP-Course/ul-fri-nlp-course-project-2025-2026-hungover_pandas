"""Generate `tests_Q&A/results.jsonl` by running sample questions through RAG.

Parses `tests_Q&A/sample.txt` (questions and EXPECTED answers), runs each
question through the RAG pipeline (loads FAISS index and LLM), and writes a
JSONL file with fields used by `eval_with_gemini.py`.
"""

import re
import json
import time
import argparse
import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from config import get_embedding_config

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def parse_sample(file_path: str) -> list:
    """Parse questions and expected answers from sample.txt.

    Returns a list of dicts: {q_num, question, expected_answer}
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip() for ln in f]

    items = []
    q_num = None
    question = None
    expected = ""

    q_re = re.compile(r"^(Q\d+):\s*(.*)")
    # accept either EXPECTED: or A: as the answer marker (case-insensitive)
    exp_re = re.compile(r"^(?:EXPECTED:|A:)\s*(.*)", re.IGNORECASE)

    for ln in lines:
        if not ln:
            continue
        m = q_re.match(ln)
        if m:
            # flush previous
            if q_num and question:
                items.append({"q_num": q_num, "question": question, "expected_answer": expected})
            q_num = m.group(1)
            question = m.group(2).strip()
            expected = ""
            continue

        m2 = exp_re.match(ln)
        if m2 and q_num:
            expected = m2.group(1).strip()

    # final flush
    if q_num and question:
        items.append({"q_num": q_num, "question": question, "expected_answer": expected})

    return items


def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source','unknown')}]\n{d.page_content}"
        for d in docs
    )


def parse_range(range_str: str, total: int) -> set:
    """Parse a range string like '1-5', '3,5,7' or '2' into a set of 1-based indices.

    If range_str is falsy or 'all', returns indices 1..total.
    """
    if not range_str:
        return set(range(1, total + 1))
    rs = range_str.strip()
    if rs.lower() == "all":
        return set(range(1, total + 1))

    indices = set()
    parts = [p.strip() for p in rs.split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            try:
                a, b = p.split("-", 1)
                a_i = int(a)
                b_i = int(b)
                if a_i <= b_i:
                    indices.update(range(a_i, b_i + 1))
                else:
                    indices.update(range(b_i, a_i + 1))
            except ValueError:
                continue
        else:
            try:
                indices.add(int(p))
            except ValueError:
                continue

    # clamp to valid range
    return {i for i in indices if 1 <= i <= total}


def build_rag():
    cfg = get_embedding_config()
    print(f"Loading embeddings (model={cfg['model_name']}, device={cfg['device']})...")
    embeddings = HuggingFaceEmbeddings(
        model_name=cfg["model_name"],
        model_kwargs={"device": cfg["device"]},
        encode_kwargs=cfg["encode_kwargs"],
    )

    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.6},
    )

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant for international students at UL FRI (Faculty of Computer and Information Science, University of Ljubljana, Slovenia).
Answer the question using ONLY the context below.
If the answer is not in the context, say exactly: "I don't have that information in my knowledge base. Please check the UL FRI website or contact the student office directly."
Be concise and friendly.

Context:
{context}

Question: {question}
Answer:"""
    )

    if not GEMINI_API_KEY:
        print("[ERROR] GEMINI_API_KEY not set. Please set environment variable or add to .env")
        raise SystemExit(1)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=GEMINI_API_KEY,
        temperature=0.1,
        max_retries=0,
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return retriever, rag_chain, prompt, llm


def run(questions, retriever, rag_chain, prompt, llm, out_path, sleep: float = 0.5):
    """Run RAG on `questions` and append each JSON result to `out_path`.

    Results are appended incrementally (one JSON line per question).
    """
    results = []
    for idx, q in enumerate(questions, start=1):
        q_num = q.get("q_num", f"Q{idx}")
        question = q.get("question", "")
        expected = q.get("expected_answer", "")

        try:
            # retrieve documents
            retrieved = retriever.invoke(question)
            retrieved_sources = [
                {"source": d.metadata.get("source", ""), "text": d.page_content}
                for d in retrieved
            ]

            # generate answer using explicit context
            context = format_docs(retrieved)
            model_answer = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})

            res = {
                "q_num": q_num,
                "question": question,
                "expected_answer": expected,
                "model_answer": model_answer,
                "retrieved_sources": retrieved_sources,
                "num_sources": len(retrieved_sources),
            }
            print(f"[{idx}/{len(questions)}] {q_num} — OK")
        except Exception as e:
            print(f"[{idx}/{len(questions)}] {q_num} — ERROR: {e}")
            res = {
                "q_num": q_num,
                "question": question,
                "expected_answer": expected,
                "model_answer": "",
                "retrieved_sources": [],
                "num_sources": 0,
                "error": True,
                "error_msg": str(e),
            }

        results.append(res)
        # append this single result to the output JSONL to avoid losing progress
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

        if idx < len(questions):
            time.sleep(sleep)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run sample questions through RAG and save JSONL results")
    parser.add_argument("--input", default=None, help="Sample questions file (overrides --dataset)")
    parser.add_argument("--dataset", choices=["sample", "kickoff"], default=None, help="Named dataset to use (sample or kickoff). If provided, overrides default input path.")
    parser.add_argument("--range", default=None, help="Question range to run (e.g. '1-5', '3,5,7', or 'all'). 1-based indices.")
    parser.add_argument("--output", default="tests_Q&A/results.jsonl", help="Output JSONL file (default: tests_Q&A/results.jsonl)")
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds to sleep between queries (default: 0.5)")
    args = parser.parse_args()

    # determine input path: explicit --input wins; otherwise choose by --dataset or default to sample
    if args.input:
        input_path = args.input
    else:
        if args.dataset == "kickoff":
            input_path = os.path.join("tests_Q&A", "kickoff_official_questions.txt")
        else:
            # default to sample if dataset not provided
            input_path = os.path.join("tests_Q&A", "sample.txt")

    if not os.path.exists(input_path):
        print(f"[ERROR] Input file not found: {input_path}")
        raise SystemExit(1)

    print(f"[INFO] Parsing questions from {input_path}...")
    questions = parse_sample(input_path)
    print(f"[INFO] Found {len(questions)} questions")

    # filter by range if requested
    if args.range:
        selected = parse_range(args.range, len(questions))
        if not selected:
            print(f"[ERROR] Range '{args.range}' did not select any valid question indices (1..{len(questions)})")
            raise SystemExit(1)
        filtered = [q for i, q in enumerate(questions, start=1) if i in selected]
        print(f"[INFO] Running {len(filtered)} questions from requested range: {sorted(selected)}")
        questions = filtered

    print("[INFO] Building RAG pipeline and loading FAISS index...")
    retriever, rag_chain, prompt, llm = build_rag()

    print(f"[INFO] Running {len(questions)} questions and appending results to {args.output}")
    run(questions, retriever, rag_chain, prompt, llm, args.output, sleep=args.sleep)


if __name__ == "__main__":
    main()
