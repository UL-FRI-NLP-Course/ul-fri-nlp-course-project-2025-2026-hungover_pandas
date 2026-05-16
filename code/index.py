"""
index.py — Build the FAISS vector index from your knowledge base documents.

Run this once (or whenever your data changes) before running rag.py or llm_test.py.

Usage:
    python code/index.py
    python code/index.py --data-dir data/ --index-dir faiss_index/
"""

import argparse
import sys
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ── Config defaults ──────────────────────────────────────────────────────────
DEFAULT_DATA_DIR  = "data/"
DEFAULT_INDEX_DIR = "faiss_index/"
EMBED_MODEL       = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE        = 500
CHUNK_OVERLAP     = 50


def load_documents(data_dir: str) -> list:
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"ERROR: Data directory '{data_dir}' does not exist.")
        sys.exit(1)

    all_docs = []

    # .txt files
    txt_loader = DirectoryLoader(
        data_dir, glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        silent_errors=True,
    )
    txt_docs = txt_loader.load()
    print(f"  Loaded {len(txt_docs)} .txt file(s)")
    all_docs.extend(txt_docs)

    # .pdf files
    pdf_loader = DirectoryLoader(
        data_dir, glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        silent_errors=True,
    )
    pdf_docs = pdf_loader.load()
    print(f"  Loaded {len(pdf_docs)} .pdf page(s)")
    all_docs.extend(pdf_docs)

    return all_docs


def build_index(data_dir: str, index_dir: str) -> None:
    print(f"\n{'='*55}")
    print("  FAISS Index Builder")
    print(f"{'='*55}")
    print(f"  Data dir  : {data_dir}")
    print(f"  Index dir : {index_dir}")
    print(f"  Embed model: {EMBED_MODEL}")
    print(f"{'='*55}\n")

    # 1. Load documents
    print("[ 1/5 ] Loading documents...")
    all_docs = load_documents(data_dir)

    if not all_docs:
        print(f"ERROR: No .txt or .pdf files found in '{data_dir}'")
        sys.exit(1)

    print(f"\n  Total pages/files loaded: {len(all_docs)}")
    for d in all_docs:
        src = d.metadata.get("source", "unknown")
        print(f"    - {src}  ({len(d.page_content):,} chars)")

    # 2. Chunk
    print(f"\n[ 2/5 ] Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(all_docs)
    avg = sum(len(c.page_content) for c in chunks) // len(chunks)
    print(f"  Created {len(chunks)} chunks  (avg {avg} chars)")

    # 3. Load embedding model
    print(f"\n[ 3/5 ] Loading embedding model: {EMBED_MODEL} ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},  # embeddings are fast on CPU
        encode_kwargs={"normalize_embeddings": True},
    )
    print("  Embedding model ready.")

    # 4. Build FAISS index
    print(f"\n[ 4/5 ] Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_dir)
    print(f"  Saved index with {vectorstore.index.ntotal} vectors → '{index_dir}'")

    # 5. Sanity check
    print(f"\n[ 5/5 ] Sanity check — top-2 results for 'Erasmus exchange application':")
    results = vectorstore.similarity_search("Erasmus exchange application", k=2)
    if results:
        for i, r in enumerate(results):
            src = r.metadata.get("source", "?")
            print(f"  [{i+1}] {src}")
            print(f"       {r.page_content[:150].strip()}...")
    else:
        print("  (No results — is your data Erasmus-related?)")

    print(f"\n  Done! Run  python code/rag.py  next.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index from documents.")
    parser.add_argument("--data-dir",  default=DEFAULT_DATA_DIR,  help="Folder with .txt/.pdf files")
    parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR, help="Where to save the FAISS index")
    args = parser.parse_args()
    build_index(args.data_dir, args.index_dir)