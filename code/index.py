from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = "test_data/"  # rename to "data/" for final version

# ── 1. Load .txt files ──────────────────────────────────────────────────────
print("Loading .txt documents...")
txt_loader = DirectoryLoader(
    DATA_DIR,
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
txt_docs = txt_loader.load()
print(f"  Loaded {len(txt_docs)} .txt files")

# ── 2. Load .pdf files ──────────────────────────────────────────────────────
print("Loading .pdf documents...")
pdf_loader = DirectoryLoader(
    DATA_DIR,
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
)
pdf_docs = pdf_loader.load()
print(f"  Loaded {len(pdf_docs)} .pdf pages")

# ── 3. Combine all docs ─────────────────────────────────────────────────────
all_docs = txt_docs + pdf_docs
print(f"Total documents loaded: {len(all_docs)}")

if not all_docs:
    print("ERROR: No documents found in", DATA_DIR)
    exit(1)

for d in all_docs:
    print(f"  - {d.metadata['source']} ({len(d.page_content)} chars)")

# ── 4. Split into chunks ────────────────────────────────────────────────────
print("\nSplitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    add_start_index=True,
)
chunks = splitter.split_documents(all_docs)
print(f"Created {len(chunks)} chunks")
print(f"Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")

# ── 5. Load embedding model ─────────────────────────────────────────────────
print("\nLoading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
print("Embedding model loaded.")

# ── 6. Build FAISS index ────────────────────────────────────────────────────
print("\nBuilding FAISS index...")
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")
print(f"Index saved ({vectorstore.index.ntotal} vectors)")

# ── 7. Sanity check ─────────────────────────────────────────────────────────
print("\nSanity check — searching for 'application deadline':")
results = vectorstore.similarity_search("application deadline", k=2)
for i, r in enumerate(results):
    print(f"\n[{i+1}] Source: {r.metadata['source']}")
    print(r.page_content[:200])

print("\nDone! Run rag.py next.")