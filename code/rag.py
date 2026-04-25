from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ── 1. Load the saved FAISS index ───────────────────────────────────────────
print("Loading index...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
print(f"Index loaded: {vectorstore.index.ntotal} vectors")

# ── 2. Create retriever ─────────────────────────────────────────────────────
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.6},
)

# ── 3. Format retrieved docs ────────────────────────────────────────────────
def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata['source']}]\n{d.page_content}"
        for d in docs
    )

# ── 4. RAG prompt ───────────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_template("""You are a helpful assistant for international students at UL FRI (Faculty of Computer and Information Science, University of Ljubljana, Slovenia).
Answer the question using ONLY the context below.
If the answer is not in the context, say exactly: "I don't have that information in my knowledge base. Please check the UL FRI website or contact the student office directly."
Be concise and friendly.

Context:
{context}

Question: {question}
Answer:""")

# ── 5. LLM ──────────────────────────────────────────────────────────────────
print("Loading LLM...")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=GEMINI_API_KEY,
    temperature=0.1,
)
print("LLM ready.")

# ── 6. Build the chain ──────────────────────────────────────────────────────
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ── 7. Debug helper — shows retrieved chunks ────────────────────────────────
def ask_with_sources(question: str):
    retrieved = retriever.invoke(question)
    context = format_docs(retrieved)
    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "question": question
    })
    print(f"\nA: {answer}")
    print("\n[Retrieved sources:]")
    for i, doc in enumerate(retrieved):
        print(f"  [{i+1}] {doc.metadata['source']} — {doc.page_content[:80]}...")

# ── 8. Predefined test questions ────────────────────────────────────────────
TEST_QUESTIONS = [
    "What is the application deadline for non-EU students?",
    "How much does tuition cost for international students?",
    "Do I need a visa to study at UL FRI?",
    "What topics are covered in the Master's programme?",
    "How much does student accommodation cost?",
]

# ── 9. Main entry point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nChoose mode:")
    print("  1 — Run predefined test questions")
    print("  2 — Interactive chat")
    mode = input("Enter 1 or 2: ").strip()

    if mode == "1":
        for q in TEST_QUESTIONS:
            print(f"\nQ: {q}")
            print("-" * 50)
            ask_with_sources(q)

    elif mode == "2":
        print("\nUL FRI Student Chatbot — type 'quit' to exit, 'debug' to toggle source display\n")
        show_sources = False
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() == "quit":
                print("Goodbye!")
                break
            if user_input.lower() == "debug":
                show_sources = not show_sources
                print(f"[Source display: {'ON' if show_sources else 'OFF'}]")
                continue

            if show_sources:
                ask_with_sources(user_input)
            else:
                answer = rag_chain.invoke(user_input)
                print(f"Bot: {answer}\n")
    else:
        print("Invalid choice.")