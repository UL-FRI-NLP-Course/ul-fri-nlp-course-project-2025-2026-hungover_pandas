# Domain-Specific AI Assistant for the UL FRI Student Office

Course project for **FRI Natural Language Processing 2025/26**.

This repository contains the report and supporting materials for a domain-specific
question-answering assistant that helps students find accurate administrative
information (enrollment, exams, study progression, deadlines, certificates, etc.)
from official UL FRI and UL sources.

The project is based on retrieval-augmented generation (RAG): relevant passages
are retrieved from curated institutional documents, then used to ground answer
generation with a language model.

## Authors

- Aleksa Sibinovic
- Hristijan Milanovski
- Sara Ivanovska

## Repository structure

- `README.md` - project overview and reproduction instructions
- `report/` - LaTeX report and bibliography
- `code/` - folder for scripts

## Planned implementation pipeline

1. Collect official web pages and PDF documents from UL FRI and UL.
2. Extract and clean text; keep metadata (source URL, retrieval date).
3. Chunk documents with overlap for retrieval.
4. Build a retriever over chunks (embeddings + similarity search).
5. Generate grounded answers with an LLM using retrieved context.
6. Evaluate answer quality and factual grounding.

## Notes


