# Retrieval-Augmented Generation (RAG) Pipeline

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that processes text from PDF files, retrieves relevant documents using dense and sparse retrieval methods, and generates natural language answers using a language model. It integrates tools like **FAISS**, **BM25**, **Sentence Transformers**, and the **Ollama API**.

---

## Features

- **PDF Text Extraction:** Extracts text from PDFs for processing.
- **Dense Retrieval with FAISS:** Leverages Sentence Transformers for embedding-based search.
- **Sparse Retrieval with BM25:** Uses term-frequency-based scoring for keyword-based search.
- **Ollama API Integration:** Generates answers by combining retrieved context with a query.
- **Customizable Pipeline:** Allows fine-tuning of retrieval and generation parameters.

---

## How It Works

1. **PDF Processing:**
   - The application uses **PyPDF2** to extract text from a PDF file.
   - Extracted text is split into smaller chunks, which are treated as individual documents for retrieval.

2. **Dense Retrieval with FAISS:**
   - Uses **Sentence Transformers** to create embeddings for each document.
   - A FAISS index is built for efficient nearest-neighbor search.
   - When a query is provided, its embedding is compared against the document embeddings to retrieve the most relevant matches.

3. **Sparse Retrieval with BM25:**
   - Tokenizes documents and queries for sparse retrieval.
   - Uses **BM25Okapi** to rank documents based on term frequency and inverse document frequency.
   - Retrieves documents that best match the query's keywords.

4. **Combining Dense and Sparse Retrieval:**
   - Results from FAISS and BM25 are combined to create a comprehensive context.
   - The top-ranked results from both methods are merged for further processing.

5. **Query Generation with Ollama API:**
   - The combined context and user query are sent to the **Ollama API**.
   - The API generates a natural language response based on the provided context and query.
   - The response is streamed back and processed into a final answer.

6. **End-to-End Retrieval-Augmented Generation (RAG) Pipeline:**
   - Retrieves relevant information from the document.
   - Combines dense and sparse retrieval methods for improved accuracy.
   - Generates a coherent answer using the retrieved context and a powerful language model.
