# 221B Baker Street Retrieval

A RAG (Retrieval-Augmented Generation) application that lets you ask questions about Arthur Conan Doyle's Sherlock
Holmes novels and receive answers with exact source citations.

## What it does

You ask a question - the app retrieves the most relevant passages from the novels, feeds them to an LLM, and returns a
grounded answer with inline citations pointing back to the specific book and chapter each fact came from. It will not
speculate or draw on outside knowledge; every claim in the answer is backed by a retrieved source.

The UI is a simple Streamlit web app with a text input, a set of sample questions to get you started, and expandable
source cards showing the raw retrieved passages beneath each answer.

## Tech stack

| Layer         | Technology                                                                                     |
|---------------|------------------------------------------------------------------------------------------------|
| UI            | [Streamlit](https://streamlit.io/)                                                             |
| LLM           | OpenAI `gpt-4o` via [LangChain](https://www.langchain.com/)                                    |
| Embeddings    | OpenAI `text-embedding-ada-002` (`OpenAIEmbeddings`)                                           |
| Vector store  | [ChromaDB](https://www.trychroma.com/) (persisted locally in `chroma_db/`)                     |
| Chunking      | LangChain `SemanticChunker` (splits by semantic similarity rather than fixed token size)       |
| Orchestration | LangChain (`langchain-core`, `langchain-openai`, `langchain-chroma`, `langchain-experimental`) |

## Data

The book data is **not included in this repository**. The texts are public domain and are freely available
from [Project Gutenberg](https://www.gutenberg.org/). The four novels used are:

- *A Study in Scarlet*
- *The Sign of the Four*
- *The Hound of the Baskervilles*
- *The Valley of Fear*

Download the plain-text `.txt` versions from Project Gutenberg, place them in the `books/` directory, and name them:

```
books/study-in-scarlet.txt
books/sign-of-the-four.txt
books/hound-of-the-baskervilles.txt
books/valley-of-fear.txt
```

**The following changes were manually added to the downloaded files in order to keep the formatting consistent
across different editions:**

Each file should include a small header at the top in the following format so that metadata is extracted correctly
during ingestion :

```
TITLE: <book title>
AUTHOR: <author name>
```

Chapter headings should follow the pattern `CHAPTER <number>`, with an optional `C NAME: <chapter title>` line
immediately after.

## Setup

1. **Install dependencies**

   ```bash
   pip install streamlit langchain langchain-openai langchain-chroma langchain-experimental chromadb
   ```

2. **Set your OpenAI API key**

   ```bash
   export OPENAI_API_KEY=your_key_here
   ```

3. **Ingest the books and build the vector store** (run once, for more information follow instructions in main.py)

   In `main.py`, uncomment the `data_processing_and_vector_store_creation()` call and run:

   ```bash
   python main.py
   ```

   This parses the books by chapter, applies semantic chunking, embeds each chunk, and persists the vector store to
   `chroma_db/`. Only run this once — re-running appends duplicates to the database.

4. **Launch the app**

   ```bash
   streamlit run rag_app.py
   ```

## How it works

1. The books are parsed into per-chapter `Document` objects, each carrying metadata (title, author, chapter number,
   chapter title).
2. `SemanticChunker` splits each chapter into semantically coherent chunks using embedding similarity.
3. The chunks are embedded and stored in a local ChromaDB vector store.
4. At query time, the top 10 most relevant chunks are retrieved and passed alongside the question to `gpt-4o` using a
   strict citation-only prompt. (this will probably be updated to a newer model, however it's been sufficient for
   initial experiments)
5. The model returns an answer with inline `[n]` citations; the app renders both the answer and the raw source passages.