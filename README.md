# 📚 RAG Research — Question Answering over Wikipedia Articles

> A Retrieval-Augmented Generation (RAG) pipeline that turns Wikipedia articles into a searchable vector knowledge base and answers questions **grounded only in those documents** — built with LangChain, OpenAI embeddings, ChromaDB and GPT-4o.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=flat-square&logo=openai&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6F61?style=flat-square)
![Gemini](https://img.shields.io/badge/Google-Gemini-4285F4?style=flat-square&logo=google&logoColor=white)
![Voyage AI](https://img.shields.io/badge/Voyage_AI-Embeddings-111111?style=flat-square)

---

## 📌 Overview

Large language models are powerful but can hallucinate and don't know your private or up-to-date data. **RAG** fixes this by retrieving relevant passages from a trusted document set and giving them to the model as context.

This repository is a hands-on research project exploring how to build that end to end:

- **Ingestion pipeline** – load documents, chunk them, embed them, and persist them in a vector database.
- **Retrieval pipeline** – embed a user question, fetch the most similar chunks, and have an LLM answer strictly from them.
- **Model experiments** – trying alternative embedding and LLM providers (Voyage AI, Google Gemini) and agent frameworks (LangChain Deep Agents).

The knowledge base is built from Wikipedia articles on **Google** and **Microsoft**, with earlier experiments on **Tesla, Nvidia, SpaceX** and the *"Attention Is All You Need"* paper.

---

## 🏗️ Architecture

```text
                ┌──────────────────── INGESTION ────────────────────┐
 docs/*.txt ──► DirectoryLoader ──► CharacterTextSplitter ──► OpenAI Embeddings
 (Wikipedia)     (TextLoader)        (chunk_size=500)        (text-embedding-3-small)
                                                                     │
                                                                     ▼
                                                          ChromaDB (cosine, persisted
                                                                 in db/chroma_db)
                └───────────────────────────────────────────────────┘
                                                                     ▲
                ┌──────────────────── RETRIEVAL ────────────────────┐│
 User question ─► embed query ─► similarity search (top k = 3) ──────┘
                                        │
                                        ▼
                     Grounded prompt: question + retrieved chunks
                                        │
                                        ▼
                                  GPT-4o answer
             (or "I don't have enough information…" if not in the docs)
                └───────────────────────────────────────────────────┘
```

---

## ⚙️ How It Works

### 1. Ingestion — `ingestion_pipeline.py`

| Step | Implementation |
|---|---|
| Load | `DirectoryLoader` + `TextLoader` reads every `*.txt` in `docs/` |
| Split | `CharacterTextSplitter` → ~500-character chunks |
| Embed | `OpenAIEmbeddings(model="text-embedding-3-small")` |
| Store | `Chroma.from_documents(...)` persisted to `db/chroma_db` with **cosine** distance (`hnsw:space = cosine`) |

### 2. Retrieval & Generation — `retriveal_pipeline.py`

1. Re-opens the persisted Chroma collection with the same embedding model.
2. Reads a question from the terminal.
3. Retrieves the **top 3** most relevant chunks (`as_retriever(search_kwargs={"k": 3})`).
4. Builds a grounded prompt that tells the model to answer **only** from the supplied documents, and to say it doesn't know otherwise.
5. Sends it to **GPT-4o** via `ChatOpenAI` and prints the answer.

### 3. Experiments

| File | What it explores |
|---|---|
| `RAG.py` | Voyage AI embeddings (`voyage-4-large`) as an alternative embedding provider |
| `gemini.py` | Calling Google Gemini through the `google-genai` SDK |
| `langchain.py` | A tool-using agent built with LangChain **Deep Agents** (`create_deep_agent`) on Gemini |

---

## 📁 Project Structure

```text
.
├── ingestion_pipeline.py   # Load → chunk → embed → store in Chroma
├── retriveal_pipeline.py   # Query → retrieve top-k → grounded GPT-4o answer
├── RAG.py                  # Voyage AI embedding experiment
├── gemini.py               # Google Gemini experiment
├── langchain.py            # LangChain Deep Agents experiment
├── docs/                   # Active knowledge base (Google, Microsoft Wikipedia articles)
├── old_docs/               # Earlier corpora (Tesla, Nvidia, SpaceX, Transformer paper)
├── db/chroma_db/           # Persisted Chroma vector store
└── .env.example            # Required API keys
```

---

## 🚀 Getting Started

```bash
git clone https://github.com/akashrajput9/rag_research_wikipedia.git
cd rag_research_wikipedia
python -m venv .venv && source .venv/bin/activate

pip install langchain langchain-openai langchain-chroma langchain-community \
            langchain-text-splitters chromadb python-dotenv
# optional, for the experiments
pip install voyageai google-genai deepagents

cp .env.example .env   # then fill in OPENAI_API_KEY (and others if needed)
```

```bash
# 1) Build the vector store from docs/
python ingestion_pipeline.py

# 2) Ask questions
python retriveal_pipeline.py
# Enter query: Who founded Google and when?
```

To use your own knowledge base, drop `.txt` files into `docs/` and re-run the ingestion step.

### Environment variables

| Variable | Used for |
|---|---|
| `OPENAI_API_KEY` | Embeddings + GPT-4o (core pipeline) |
| `VOYAGE_API_KEY` | Voyage AI embedding experiment |
| `GOOGLE_STUDIO_API` | Gemini experiments |
| `NOMIC_API_KEY` | Reserved for Nomic embedding experiments |

---

## 🧭 Roadmap

- [ ] Compare chunking strategies (`RecursiveCharacterTextSplitter`, chunk overlap) on answer quality
- [ ] Benchmark OpenAI vs Voyage vs Nomic embeddings on the same corpus
- [ ] Return source citations alongside each answer
- [ ] PDF ingestion (e.g. the Transformer paper in `old_docs/`)
- [ ] Expose retrieval as a tool for an agent (Deep Agents / CrewAI)

## 🛠️ Tech Stack

`Python` · `LangChain` · `OpenAI (text-embedding-3-small, GPT-4o)` · `ChromaDB` · `Voyage AI` · `Google Gemini` · `LangChain Deep Agents` · `python-dotenv`

---

<sub>Built by <a href="https://github.com/akashrajput9">Akash Ahmed</a> — AI Systems & Agentic Automation Engineer.</sub>
