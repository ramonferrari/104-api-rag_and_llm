# 📚 ExtrAI - High-Performance RAG System

### A Robust Framework for Structured Knowledge Retrieval & Generation

This project implements a **Retrieval-Augmented Generation (RAG)** system engineered to process and analyze large-scale technical corpora. While validated using a dataset of **150+ structured scientific articles** (e.g., Technological Paradigms, Moore’s Law, and Difusion of Innovations), the architecture is designed to be modular and adaptable to various structured data types.

The system enables complex, cross-lingual queries, allowing users to interact with English-centric knowledge bases using native-language prompts with high semantic precision.

---

## 🏗️ System Architecture

The pipeline is built on a modern NLP stack optimized for high-performance GPU environments:

1. **Structured Parsing:** A custom header-aware parser (`src/parser.py`) that segments documents into logical units (e.g., Abstract, Methodology, Results) to preserve context.
2. **Cross-Lingual Embeddings:** Utilizes the **BAAI/bge-m3** model, supporting over 100 languages and mapping semantic intent across the Portuguese-English gap with an 8192-token context window.
3. **Vector Database:** **ChromaDB** for high-performance, persistent vector storage and similarity search.
4. **Orchestration:** **LangChain** for component integration and retrieval logic.
5. **Hybrid LLM Integration:** A dual-provider setup utilizing **AWS Claude 3/3.5** (via Bedrock) and **Azure GPT-4o** through a secure internal API Gateway.

---

## 📂 Project Structure

The directory is organized for portability and environment independence:

```text
rag_system/
├── data/               # Input corpus (e.g., structured .md files)
├── src/
│   └── parser.py       # Logic for header-based document fatiamento
├── vector_db/          # Persistent ChromaDB storage (SQLite + vectors)
├── utils/              # Wrappers for API Gateway & text post-processing
├── ingest.py           # Mass ingestion pipeline optimized for GPU
├── chat_phd.py         # Interactive chat interface with streaming support
├── test_parser.py      # Script for validating document segmentation
├── test_query.py       # Semantic search (Retrieval) smoke test
├── config.ini.example  # Configuration template for local/server setup
├── pyproject.toml      # Dependency management via 'uv'
└── uv.lock             # Deterministic lockfile for environment replication

```

---

## 🛠️ Configuration & Security

The system employs a `.ini` configuration pattern to decouple code from environment-specific credentials and network settings.

1. **Environment Isolation:** All sensitive endpoints, proxy hosts, and certificate paths are stored in a local `config.ini` (excluded from version control).
2. **Portability:** Relative paths are used throughout to ensure the system runs seamlessly across different HPC nodes or local development machines.
3. **Proxy Management:** Handles corporate network authentication and custom SSL certificate verification for secure model downloads.

---

## 🧬 Core Components & Engineering

### 1. GPU-Optimized Ingestion (`ingest.py`)

To handle the **1400+ document segments** generated from the corpus, the ingestion script is tuned for professional-grade GPUs (e.g., NVIDIA A30/V100):

* **Safety Batching:** Implements a controlled batching strategy (default: 10) to prevent `CUDA Out of Memory` (OOM) errors during heavy embedding tasks.
* **Memory Management:** Explicit use of `torch.cuda.empty_cache()` and `gc.collect()` to mitigate VRAM fragmentation.
* **Offline First:** Models are cached locally to ensure high availability in restricted network environments.

### 2. Context-Aware Parsing

The system moves beyond simple character-based splitting, utilizing a **Header-Aware Parser**:

* **Sectional Integrity:** Recognizes hierarchy within technical documents to keep related concepts together.
* **Metadata Injection:** Every retrieved chunk carries its source, author, and section. This allows the LLM to perform **grounded citations** (e.g., *"According to Author X in the Methodology section..."*).

---

## 🚀 Key Technical Challenges Overcome

* **Corporate Network Integration:** Successfully configured TLS/SSL handshakes and proxy tunnels for high-bandwidth model synchronization.
* **Cross-Lingual Retrieval:** Validated that Portuguese queries can accurately retrieve deep-technical English content without loss of nuance.
* **Resource Efficiency:** Maximized PyTorch memory allocation through `expandable_segments` to handle large context windows on shared infrastructure.

---

## 🧪 Validation Framework

* **`test_parser.py`:** Validates that the regex-based segmentation is not "dropping" text and is correctly identifying metadata fields.
* **`test_query.py`:** A pure Retrieval test. It confirms the "brain" (Vector DB) is responding with relevant context before involving the "voice" (LLM).

---

## 📈 Future Roadmap

* **Domain Evaluation:** While the flow is highly effective for structured academic/technical `.md` files (I used for prototyping), further evaluation is required for unstructured or semi-structured corporate data.
* **Metadata Filtering:** Implementation of granular search filters (e.g., "Search only within 'Conclusion' sections").
* **Automated Synthesis:** Exporting RAG-generated literature reviews directly into CSV or bibliography management formats.

---

**Lead Developer:** Ramon Ferrari

**Role:** Data Science Leader

**Project Date:** March 2026

---