# 🤖 Codebase-Explainer-QnA (MAKER Edition)

Tired of spending hours trying to understand a new open-source project? This tool is built for **open-source contributors** to get up to speed on any new codebase in minutes.

It uses a local LLM (Ollama) and the **MAKER Framework** to decompose the codebase into micro-tasks, generating a high-quality architectural tutorial and an interactive Q&A system.

## ✨ Features

*   **Web Interface (NEW):** A modern, dark-themed dashboard to chat with your code and view reports.
*   **MAKER Framework:** Uses "Micro-Agents" to summarize files individually (Decomposition) and validates outputs (Red-Flagging) for higher reliability.
*   **Code Graph Context (NEW):** Extract AST-based code structure, dependencies, and relationships. Use code structure to augment retrieval even without FAISS.
*   **Hybrid Retrieval:** Combine FAISS vector similarity with Code Graph structural context for better results.
*   **Issue Resolver:** A dedicated chat mode to debug specific issues, suggesting files and fixes.
*   **Automated Tutorial:** Generates an HTML report with project overview and architecture.
*   **Graph Export:** Export code dependency graph as JSON for visualization and analysis.
*   **Local & Private:** Uses your local [Ollama](https://ollama.com/) instance.
*   **Flexible Configuration:** Enable/disable FAISS and Code Graph independently via CLI or Web UI.
*   **Persistence:** Saves the vector database to disk so you don't have to re-analyze the same repo twice.

## 🛠️ Installation & Setup

### 1. Prerequisites
*   [Git](https://git-scm.com/downloads)
*   [Python 3.8+](https://www.python.org/downloads/)
*   [Ollama](https://ollama.com/) installed and running.

### 2. Setup
1.  **Clone this repository:**
    ```bash
    git clone https://github.com/your-username/codebase-quickstart.git
    cd codebase-quickstart
    ```

2.  **Install dependencies:**
    ```bash
    pip install langchain langchain-community langchain-core langchain-text-splitters faiss-cpu requests markdown tqdm fastapi uvicorn python-multipart
    ```

3.  **Pull Ollama models:**
    ```bash
    ollama pull llama3.1:8b
    ollama pull nomic-embed-text
    ```

## 🏃‍♂️ How to Use

### Option A: Web Interface (Recommended)
The best way to experience the tool is via the simplified Web UI.

1.  **Start the server:**
    ```bash
    python server.py
    ```
2.  **Open your browser:** Go to `http://localhost:8000`
3.  **Enter a GitHub URL:** Click "Analyze" and watch the micro-agents work.
4.  **Chat:** Use the "Chat" or "Issue Resolver" tabs to interact with the codebase.

### Option B: CLI Tool
If you prefer the terminal:

```bash
# Analyze with both FAISS and Code Graph (default)
python tutorial_generator.py --url https://github.com/username/repo --persist

# Use only Code Graph (no FAISS embeddings)
python tutorial_generator.py --url https://github.com/username/repo --no-faiss

# Use only FAISS (traditional vector search)
python tutorial_generator.py --url https://github.com/username/repo --no-graph

# Disable both (not recommended - will use fallback retriever)
python tutorial_generator.py --url https://github.com/username/repo --no-graph --no-faiss
```

## 🔧 How It Works (The MAKER Method + Code Graph)

This tool applies the researched **MAKER Framework** (Massively Agentic decomposed processes) plus **Code Graph Context**:

### MAKER Framework:
1.  **Decomposition (Micro-Agents):** Instead of one giant prompt, the tool spawns a "Micro-Agent" for every file to summarize its purpose.
2.  **Red-Flagging:** Bad outputs from agents are detected and discarded/retried.
3.  **Aggregation:** Verified summaries are combined to produce the final architectural report.

### Code Graph (NEW):
4.  **AST Analysis:** Parses Python files to extract functions, classes, methods, imports, and dependencies.
5.  **Relationship Mapping:** Builds a directed graph of code entities and how they call each other.
6.  **Context Enrichment:** Augments documents with related file and dependency information.
7.  **Hybrid Retrieval:** Combines FAISS vector similarity with code structure for better context.

### Q&A System:
8.  **RAG Q&A:** The full codebase is embedded into a FAISS vector store and enhanced with graph context.
9.  **Flexible Retrieval:** Choose between FAISS-only, Graph-only, or hybrid retrieval strategies.