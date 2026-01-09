# 🤖 Codebase-Explainer-QnA (MAKER Edition)

Tired of spending hours trying to understand a new open-source project? This tool is built for **open-source contributors** to get up to speed on any new codebase in minutes.

It uses a local LLM (Ollama) and the **MAKER Framework** to decompose the codebase into micro-tasks, generating a high-quality architectural tutorial and an interactive Q&A system.

## ✨ Features

*   **Web Interface (NEW):** A modern, dark-themed dashboard to chat with your code and view reports.
*   **MAKER Framework:** Uses "Micro-Agents" to summarize files individually (Decomposition) and validates outputs (Red-Flagging) for higher reliability.
*   **Issue Resolver:** A dedicated chat mode to debug specific issues, suggesting files and fixes.
*   **Automated Tutorial:** Generates an HTML report with project overview and architecture.
*   **Local & Private:** Uses your local [Ollama](https://ollama.com/) instance.
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
# Analyze a repo and save the database
python tutorial_generator.py --url https://github.com/username/repo --persist
```

## 🔧 How It Works (The MAKER Method)

This tool applies the researched **MAKER Framework** (Massively Agentic decomposed processes):

1.  **Decomposition (Micro-Agents):** Instead of one giant prompt, the tool spawns a "Micro-Agent" for every file to summarize its purpose.
2.  **Red-Flagging:** Bad outputs from agents are detected and discarded/retried.
3.  **Aggregation:** Verified summaries are combined to produce the final architectural report.
4.  **RAG Q&A:** The full codebase is embedded into a FAISS vector store for the chat system.