# 🤖 Codebase-Explainer-QnA (Gemini MAKER Edition)

Tired of spending hours trying to understand a new open-source project? This tool is built for **developers and open-source contributors** to get up to speed on any new codebase in minutes.

Powered by **Google Gemini** (`gemini-3.1-flash-lite`) and the **MAKER Framework**, it decomposes any GitHub repository into micro-tasks, builds an AST-based code dependency graph, generates a comprehensive architectural tutorial report, and provides an interactive AI Q&A and debugging system.

---

## ✨ Key Features

*   🧠 **Gemini Powered:** High-speed, high-context intelligence using Google's `gemini-3.1-flash-lite` and `gemini-embedding-2`.
*   🖥️ **Modern Web Dashboard:** Dark-themed web interface with real-time analysis progress, chat mode, issue resolver, and an embedded tutorial report viewer.
*   🤖 **MAKER Framework:** Applies Massively Agentic Decomposed Processes—spawns micro-agents to analyze code files individually (Map), validates outputs (Red-Flagging), and aggregates them into a comprehensive architectural guide (Reduce).
*   🔗 **AST Code Graph Context:** Extracts Python AST entities (functions, classes, methods, imports, and calls) and maps relationships using NetworkX. Enables structural context retrieval even without vector embeddings.
*   🔍 **Hybrid & Fallback Retrieval:** Combines FAISS vector similarity with structural Code Graph matching. Gracefully falls back to pure graph keyword search if API rate limits or quota boundaries are reached.
*   ⚡ **Smart Analysis Caching:** Automatically detects previously analyzed repositories and loads the cached tutorial report and dependency graph in **under 2 seconds**.
*   🐞 **Issue Resolver Mode:** Switch modes in the chat interface to pinpoint bug locations and suggest concrete fixes across the codebase.
*   📊 **Interactive Graph Export:** Exports the complete dependency graph as JSON (`reports/<repo>_codegraph.json`) for downstream analysis.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
*   [Git](https://git-scm.com/downloads)
*   [Python 3.10+](https://www.python.org/downloads/)
*   A **Gemini API Key** from [Google AI Studio](https://aistudio.google.com/app/apikey)

### 2. Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/AvirupSahaAug/Codebase-Explainer-QnA.git
   cd Codebase-Explainer-QnA
   ```

2. **Set up a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate    # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your API Key:**
   Copy `.env.example` to `.env` and add your Gemini API Key:
   ```bash
   cp .env.example .env
   ```
   Edit `.env`:
   ```env
   GEMINI_API_KEY=AIzaSyYourActualKeyHere
   ```
   *(Alternatively, you can paste the API key directly into the sidebar in the Web UI).*

---

## 🏃‍♂️ How to Use

### Option A: Web Interface (Recommended)

1. **Start the FastAPI server:**
   ```bash
   python server.py
   ```
2. **Open your browser:** Navigate to `http://localhost:8000`
3. **Connect a Repository:** Enter any public GitHub URL (e.g. `https://github.com/encode/starlette`) and click **"Analyze & Decompose"**.
4. **Interact:**
   * **Chat:** Ask questions about architecture, lifecycle, or specific files.
   * **Issue Resolver:** Describe a bug to get targeted suggestions and source references.
   * **View Tutorial Report:** Open the generated HTML architectural report directly from the sidebar.

---

### Option B: CLI Tool

If you prefer working strictly in the terminal:

```bash
# Full analysis with both FAISS and Code Graph (default)
python tutorial_generator.py --url https://github.com/username/repo --persist

# Code Graph only (bypasses vector embeddings to save API quota)
python tutorial_generator.py --url https://github.com/username/repo --no-faiss

# Use a specific Gemini model
python tutorial_generator.py --url https://github.com/username/repo --model gemini-3.5-flash-lite
```

---

## 🔧 Architecture: The MAKER Method + Code Graph

```
 GitHub Repository
        │
        ├── AST Parsing (code_graph_builder.py) ──► Directed Entity & Dependency Graph
        │
        ├── Document Chunking & Loading
        │
        ▼
 [1. Decomposition]  ──► Micro-Agents summarize each file with Gemini
        │
 [2. Red-Flagging]   ──► Validate output & filter errors
        │
 [3. Aggregation]    ──► Combine summaries & graph context (Map-Reduce)
        │
        ├──► Generated HTML Tutorial Report (saved in reports/)
        └──► RetrievalQA Chain (Hybrid FAISS Vectors + Graph Context)
```

---

## 📁 Project Structure

```
├── server.py                 # FastAPI backend & streaming analysis endpoints
├── tutorial_generator.py     # Core MAKER engine & Gemini Q&A chain
├── code_graph_builder.py     # AST dependency graph extraction
├── static/
│   ├── index.html            # Web interface with inline reactive client
│   ├── style.css             # Dark-themed UI styles
│   └── app.js               # Client controller logic
├── reports/                  # Generated HTML reports & graph JSON files
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variables template
└── .gitignore                # Git exclusions (protects .env & cache)
```

---
