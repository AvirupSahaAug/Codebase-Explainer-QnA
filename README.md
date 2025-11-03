

# 🤖 Codebase-Explainer-QnA

Tired of spending hours trying to understand a new open-source project? This tool is built for **open-source contributors** to get up to speed on any new codebase in minutes.

It clones a GitHub repository, uses a local LLM (Ollama) to generate a high-level tutorial, and then builds an interactive Q\&A system so you can ask specific questions about the code.

## 🚀 The 'Get Into The Thing' Tool for OSS

The biggest barrier to contributing to open source is the steep learning curve. This tool flattens that curve. Instead of manually reading thousands of lines of code, you get:

1.  **An Automated Tutorial:** An `index.html` report with a project overview, setup instructions, and key components.
2.  **A Code-Aware AI Assistant:** A Q\&A bot that has already "read" the entire repo.

This lets you find the exact file you need to edit, understand the project's architecture, and **make your first contribution faster.**

## ✨ Features

  * **One-Click Analysis:** Just provide a GitHub URL.
  * **Automated Tutorial:** Generates an HTML report with a project overview, setup instructions, and key component analysis.
  * **Interactive Q\&A:** A terminal-based chatbot to ask specific questions (e.g., "What does the `User` class do?", "Where are API keys handled?").
  * **Local & Private:** Uses your local [Ollama](https://ollama.com/) instance, so the code never leaves your machine.
  * **RAG-Powered:** Uses LangChain and FAISS to build a Retrieval-Augmented Generation (RAG) pipeline for accurate, source-aware answers.

## 🛠️ Installation & Setup

### 1\. Prerequisites

  * [Git](https://www.google.com/search?q=https://git-scm.com/downloads)
  * [Python 3.8+](https://www.python.org/downloads/)
  * [Ollama](https://ollama.com/) installed and running.

### 2\. Setup Instructions

1.  **Clone this repository:**

    ```bash
    git clone https://github.com/your-username/codebase-quickstart.git
    cd codebase-quickstart
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required Python packages:**

    ```bash
    pip install langchain langchain-community langchain-core langchain-text-splitters faiss-cpu requests markdown
    ```

4.  **Pull the required Ollama models:**
    This script uses `llama3.1:8b` for generation and `nomic-embed-text` for embeddings.

    ```bash
    ollama pull llama3.1:8b
    ollama pull nomic-embed-text
    ```

## 🏃‍♂️ How to Use

1.  **Run Ollama:**
    In a separate terminal, make sure the Ollama server is running:

    ```bash
    ollama serve
    ```

2.  **Run the script:**

    ```bash
    python tutorial_generator.py
    ```

3.  **Enter a GitHub URL:**
    When prompted, paste the URL of the repository you want to analyze.

    ```
    ==================================================
    🤖 Tutorial Generator - Ollama + LangChain
    ==================================================

    📥 Enter GitHub URL: https://github.com/langchain-ai/langchain
    ```

4.  **Get Your Outputs:**

      * **HTML Report:** The script will clone the repo, analyze the code, and generate a tutorial. You'll find it in the `reports/` directory (e.g., `reports/repo_langchain_tutorial.html`).
      * **Q\&A System:** After the report is built, the script will launch the interactive Q\&A system in your terminal.

    <!-- end list -->

    ```
    ✅ Tutorial generated: reports/repo_langchain_tutorial.html

    💬 Q&A System Ready! Ask questions about the codebase.
    Type 'quit' to exit.

    ❓ Question: What is the main purpose of the Document class?

    📝 Answer: The `Document` class, found in `langchain_core/documents/`, acts as a container for a piece of text and its associated metadata. It's a fundamental unit of data used throughout the library, representing a single "document" that can be processed, retrieved, or used in a chain.

    📚 Sources:
       1. langchain_core/documents/base.py
    --------------------------------------------------
    ❓ Question:
    ```

## 🔧 How It Works (Technical Details)

This tool uses two distinct methods for its two main features:

1.  **Tutorial Generation (Direct API Call):**

      * It samples the first 20 code files from the repository.
      * It combines their content into a single, large prompt.
      * It sends this prompt directly to the Ollama `/api/generate` endpoint to get a high-level markdown tutorial.
      * It converts this markdown to a styled HTML file.

2.  **Q\&A System (LangChain RAG):**

      * **Load:** Scans all code files into LangChain `Document` objects.
      * **Split:** Breaks down large files into smaller chunks using `RecursiveCharacterTextSplitter`.
      * **Embed & Store:** Uses `OllamaEmbeddings` (with `nomic-embed-text`) to create numerical vector representations of each chunk and stores them in a local `FAISS` vector store.
      * **Retrieve & Answer:** When you ask a question, the `RetrievalQA` chain:
        1.  Embeds your question.
        2.  Searches the `FAISS` store for the most relevant code chunks (the "context").
        3.  Passes your question and the retrieved context to the `Ollama` LLM (`llama3.1:8b`).
        4.  Generates a final answer based *only* on the provided context, with links to the source files.