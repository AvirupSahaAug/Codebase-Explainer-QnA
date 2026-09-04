#!/usr/bin/env python3
"""
Tutorial Generator with Gemini + LangChain
MAKER Framework Edition: Decomposition, Error Correction, and Scale
Enhanced with Code Graph Context + FAISS Vector Store

Usage:
    python tutorial_generator.py --url <github_url> [--model <model_name>] [--persist]
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Optional
import markdown
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from current dir and parent dir
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# LangChain & Google GenAI imports
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from langchain_core.documents import Document
    from langchain_classic.chains import RetrievalQA
    from langchain_classic.prompts import PromptTemplate
except ImportError as e:
    print(f"❌ Missing required library: {e}")
    print("Please install dependencies: pip install langchain-google-genai faiss-cpu langchain-classic langchain-text-splitters")
    sys.exit(1)

# Code Graph
try:
    from code_graph_builder import CodeGraphBuilder
except ImportError as e:
    print(f"⚠️ Code Graph not available: {e}")
    CodeGraphBuilder = None


def _extract_text(content) -> str:
    """Safely convert string, list of parts/dicts, or content objects to string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content) if content is not None else ""


class TutorialGeneratorMAKER:
    def __init__(
        self,
        model_name: str = "gemini-3.1-flash-lite",
        embedding_model: str = "models/gemini-embedding-2",
        persist_dir: str = "db_faiss",
        progress_callback=None,
        use_graph: bool = True,
        use_faiss: bool = True,
        api_key: Optional[str] = None
    ):
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.persist_dir = persist_dir
        self.progress_callback = progress_callback  # function(current, total, status_msg)
        self.use_graph = use_graph
        self.use_faiss = use_faiss
        self.repo_path = None
        self.vector_store = None
        self.qa_chain = None
        self.code_graph = None
        self.documents = None
        
        # Resolve API Key
        raw_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not raw_key:
            raise ValueError(
                "❌ Gemini API Key not found! Please set GEMINI_API_KEY in your .env file or enter it in the sidebar."
            )
        self.api_key = raw_key.strip().strip('"').strip("'")
        if self.api_key == "your_gemini_api_key_here" or not self.api_key:
            raise ValueError(
                "❌ Placeholder key detected. Please paste your actual Gemini API Key (starts with AIzaSy...) in your .env file."
            )
        
        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=0.1
        )

    def _update_progress(self, current, total, message):
        if self.progress_callback:
            self.progress_callback(current, total, message)

    def clone_repository(self, repo_url: str) -> str:
        """Clone or update repository"""
        repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        self.repo_path = f"./repo_{repo_name}"
        
        if os.path.exists(self.repo_path):
            print(f"📂 Updating existing repository: {repo_name}")
            try:
                subprocess.run(["git", "pull"], cwd=self.repo_path, check=True, capture_output=True)
            except Exception:
                print("⚠️ Could not pull updates, using existing code")
        else:
            print(f"📦 Cloning repository: {repo_name}")
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, self.repo_path], 
                check=True,
                capture_output=True
            )
            
        return self.repo_path
    
    def load_code_documents(self) -> List[Document]:
        """Load code files as LangChain Documents"""
        exclude_dirs = {'node_modules', 'venv', '.venv', '.git', '__pycache__', 'dist', 'build', 'site-packages'}
        include_exts = {'.py', '.js', '.jsx', '.ts', '.tsx', '.c', '.cc', '.cpp', '.md', '.json', '.html', '.css', '.java', '.go', '.rs'}
        
        documents = []
        files_to_process = []

        for root, dirs, files in os.walk(self.repo_path):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in include_exts:
                    files_to_process.append(file_path)
        
        print(f"📚 Found {len(files_to_process)} eligible files. Loading content...")
        self._update_progress(0, len(files_to_process), "Loading files...")
        
        loaded_count = 0
        for file_path in tqdm(files_to_process, unit="file"):
            try:
                content = self.read_file_content(file_path)
                relative_path = file_path.relative_to(self.repo_path)
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(relative_path),
                        "file_type": file_path.suffix,
                    }
                )
                documents.append(doc)
                loaded_count += 1
                if loaded_count % 10 == 0:
                    self._update_progress(loaded_count, len(files_to_process), f"Loaded {loaded_count} files")
            except Exception:
                pass  # Skip files that fail to read
        
        self._update_progress(len(files_to_process), len(files_to_process), "Files loaded.")
        
        # Build code graph if enabled
        if self.use_graph and CodeGraphBuilder:
            self._update_progress(len(files_to_process), len(files_to_process), "Building code graph...")
            self.code_graph = CodeGraphBuilder(self.repo_path)
            self.code_graph.build_graph(documents)
            documents = self.code_graph.export_as_documents(documents)
            graph_summary = self.code_graph.get_graph_summary()
            print(f"✅ Code Graph: {graph_summary['nodes']} nodes, {graph_summary['edges']} edges")
        
        self.documents = documents
        return documents
    
    def read_file_content(self, file_path: Path) -> str:
        """Read file content with robust encoding handling"""
        encodings = ['utf-8', 'latin-1', 'cp1252']
        for enc in encodings:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    return f.read()
            except Exception:
                continue
        raise Exception("Could not read file")

    def _micro_agent_summarize(self, doc: Document) -> Optional[str]:
        """
        Micro-Agent: Summarizes a single file using Gemini.
        Red-Flagging: Retries or fails if output is malformed.
        """
        prompt = f"""Role: Senior Developer
Task: Provide a concise 1-sentence summary of this file's purpose.
File: {doc.metadata['source']}
Code Snippet:
{doc.page_content[:2500]}

Format: "File: {doc.metadata['source']} - [Summary]"
"""
        try:
            response = self.llm.invoke(prompt)
            raw = getattr(response, "content", response)
            text = _extract_text(raw).strip()
            
            # Basic validation
            if len(text) < 10 or "error" in text.lower():
                return None
                
            return text
        except Exception as e:
            print(f"⚠️ Micro-agent summary skipped for {doc.metadata.get('source')}: {e}")
            return None

    def generate_tutorial_maker_style(self, repo_name: str, documents: List[Document]) -> str:
        """
        MAKER Framework Implementation:
        1. Decompose: Summarize each file individually (Map).
        2. Aggregate: Combine summaries.
        3. Generate: Create final tutorial from aggregated context (Reduce).
        """
        print("\n🤖 MAKER: Decomposing task into micro-agents...")
        self._update_progress(0, 100, "Starting MAKER analysis with Gemini...")
        
        file_summaries = []
        sample_docs = documents[:40]  # Representative sample
        total_docs = len(sample_docs)
        
        for i, doc in enumerate(tqdm(sample_docs, desc="Micro-Agent Summarization")):
            summary = self._micro_agent_summarize(doc)
            if summary:
                file_summaries.append(summary)
            self._update_progress(i + 1, total_docs, f"Analyzing {doc.metadata['source']}...")
        
        print(f"✅ Aggregated {len(file_summaries)} file summaries.")
        self._update_progress(100, 100, "Generating final tutorial report...")
        
        if not file_summaries:
            print("⚠️ No micro-agent summaries generated (possible API issue).")
            return "## Code Graph Analysis Only\nNo micro-agent summaries were generated due to an API error (check API key and billing/suspension status)."

        # MAP-REDUCE: Final generation
        context_blob = "\n".join(file_summaries)
        
        final_prompt = f"""You are a Principal Software Architect.
Create a comprehensive, high-level architectural tutorial for the repository '{repo_name}'.

Based on these code summaries and structure:
{context_blob}

Structure your tutorial as follows:
# 1. Project Overview
# 2. Architecture & Key Components
# 3. Main Logic Flow & Dependencies
# 4. Usage & Execution Walkthrough
"""
        try:
            print("📝 Generating final tutorial with Gemini Flash...")
            response = self.llm.invoke(final_prompt)
            raw = getattr(response, "content", response)
            return _extract_text(raw)
        except Exception as e:
            return f"Generation Failed: {str(e)}"

    def setup_qa_system(self, documents: List[Document], use_persist: bool = False) -> bool:
        """Setup Q&A System with optional FAISS and/or Code Graph"""
        print("\n🔧 Setting up Q&A System...")
        print(f"   Strategy: FAISS={self.use_faiss}, CodeGraph={self.use_graph}")
        
        # Setup FAISS if enabled
        if self.use_faiss:
            try:
                persist_path = Path(self.persist_dir)
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/gemini-embedding-2",
                    google_api_key=self.api_key
                )

                if use_persist and persist_path.exists() and (persist_path / "index.faiss").exists():
                    print("💾 Loading existing vector store from disk...")
                    try:
                        self.vector_store = FAISS.load_local(
                            self.persist_dir,
                            embeddings,
                            allow_dangerous_deserialization=True
                        )
                        print("✅ Loaded from cache.")
                    except Exception as e:
                        print(f"⚠️ Cache load failed ({e}), rebuilding...")
                        self.vector_store = None

                if not self.vector_store:
                    print("📝 Creating new Gemini embeddings...")
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = text_splitter.split_documents(documents)
                    self.vector_store = FAISS.from_documents(chunks, embeddings)
                    
                    if use_persist:
                        os.makedirs(self.persist_dir, exist_ok=True)
                        self.vector_store.save_local(self.persist_dir)
                        print("💾 Vector store saved to disk.")
            except Exception as e:
                print(f"⚠️ FAISS embedding failed ({e}), falling back to Code Graph retriever...")
                self.use_faiss = False
                self.vector_store = None
        
        # Setup Retriever (hybrid if both enabled)
        if self.use_faiss and self.vector_store:
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        elif self.use_graph and self.code_graph:
            retriever = self._create_graph_retriever()
        else:
            retriever = self._create_simple_retriever(documents)
        
        # Setup QA Chain
        qa_prompt = PromptTemplate(
            template="""You are an expert AI code assistant. Use the following context to answer the question clearly and accurately.

Context:
{context}

Question: {question}

Answer:""",
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": qa_prompt},
            return_source_documents=True
        )
        return True
    
    def _create_graph_retriever(self):
        """Create a graph-based retriever using code structure"""
        from typing import Any
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.callbacks import CallbackManagerForRetrieverRun
        
        class GraphRetriever(BaseRetriever):
            graph_builder: Any = None
            documents: List[Any] = []
            
            def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None):
                relevant = []
                query_lower = query.lower()
                for doc in self.documents:
                    content_lower = doc.page_content.lower()
                    if any(term in content_lower for term in query_lower.split()):
                        relevant.append(doc)
                return relevant[:5]
        
        return GraphRetriever(graph_builder=self.code_graph, documents=self.documents)
    
    def _create_simple_retriever(self, documents):
        """Create a simple keyword-based retriever"""
        from typing import Any
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.callbacks import CallbackManagerForRetrieverRun
        
        class SimpleRetriever(BaseRetriever):
            docs: List[Any] = []
            
            def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None):
                relevant = []
                query_lower = query.lower()
                for doc in self.docs:
                    content_lower = doc.page_content.lower()
                    if any(term in content_lower for term in query_lower.split()):
                        relevant.append(doc)
                return relevant[:5]
        
        return SimpleRetriever(docs=documents)

    def create_html_report(self, tutorial: str, repo_name: str) -> str:
        """Create HTML report"""
        tutorial_str = _extract_text(tutorial)
        tutorial_html = markdown.markdown(tutorial_str)
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{repo_name} - Gemini MAKER Analysis</title>
    <style>
        body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 900px; margin: 0 auto; padding: 40px; line-height: 1.6; color: #333; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        pre {{ background: #f4f6f8; padding: 15px; border-radius: 8px; overflow-x: auto; }}
        code {{ font-family: monospace; background: #eef2f5; padding: 2px 6px; border-radius: 4px; }}
        .tag {{ background: #e1f5fe; color: #0277bd; padding: 4px 8px; border-radius: 4px; font-size: 0.9em; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>📦 {repo_name} <span class="tag">Gemini Analysis</span></h1>
    {tutorial_html}
</body>
</html>
"""
        os.makedirs("reports", exist_ok=True)
        report_file = f"reports/{repo_name}_maker.html"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        return report_file

    def ask_question(self, question: str) -> Dict:
        """Ask a question about the codebase"""
        if not self.qa_chain:
            return {"error": "System not ready. Please analyze a repo first."}
        
        try:
            print(f"🤔 Asking Gemini: {question}")
            result = self.qa_chain.invoke({"query": question})
            
            sources = []
            for doc in result.get("source_documents", []):
                sources.append({
                    "file": doc.metadata.get("source", "unknown"),
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                })
            
            answer_text = _extract_text(result.get("result", ""))
            return {
                "answer": answer_text,
                "sources": sources
            }
        except Exception as e:
            return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Codebase Explainer (MAKER Edition with Gemini & Code Graph)")
    parser.add_argument("--url", help="GitHub Repository URL")
    parser.add_argument("--model", default="gemini-2.0-flash-lite", help="Gemini model to use (default: gemini-2.0-flash-lite)")
    parser.add_argument("--api-key", help="Google/Gemini API Key (optional, defaults to GEMINI_API_KEY env var)")
    parser.add_argument("--persist", action="store_true", help="Save/Load vector DB from disk")
    parser.add_argument("--graph", action="store_true", default=True, help="Use code graph context (default: True)")
    parser.add_argument("--no-graph", action="store_false", dest="graph", help="Disable code graph")
    parser.add_argument("--faiss", action="store_true", default=True, help="Use FAISS vectors (default: True)")
    parser.add_argument("--no-faiss", action="store_false", dest="faiss", help="Disable FAISS")
    args = parser.parse_args()

    repo_url = args.url
    if not repo_url:
        print("=" * 55)
        print("🤖 Codebase Explainer - Gemini MAKER Framework")
        print("   with Code Graph Context & FAISS")
        print("=" * 55)
        repo_url = input("\n📥 Enter GitHub URL: ").strip()

    if not repo_url:
        print("❌ URL required.")
        return

    try:
        generator = TutorialGeneratorMAKER(
            model_name=args.model,
            use_graph=args.graph,
            use_faiss=args.faiss,
            api_key=args.api_key
        )
    except ValueError as e:
        print(e)
        return
    
    # 1. Clone
    repo_path = generator.clone_repository(repo_url)
    repo_name = Path(repo_path).name
    
    # 2. Load Docs
    documents = generator.load_code_documents()
    if not documents:
        print("❌ No documents found.")
        return

    # 3. MAKER Tutorial Generation
    tutorial = generator.generate_tutorial_maker_style(repo_name, documents)
    report_path = generator.create_html_report(tutorial, repo_name)
    print(f"\n✨ Report generated: {report_path}")
    
    # Export code graph if available
    if generator.code_graph:
        graph_export = f"reports/{repo_name}_codegraph.json"
        generator.code_graph.export_graph_json(graph_export)

    # 4. Q&A System
    generator.setup_qa_system(documents, use_persist=args.persist)
    
    print("\n💬 Q&A System Ready! (Type 'quit' to exit)")
    print(f"   Strategy: CodeGraph={args.graph}, FAISS={args.faiss}, Model={args.model}")
    while True:
        q = input("\n❓ Question: ").strip()
        if q.lower() in ['quit', 'exit', 'q']:
            break
        
        if not generator.qa_chain:
            print("⚠️ System not ready.")
            continue
            
        res = generator.ask_question(q)
        if "error" in res:
            print(f"⚠️ Error: {res['error']}")
        else:
            print(f"\n📝 Answer: {res['answer']}")
            if res.get("sources"):
                print("\nSources:")
                for s in res["sources"]:
                    print(f"- {s['file']}")


if __name__ == "__main__":
    main()
