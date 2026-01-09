#!/usr/bin/env python3
"""
Tutorial Generator with Ollama + LangChain
MAKER Framework Edition: Decomposition, Error Correction, and Scale

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
import requests
import markdown
from tqdm import tqdm

# LangChain imports (Standard)
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.llms import Ollama
    from langchain_core.documents import Document
    from langchain_classic.chains import RetrievalQA
    from langchain_classic.prompts import PromptTemplate
except ImportError as e:
    print(f"❌ Missing LangChain component: {e}")
    sys.exit(1)

class TutorialGeneratorMAKER:
    def __init__(self, model_name: str = "llama3.1:8b", persist_dir: str = "db_faiss", progress_callback=None):
        self.model_name = model_name
        self.repo_path = None
        self.vector_store = None
        self.qa_chain = None
        self.persist_dir = persist_dir
        self.progress_callback = progress_callback # function(current, total, status_msg)
        
        # Check Ollama connection immediately
        self._check_ollama()

    def _update_progress(self, current, total, message):
        if self.progress_callback:
            self.progress_callback(current, total, message)

    def _check_ollama(self):
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code != 200:
                print("❌ Ollama not running! Run: ollama serve")
                # Don't exit in API mode, just raise error
                raise Exception("Ollama not running")
        except:
            raise Exception("Could not connect to Ollama")

    def clone_repository(self, repo_url: str) -> str:
        """Clone or update repository"""
        repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        self.repo_path = f"./repo_{repo_name}"
        
        if os.path.exists(self.repo_path):
            print(f"📂 Updating existing repository: {repo_name}")
            try:
                subprocess.run(["git", "pull"], cwd=self.repo_path, check=True, capture_output=True)
            except:
                print("⚠️ Could not pull updates, using existing code")
        else:
            print(f"📦 Cloning repository: {repo_name}")
            subprocess.run(["git", "clone", "--depth", "1", repo_url, self.repo_path], 
                         check=True, capture_output=True)
            
        return self.repo_path
    
    def load_code_documents(self) -> List[Document]:
        """Load code files as LangChain Documents"""
        exclude_dirs = {'node_modules', 'venv', '.git', '__pycache__', 'dist', 'build', 'site-packages'}
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
                pass # Skip files that fail to read
        
        self._update_progress(len(files_to_process), len(files_to_process), "Files loaded.")
        return documents
    
    def read_file_content(self, file_path: Path) -> str:
        """Read file content with robust encoding handling"""
        encodings = ['utf-8', 'latin-1', 'cp1252']
        for enc in encodings:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    return f.read()
            except:
                continue
        raise Exception("Could not read file")

    def _micro_agent_summarize(self, doc: Document) -> Optional[str]:
        """
        Micro-Agent: Summarizes a single file.
        Red-Flagging: Retries or fails if output is malformed.
        """
        prompt = f"""
        Role: Senior Developer
        Task: 1-sentence summary of this file's purpose.
        File: {doc.metadata['source']}
        Code Snippet:
        {doc.page_content[:2000]}
        
        Format: "File: [filename] - [Summary]"
        """
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1} # Low temp for deterministic execution
                },
                timeout=30
            )
            text = response.json().get("response", "").strip()
            
            # Red-Flagging: Basic validation
            if len(text) < 10 or "error" in text.lower(): 
                return None # Discard bad response
                
            return text
        except:
            return None

    def generate_tutorial_maker_style(self, repo_name: str, documents: List[Document]) -> str:
        """
        MAKER Framework Implementation:
        1. Decompose: Summarize each file individually (Map).
        2. Aggregate: Combine summaries.
        3. Generate: Create final tutorial from aggregated context (Reduce).
        """
        print("\n🤖 MAKER: Decomposing task into micro-agents...")
        self._update_progress(0, 100, "Starting MAKER analysis...")
        
        file_summaries = []
        # Limit to top 50 files to save time/tokens for this demo, 
        # but in full production this would run on all files parallelized.
        sample_docs = documents[:50] 
        total_docs = len(sample_docs)
        
        for i, doc in enumerate(tqdm(sample_docs, desc="Micro-Agent Summarization")):
            summary = self._micro_agent_summarize(doc)
            if summary:
                file_summaries.append(summary)
            self._update_progress(i + 1, total_docs, f"Analyzing {doc.metadata['source']}...")
        
        print(f"✅ Aggregated {len(file_summaries)} file summaries.")
        self._update_progress(100, 100, "Generating final report...")
        
        # MAP-REDUCE: Final generation
        context_blob = "\n".join(file_summaries)
        
        final_prompt = f"""
        Create a high-level architectural tutorial for the repository '{repo_name}'.
        
        Based ONLY on these file summaries:
        {context_blob}
        
        Structure:
        # 1. Project Overview
        # 2. Architecture & Key Components
        # 3. Main Logic Flow
        # 4. Usage Inference (How it likely works)
        """
        
        print("📝 Generating final tutorial...")
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model_name,
                "prompt": final_prompt,
                "stream": False
            },
            timeout=120
        )
        return response.json().get("response", "Generation Failed")

    def setup_qa_system(self, documents: List[Document], use_persist: bool = False) -> bool:
        """Setup Vector Store with Persistence"""
        print("\n🔧 Setting up Q&A System...")
        
        persist_path = Path(self.persist_dir)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        if use_persist and persist_path.exists():
            print("💾 Loading existing vector store from disk...")
            try:
                self.vector_store = FAISS.load_local(self.persist_dir, embeddings, allow_dangerous_deserialization=True)
                print("✅ Loaded from cache.")
            except Exception as e:
                print(f"⚠️ Cache load failed ({e}), rebuilding...")
                use_persist = False # Fallback to rebuild

        if not self.vector_store:
            print("creating new embeddings...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            
            # Batch embedding could include tqdm here
            self.vector_store = FAISS.from_documents(chunks, embeddings)
            
            if use_persist:
                self.vector_store.save_local(self.persist_dir)
                print("💾 Vector store saved to disk.")

        # Setup Chain
        llm = Ollama(model=self.model_name, temperature=0.1)
        qa_prompt = PromptTemplate(
            template="""Context: {context}\n\nQuestion: {question}\n\nAnswer:""",
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
            chain_type_kwargs={"prompt": qa_prompt},
            return_source_documents=True
        )
        return True

    def create_html_report(self, tutorial: str, repo_name: str) -> str:
        """Create HTML report"""
        tutorial_html = markdown.markdown(tutorial)
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{repo_name} - MAKER Analysis</title>
            <style>
                body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 900px; margin: 0 auto; padding: 40px; line-height: 1.6; color: #333; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                pre {{ background: #f4f6f8; padding: 15px; border-radius: 8px; overflow-x: auto; }}
                .tag {{ background: #e1f5fe; color: #0277bd; padding: 4px 8px; border-radius: 4px; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <h1>📦 {repo_name} <span class="tag">AI Analysis</span></h1>
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
            print(f"🤔 Asking: {question}")
            result = self.qa_chain({"query": question})
            
            # Extract sources
            sources = []
            for doc in result.get("source_documents", []):
                sources.append({
                    "file": doc.metadata.get("source", "unknown"),
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                })
            
            return {
                "answer": result["result"],
                "sources": sources
            }
        except Exception as e:
            return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Codebase Explainer (MAKER Edition)")
    parser.add_argument("--url", help="GitHub Repository URL")
    parser.add_argument("--model", default="llama3.1:8b", help="Ollama model to use")
    parser.add_argument("--persist", action="store_true", help="Save/Load vector DB from disk")
    args = parser.parse_args()

    # Interactive mode if no URL
    repo_url = args.url
    if not repo_url:
        print("="*50)
        print("🤖 Codebase Explainer - MAKER Framework")
        print("="*50)
        repo_url = input("\n📥 Enter GitHub URL: ").strip()

    if not repo_url:
        print("❌ URL required.")
        return

    generator = TutorialGeneratorMAKER(model_name=args.model)
    
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

    # 4. Q&A System
    generator.setup_qa_system(documents, use_persist=args.persist)
    
    print("\n💬 Q&A System Ready! (Type 'quit' to exit)")
    while True:
        q = input("\n❓ Question: ").strip()
        if q.lower() in ['quit', 'exit', 'q']:
            break
        
        if not generator.qa_chain:
            print("⚠️ System not ready.")
            continue
            
        res = generator.qa_chain({"query": q})
        print(f"\n📝 Answer: {res['result']}")
        print("\nSources:")
        for doc in res.get("source_documents", []):
            print(f"- {doc.metadata['source']}")

if __name__ == "__main__":
    main()
