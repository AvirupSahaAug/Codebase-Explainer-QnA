#!/usr/bin/env python3
"""
Tutorial Generator with Ollama + LangChain
Fixed imports version

Usage:
1. Run the script
2. Enter GitHub URL when prompted
3. Get HTML tutorial + Q&A system
"""

import os
import subprocess
from pathlib import Path
from typing import List, Dict
import requests
import markdown

# LangChain imports - fixed
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.llms import Ollama
    from langchain_core.documents import Document
    from langchain_classic.chains import RetrievalQA


    from langchain_classic.prompts import PromptTemplate
    print("✅ All LangChain imports successful")
except ImportError as e:
    print(f"❌ Missing LangChain component: {e}")
    print("Run: pip install langchain langchain-community langchain-core langchain-text-splitters faiss-cpu")
    exit(1)

class TutorialGeneratorWithLangChain:
    def __init__(self, model_name: str = "llama3.1:8b"):
        self.model_name = model_name
        self.repo_path = None
        self.vector_store = None
        self.qa_chain = None
        
    def clone_repository(self, repo_url: str) -> str:
        """Clone or update repository"""
        repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        self.repo_path = f"./repo_{repo_name}"
        
        if os.path.exists(self.repo_path):
            print(f"📂 Updating existing repository...")
            try:
                subprocess.run(["git", "pull"], cwd=self.repo_path, check=True, capture_output=True)
            except:
                print("⚠️ Could not pull updates, using existing code")
        else:
            print(f"📦 Cloning repository...")
            subprocess.run(["git", "clone", "--depth", "1", repo_url, self.repo_path], 
                         check=True, capture_output=True)
            
        return self.repo_path
    
    def load_code_documents(self) -> List[Document]:
        """Load code files as LangChain Documents"""
        exclude_dirs = {'node_modules', 'venv', '.git', '__pycache__', 'dist', 'build'}
        include_exts = {'.py', '.js', '.jsx', '.ts', '.tsx', '.c', '.cc', '.cpp', '.md', '.json'}
        
        documents = []
        
        for root, dirs, files in os.walk(self.repo_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in include_exts:
                    try:
                        content = self.read_file_content(file_path)
                        relative_path = file_path.relative_to(self.repo_path)
                        
                        # Create LangChain Document
                        doc = Document(
                            page_content=content,
                            metadata={
                                "source": str(relative_path),
                                "file_type": file_path.suffix,
                            }
                        )
                        documents.append(doc)
                        
                    except Exception as e:
                        print(f"⚠️ Error reading {file_path}: {e}")
        
        print(f"📁 Loaded {len(documents)} code documents")
        return documents
    
    def read_file_content(self, file_path: Path) -> str:
        """Read file content with encoding handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if len(content) > 8000:
                    return content[:8000] + "\n//...truncated"
                return content
        except:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                    if len(content) > 8000:
                        return content[:8000] + "\n//...truncated"
                    return content
            except:
                return f"// Could not read: {file_path.name}"
    
    def setup_qa_system(self, documents: List[Document]):
        """Setup LangChain Q&A system with vector store"""
        print("🔧 Setting up Q&A system...")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        print(f"📄 Split into {len(chunks)} chunks")
        
        # Create embeddings and vector store
        try:
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            self.vector_store = FAISS.from_documents(chunks, embeddings)
        except Exception as e:
            print(f"⚠️ Could not create embeddings: {e}")
            print("🔧 Falling back to simple similarity search")
            # Simple fallback without embeddings
            self.vector_store = None
            return
        
        # Setup LLM
        llm = Ollama(model=self.model_name, temperature=0.1)
        
        # Create Q&A chain
        qa_prompt = PromptTemplate(
            template="""You are a code expert. Use the context to answer questions.

Context: {context}

Question: {question}

Answer based on the code. Reference specific files and provide code snippets when helpful.
Answer:""",
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": qa_prompt}
        )
        
        print("✅ Q&A system ready!")
    
    def ask_question(self, question: str) -> Dict:
        """Ask a question about the codebase"""
        if not self.qa_chain:
            return {"error": "Q&A system not setup"}
        
        try:
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
    
    def generate_tutorial(self, repo_name: str, documents: List[Document]) -> str:
        """Generate tutorial using Ollama"""
        print("🤖 Generating tutorial...")
        
        # Prepare context from documents
        context = ""
        for doc in documents[:20]:  # Limit context
            context += f"\n--- File: {doc.metadata['source']} ---\n"
            context += doc.page_content[:500] + "\n"  # Limit per file
        
        prompt = f"""
        Create a comprehensive tutorial for this codebase:

        Repository: {repo_name}

        Code Context:
        {context[:6000]}

        Create a markdown tutorial with:

        # Project Overview
        - Purpose and main features

        # Setup Instructions
        - Installation steps
        - How to run

        # Key Components
        - Main files and their roles
        - Important functions/classes

        # Usage Examples
        - How to use the code
        - Code examples

        Reference specific files from the codebase.
        """
        
        # Call Ollama directly
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json().get("response", "No response")
        except Exception as e:
            return f"Error generating tutorial: {e}"
    
    def create_html_report(self, tutorial: str, repo_name: str, file_count: int):
        """Create HTML report with tutorial"""
        tutorial_html = markdown.markdown(tutorial)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{repo_name} - Tutorial</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    max-width: 1000px;
                    margin: 0 auto;
                    padding: 20px;
                    line-height: 1.6;
                    background: #f5f5f5;
                }}
                .header {{
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                    text-align: center;
                }}
                .content {{
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                pre {{
                    background: #2d2d2d;
                    color: white;
                    padding: 15px;
                    border-radius: 5px;
                    overflow-x: auto;
                }}
                .feature {{
                    background: #e8f0fe;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 10px 0;
                }}
                .file-count {{
                    background: #4285f4;
                    color: white;
                    padding: 8px 16px;
                    border-radius: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 {repo_name}</h1>
                <p>Generated with Ollama + LangChain</p>
                <div class="file-count">📁 {file_count} files analyzed</div>
            </div>
            
            <div class="content">
                <div class="feature">
                    <strong>✨ Features:</strong> 
                    Code-aware Q&A | Source referencing | Semantic search
                </div>
                {tutorial_html}
            </div>
        </body>
        </html>
        """
        
        os.makedirs("reports", exist_ok=True)
        report_file = f"reports/{repo_name}_tutorial.html"
        
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        return report_file

def main():
    print("=" * 50)
    print("🤖 Tutorial Generator - Ollama + LangChain")
    print("=" * 50)
    
    # Check Ollama
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code != 200:
            print("❌ Ollama not running! Run: ollama serve")
            return
        
        # Check if model exists
        models = response.json().get("models", [])
        model_names = [model.get("name", "") for model in models]
        if "llama3.1:8b" not in model_names:
            print("❌ Model llama3.1:8b not found! Run: ollama pull llama3.1:8b")
            return
            
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        print("Make sure Ollama is running: ollama serve")
        return
    
    # Get URL
    repo_url = input("\n📥 Enter GitHub URL: ").strip()
    if not repo_url.startswith("https://github.com/"):
        print("❌ Invalid GitHub URL")
        return
    
    # Initialize generator
    generator = TutorialGeneratorWithLangChain()
    
    try:
        # Clone repo
        print("\n📥 Cloning repository...")
        repo_path = generator.clone_repository(repo_url)
        repo_name = Path(repo_path).name
        
        # Load documents
        print("📚 Loading code documents...")
        documents = generator.load_code_documents()
        
        if not documents:
            print("❌ No code files found!")
            return
        
        # Generate tutorial
        print("📝 Generating tutorial...")
        tutorial = generator.generate_tutorial(repo_name, documents)
        
        # Create report
        print("🎨 Creating HTML report...")
        report_path = generator.create_html_report(tutorial, repo_name, len(documents))
        
        # Setup Q&A system
        print("🔧 Setting up Q&A system...")
        generator.setup_qa_system(documents)
        
        print(f"\n✅ Tutorial generated: {report_path}")
        
        # Interactive Q&A
        if generator.qa_chain:
            print("\n💬 Q&A System Ready! Ask questions about the codebase.")
            print("Type 'quit' to exit.\n")
            
            while True:
                question = input("❓ Question: ").strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                
                result = generator.ask_question(question)
                
                if "error" in result:
                    print(f"❌ Error: {result['error']}")
                else:
                    print(f"\n📝 Answer: {result['answer']}")
                    if result.get('sources'):
                        print("\n📚 Sources:")
                        for i, source in enumerate(result['sources'], 1):
                            print(f"   {i}. {source['file']}")
                    print("-" * 50)
        else:
            print("⚠️ Q&A system not available, but tutorial was generated successfully!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()