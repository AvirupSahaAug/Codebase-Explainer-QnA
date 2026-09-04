from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import asyncio
from typing import Optional, List
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from current dir and parent dir
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Import our generator
from tutorial_generator import TutorialGeneratorMAKER

app = FastAPI(title="Codebase Explainer AI (Gemini Edition)")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# -- State Management --
class GlobalState:
    generator: Optional[TutorialGeneratorMAKER] = None
    progress: dict = {"current": 0, "total": 100, "message": "Idle", "status": "idle"}
    report_path: Optional[str] = None

state = GlobalState()

import logging

# Filter out frequent /api/status polling from terminal logs
class StatusLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/api/status" not in record.getMessage()

logging.getLogger("uvicorn.access").addFilter(StatusLogFilter())

# -- Models --
class AnalyzeRequest(BaseModel):
    url: str
    model: str = "gemini-2.0-flash-lite"
    model: str = "gemini-3.5-flash-lite"
    model: str = "gemini-3.1-flash-lite"
    use_graph: bool = True
    use_faiss: bool = True
    api_key: Optional[str] = None

class ChatRequest(BaseModel):
    question: str
    mode: str = "general" # 'general' or 'issue'

# -- Helpers --
def progress_callback(current, total, message):
    state.progress = {
        "current": current, 
        "total": total, 
        "message": message,
        "status": "busy"
    }

async def run_analysis(repo_url: str, model: str, use_graph: bool = True, use_faiss: bool = True, api_key: Optional[str] = None):
    """Background task to run the analysis"""
    try:
        progress_callback(0, 100, "Initializing Gemini MAKER engine...")
        state.generator = TutorialGeneratorMAKER(
            model_name=model, 
            progress_callback=progress_callback,
            use_graph=use_graph,
            use_faiss=use_faiss,
            api_key=api_key
        )
        
        # 1. Clone
        progress_callback(10, 100, "Cloning repository...")
        repo_path = state.generator.clone_repository(repo_url)
        repo_name = Path(repo_path).name
        
        # 2. Load
        progress_callback(25, 100, "Loading code documents...")
        documents = state.generator.load_code_documents()
        if not documents:
            progress_callback(0, 100, "Error: No documents found")
            state.progress["status"] = "error"
            return

        # 3. Generate Tutorial
        progress_callback(40, 100, "Generating tutorial with Gemini MAKER framework...")
        tutorial = state.generator.generate_tutorial_maker_style(repo_name, documents)
        state.report_path = state.generator.create_html_report(tutorial, repo_name)
        # 3. Tutorial Report (re-use cached report if already analyzed)
        existing_report = f"reports/{repo_name}_maker.html"
        if os.path.exists(existing_report) and os.path.getsize(existing_report) > 100:
            progress_callback(50, 100, f"Found existing analysis for {repo_name}! Loading cached report...")
            print(f"⚡ Reusing existing tutorial report: {existing_report}")
            state.report_path = existing_report
        else:
            progress_callback(40, 100, "Generating tutorial with Gemini MAKER framework...")
            tutorial = state.generator.generate_tutorial_maker_style(repo_name, documents)
            state.report_path = state.generator.create_html_report(tutorial, repo_name)
        
        # Export code graph if available
        if state.generator.code_graph:
            progress_callback(75, 100, "Exporting code graph...")
            graph_export = f"reports/{repo_name}_codegraph.json"
            state.generator.code_graph.export_graph_json(graph_export)
            if not os.path.exists(graph_export):
                progress_callback(75, 100, "Exporting code graph...")
                state.generator.code_graph.export_graph_json(graph_export)
            else:
                print(f"⚡ Reusing existing code graph export: {graph_export}")
        
        # 4. Setup Q&A
        progress_callback(85, 100, "Setting up Gemini + FAISS Q&A System...")
        state.generator.setup_qa_system(documents, use_persist=True)
        
        progress_callback(100, 100, "Ready!")
        state.progress["status"] = "ready"
        
    except Exception as e:
        print(f"Analysis Error: {e}")
        import traceback
        traceback.print_exc()
        state.progress = {
            "current": 100,
            "total": 100,
            "message": f"Error: {str(e)}",
            "status": "error"
        }

# -- Endpoints --

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.post("/api/analyze")
async def start_analysis(req: AnalyzeRequest, background_tasks: BackgroundTasks):
    if state.progress["status"] == "busy":
        raise HTTPException(status_code=400, detail="System is busy with another analysis")
    
    # Check for API key in request or env
    api_key = req.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key.strip().strip('"').strip("'") == "your_gemini_api_key_here":
        raise HTTPException(
            status_code=400, 
            detail="Valid Gemini API Key required. Please replace 'your_gemini_api_key_here' in your .env file with your actual Google AI Studio API key (starts with AIzaSy...), or enter it in the UI."
        )
    api_key = api_key.strip().strip('"').strip("'")
    
    state.progress["status"] = "starting"
    background_tasks.add_task(run_analysis, req.url, req.model, req.use_graph, req.use_faiss, api_key)
    return {"status": "started", "config": {"use_graph": req.use_graph, "use_faiss": req.use_faiss, "model": req.model}}

@app.get("/api/status")
async def get_status():
    return state.progress

@app.get("/api/report")
async def get_report():
    if not state.report_path or not os.path.exists(state.report_path):
        raise HTTPException(status_code=404, detail="Report not generated yet")
    return FileResponse(state.report_path)

@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not state.generator or not state.generator.qa_chain:
        raise HTTPException(status_code=400, detail="Repository not analyzed yet. Please analyze a repo first.")
    
    if req.mode == "issue":
        enhanced_q = f"""
ISSUE REPORT: {req.question}

Please analyze this issue with reference to the codebase:
1. 🔍 Suspected Features/Components responsible.
2. 📂 Specific files to investigate.
3. 💡 Potential fixes or improvement strategies.
"""
        response = state.generator.ask_question(enhanced_q)
    else:
        response = state.generator.ask_question(req.question)
        
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
