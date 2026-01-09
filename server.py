from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import asyncio
from typing import Optional, List
import os
from pathlib import Path

# Import our generator
from tutorial_generator import TutorialGeneratorMAKER

app = FastAPI(title="Codebase Explainer AI")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# -- State Management --
class GlobalState:
    generator: Optional[TutorialGeneratorMAKER] = None
    progress: dict = {"current": 0, "total": 100, "message": "Idle", "status": "idle"}
    report_path: Optional[str] = None

state = GlobalState()

# -- Models --
class AnalyzeRequest(BaseModel):
    url: str
    model: str = "llama3.1:8b"

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

async def run_analysis(repo_url: str, model: str):
    """Background task to run the analysis"""
    try:
        progress_callback(0, 100, "Cloning repository...")
        state.generator = TutorialGeneratorMAKER(model_name=model, progress_callback=progress_callback)
        
        # 1. Clone
        repo_path = state.generator.clone_repository(repo_url)
        repo_name = Path(repo_path).name
        
        # 2. Load
        documents = state.generator.load_code_documents()
        if not documents:
            progress_callback(0, 100, "Error: No documents found")
            state.progress["status"] = "error"
            return

        # 3. Generate (The Gimmick)
        # Note: We run this because the Q&A system needs the documents.
        # But we could skip the report generation if only Q&A is needed, 
        # but the user *asked* for the generator to be available.
        # We will run the MAKER logic to build the report as requested.
        tutorial = state.generator.generate_tutorial_maker_style(repo_name, documents)
        state.report_path = state.generator.create_html_report(tutorial, repo_name)
        
        # 4. Setup Q&A
        progress_callback(90, 100, "Setting up Q&A Vector Store...")
        state.generator.setup_qa_system(documents, use_persist=True)
        
        progress_callback(100, 100, "Ready!")
        state.progress["status"] = "ready"
        
    except Exception as e:
        print(f"Analysis Error: {e}")
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
        raise HTTPException(status_code=400, detail="System is busy")
    
    state.progress["status"] = "starting"
    background_tasks.add_task(run_analysis, req.url, req.model)
    return {"status": "started"}

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
        raise HTTPException(status_code=400, detail="Repository not analyzed yet")
    
    if req.mode == "issue":
        # Enhanced Prompt for Issue Resolution
        enhanced_q = f"""
        ISSUE REPORT: {req.question}
        
        Please analyze this issue and provide:
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
