# ⚡ Quick Reference Guide

## Installation

```bash
# Extract the zip
unzip Codebase-Explainer-QnA-Updated.zip
cd Codebase-Explainer-QnA-main

# Install dependencies
pip install -r requirements.txt

# Pull Ollama models (one-time)
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

---

## Start the Server

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Start the Web Server
python server.py

# Open browser
http://localhost:8000
```

---

## Web UI Usage

1. **Paste GitHub URL** in the input field
2. **Check Options**:
   - ✅ Code Graph Context (recommended)
   - ✅ FAISS Vector Store (recommended)
3. **Click "Analyze & Decompose"**
4. **Wait for completion** (~2-5 min depending on repo size)
5. **Chat or use Issue Resolver**

---

## CLI Usage

### Standard (Both Graph + FAISS)
```bash
python tutorial_generator.py --url https://github.com/user/repo --persist
```

### Fast Mode (Graph Only, No FAISS)
```bash
python tutorial_generator.py --url https://github.com/user/repo --no-faiss
```

### Traditional (FAISS Only, No Graph)
```bash
python tutorial_generator.py --url https://github.com/user/repo --no-graph
```

### Custom Model
```bash
python tutorial_generator.py --url https://github.com/user/repo --model llama2:7b --persist
```

---

## Configuration Options

| Option | Default | Effect |
|--------|---------|--------|
| `--url` | Required | GitHub repository URL |
| `--model` | llama3.1:8b | LLM to use |
| `--persist` | False | Cache embeddings to disk |
| `--graph` | True | Enable code structure analysis |
| `--no-graph` | - | Disable code structure analysis |
| `--faiss` | True | Enable vector embeddings |
| `--no-faiss` | - | Disable vector embeddings |

---

## Feature Comparison

### Code Graph
- ✅ Understands code structure
- ✅ Finds related files
- ✅ Tracks imports & dependencies
- ✅ Fast processing
- ❌ No semantic understanding

### FAISS Vectors
- ✅ Semantic understanding
- ✅ Handles paraphrasing
- ✅ General similarity search
- ❌ Misses structural relationships
- ⚠️ Slower (embedding computation)

### Hybrid (Both)
- ✅ Best of both worlds
- ✅ Structural + Semantic understanding
- ✅ Superior accuracy
- ⚠️ Slightly slower
- ✅ **RECOMMENDED**

---

## Example Queries

### For Code Graph
```
"What imports the database module?"
"List all functions in utils.py"
"Show me the dependency chain"
"Which files use the Auth class?"
```

### For FAISS
```
"How does authentication work?"
"Explain the API structure"
"What's the purpose of this module?"
"Find error handling code"
```

### For Hybrid
```
"How does the login process work?"
"Show me all database operations"
"What files make HTTP requests?"
"Debug: why is the cache not working?"
```

---

## Output Files

```
After analysis, you'll find:

reports/
├── {repo_name}_maker.html          # MAKER tutorial report
└── {repo_name}_codegraph.json      # Code dependency graph

db_faiss/
├── index.faiss                     # Vector embeddings
└── index.pkl                       # Metadata
```

---

## Performance Tips

1. **First run is slowest** (embeddings & graph building)
   - Use `--persist` to cache embeddings
   - Graph is lightweight, re-built each time

2. **For large repos** (10k+ files)
   - Consider `--no-graph` for speed
   - Or use smaller model: `--model llama2:7b`

3. **For repeated analysis**
   - Keep `--persist` enabled
   - Embeddings are cached, only graph rebuilds

4. **Memory usage**
   - Graph: ~100MB
   - FAISS: ~2-5GB (depends on repo)
   - Total: ~3GB recommended

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Ollama not running" | Start Ollama: `ollama serve` |
| "FAISS creation slow" | Use `--no-faiss` or be patient |
| "Graph not building" | Run: `pip install networkx` |
| "Out of memory" | Try `--no-faiss` or use smaller model |
| "Web UI not connecting" | Check server is running, visit http://localhost:8000 |
| "Python import error" | Run: `pip install -r requirements.txt` |

---

## Useful Commands

```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# List available models
ollama list

# Pull new model
ollama pull mistral:7b

# Quick test
python -c "from code_graph_builder import CodeGraphBuilder; print('✅ Code Graph OK')"
```

---

## Keyboard Shortcuts (Web UI)

| Shortcut | Action |
|----------|--------|
| `Shift + Enter` | Send message (in textarea) |
| `Enter` | Send message (standalone) |

---

## Chat Modes

### General Chat
- Ask anything about the code
- Gets full context from documents
- Best for understanding code

### Issue Resolver
- Describe a bug or issue
- System suggests files and fixes
- Better for debugging

---

## Tips for Best Results

1. **Be specific**: Instead of "How does it work?", ask "How does authentication work in the login flow?"

2. **Reference files**: "In the auth.py file, why does login fail?"

3. **Use proper terminology**: Use actual class/function names when possible

4. **Follow-ups**: The system remembers context in conversation

5. **Graph queries**: For structural questions, enable Code Graph:
   - "What imports X?"
   - "Show related functions"
   - "Which files define Y?"

---

## Advanced Usage

### Export Code Graph for Visualization
After analysis, find `{repo_name}_codegraph.json` in reports folder:

```json
{
  "nodes": [
    {"id": "file:path/to/file.py", "type": "file"},
    {"id": "func:path/to/file.py:main", "type": "function"},
    ...
  ],
  "edges": [
    {"source": "file:...", "target": "func:...", "type": "defines"},
    ...
  ],
  "summary": {
    "nodes": 450,
    "edges": 1200,
    "density": 0.023
  }
}
```

Use this with visualization tools like:
- Cytoscape.js
- D3.js
- Gephi

### Programmatic Usage

```python
from tutorial_generator import TutorialGeneratorMAKER

# Create generator with custom config
gen = TutorialGeneratorMAKER(
    model_name="llama2:7b",
    use_graph=True,
    use_faiss=True
)

# Clone and load
repo = gen.clone_repository(url)
docs = gen.load_code_documents()

# Ask questions
answer = gen.ask_question("How does this work?")
print(answer["answer"])
```

---

## Environment Setup

```bash
# Virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install
pip install -r requirements.txt

# Run
python server.py
```

---

## Common Issues & Solutions

### "ModuleNotFoundError: No module named 'networkx'"
```bash
pip install networkx==3.6.1
```

### "FAISS index corrupted"
```bash
rm -rf db_faiss/
# Re-run analysis with --persist
```

### "Ollama connection refused"
```bash
# Make sure Ollama is running
ollama serve
# Default port: 11434
```

### Web server won't start
```bash
# Check if port 8000 is in use
# Use different port:
python -c "
import uvicorn
from server import app
uvicorn.run(app, host='0.0.0.0', port=8001)
"
```

---

## Getting Help

1. Check this guide first
2. Run with `--no-graph` to isolate graph issues
3. Run with `--no-faiss` to isolate FAISS issues
4. Check Ollama logs: `ollama serve` output
5. Review generated JSON export for graph structure

---

## What's New in This Version

✨ **Code Graph Context**: AST-based code structure analysis
✨ **Hybrid Retrieval**: Combine FAISS + Graph for better results
✨ **Flexible Configuration**: Enable/disable each component
✨ **Graph Export**: JSON export for visualization
✨ **Better UI**: Toggle buttons for strategies
✨ **CLI Options**: New command-line flags

---

Enjoy! 🚀
