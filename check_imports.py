import sys
import os

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

# Check imports
modules = [
    ("fastapi", "FastAPI"),
    ("uvicorn", "Uvicorn"),
    ("networkx", "NetworkX (Code Graph)"),
    ("faiss", "FAISS (Vector DB)"),
    ("markdown", "Markdown"),
    ("langchain_core", "LangChain Core"),
    ("langchain_google_genai", "LangChain Google GenAI"),
    ("google.genai", "Google GenAI SDK"),
    ("dotenv", "python-dotenv"),
]

all_ok = True
for mod, desc in modules:
    try:
        __import__(mod)
        print(f"  ✅ {desc} ({mod})")
    except ImportError as e:
        print(f"  ❌ {desc} ({mod}) - {e}")
        all_ok = False

if all_ok:
    print("\n🎉 All required packages are installed and ready!")
else:
    print("\n⚠️ Some packages are missing. Install them using the parent .venv.")
