import sys
print(sys.executable)
try:
    import langchain
    print(f"Langchain version: {langchain.__version__}")
    from langchain.chains import RetrievalQA
    print("RetrievalQA imported successfully from langchain.chains")
except ImportError as e:
    print(f"Error: {e}")
    try:
        from langchain.chains import RetrievalQA
        print("RetrievalQA imported (2nd attempt)")
    except ImportError as e2:
        print(f"Error 2: {e2}")
