from pypdf import PdfReader
import os

pdf_path = "maker  method of million step llm task with zero errors.pdf"

if not os.path.exists(pdf_path):
    print(f"Error: File not found at {pdf_path}")
    # Try finding it with listdir to be sure concerning spaces
    for f in os.listdir('.'):
        if 'maker' in f.lower() and f.endswith('.pdf'):
            print(f"Found similar file: {f}")
            pdf_path = f
            break

try:
    reader = PdfReader(pdf_path)
    text = ""
    # Read first 5 pages to get the gist, as full text might be too long
    for page in reader.pages[:10]:
        text += page.extract_text() + "\n"
    print(text)
except Exception as e:
    print(f"Error reading PDF: {e}")
