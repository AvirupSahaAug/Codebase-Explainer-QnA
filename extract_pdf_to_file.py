from pypdf import PdfReader
import os

pdf_path = "maker  method of million step llm task with zero errors.pdf"

# Handle potential filename mismatch if any (reusing previous logic)
if not os.path.exists(pdf_path):
    for f in os.listdir('.'):
        if 'maker' in f.lower() and f.endswith('.pdf'):
            pdf_path = f
            break

try:
    reader = PdfReader(pdf_path)
    with open("pdf_content.txt", "w", encoding="utf-8") as f:
        for i, page in enumerate(reader.pages):
            f.write(f"--- Page {i+1} ---\n")
            text = page.extract_text()
            f.write(text if text else "[No text extracted]")
            f.write("\n\n")
    print(f"Successfully wrote {len(reader.pages)} pages to pdf_content.txt")
except Exception as e:
    print(f"Error: {e}")
