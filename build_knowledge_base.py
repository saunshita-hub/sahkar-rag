import os
import re
from pypdf import PdfReader
import pytesseract
from pdf2image import convert_from_path
import chromadb
from chromadb.utils import embedding_functions

PDF_FOLDER = "pdfs"
DB_FOLDER = "chroma_db"

# Common junk patterns found in "saved web page" PDFs
NOISE_PATTERNS = [
    r'^\d{1,2}/\d{1,2}/\d{2,4},?\s*\d{1,2}:\d{2}\s*(AM|PM)$',
    r'^https?://\S+$',
    r'^Discover more$',
    r'^Get the Complete Guide$',
    r'^CrowdStrike',
    r'^Download$',
    r'^Post Comment$',
    r'^Leave a Reply$',
    r'^Comment \*$',
    r'^Name \*$',
    r'^Email \*$',
    r'^Website$',
    r'^Save my name, email',
    r'^Copyright ©',
    r'^Previous:',
    r'^Next:',
    r'^Your email address will not be published',
]


def clean_text(text):
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if any(re.match(p, stripped, re.IGNORECASE) for p in NOISE_PATTERNS):
            continue
        if len(stripped) < 25 and not stripped.endswith((".", ":", ")", "?", "!")):
            continue
        cleaned_lines.append(stripped)
    return "\n".join(cleaned_lines)


def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    return full_text


def extract_pdf_text_ocr(pdf_path):
    print("    (using OCR — this PDF has no readable text, will take longer)")
    pages = convert_from_path(pdf_path, dpi=150)
    full_text = ""
    for i, page_image in enumerate(pages):
        text = pytesseract.image_to_string(page_image)
        full_text += text + "\n"
        if (i + 1) % 20 == 0:
            print(f"    OCR progress: page {i+1}/{len(pages)}")
    return full_text


def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
    return chunks


def build():
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF files\n")

    all_chunks = []
    all_ids = []
    all_metadata = []

    for filename in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER, filename)
        print(f"Processing {filename}...")

        try:
            text = extract_pdf_text(pdf_path)

            # If normal extraction found almost nothing, it's likely a scanned PDF — use OCR
            if len(text.strip()) < 50:
                text = extract_pdf_text_ocr(pdf_path)

            text = clean_text(text)

            if not text.strip():
                print("  Skipped (no usable text found even after OCR/cleaning)\n")
                continue

            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_ids.append(f"{filename}_chunk_{i}")
                all_metadata.append({"source": filename})

            print(f"  -> {len(chunks)} chunks created\n")

        except Exception as e:
            print(f"  Error processing {filename}: {e}\n")

    print(f"Total chunks across all documents: {len(all_chunks)}")

    print("\nSetting up vector database (this step computes embeddings, may take several minutes)...")
    chroma_client = chromadb.PersistentClient(path=DB_FOLDER)
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()

    try:
        chroma_client.delete_collection(name="cooperative_docs")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name="cooperative_docs",
        embedding_function=embedding_fn
    )

    batch_size = 200
    total = len(all_chunks)
    for i in range(0, total, batch_size):
        batch_docs = all_chunks[i:i + batch_size]
        batch_ids = all_ids[i:i + batch_size]
        batch_meta = all_metadata[i:i + batch_size]
        collection.add(documents=batch_docs, ids=batch_ids, metadatas=batch_meta)
        print(f"  Stored batch {i}-{min(i + batch_size, total)} of {total}")

    print(f"\nDone! Knowledge base built and saved to '{DB_FOLDER}/' folder.")
    print(f"Total chunks stored: {collection.count()}")


if __name__ == "__main__":
    build()