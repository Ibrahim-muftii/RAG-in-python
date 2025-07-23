import os
import pickle
import fitz
import faiss
from sentence_transformers import SentenceTransformer

PDF_PATH = './pdfs/artificial_intelligence_tutorial.pdf'
VECTOR_STORE_DIR = 'verctor_store'
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

def load_pdf_text(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    
    return text

def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks

def save_vectore_store(index, chunks):
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(VECTOR_STORE_DIR, 'faiss.index'))
    with open(os.path.join(VECTOR_STORE_DIR, 'chunks.pkl'), "wb") as f:
        pickle.dump(chunks, f)

def main():
    if(os.path.exists(os.path.join(VECTOR_STORE_DIR, "faiss.index"))):
        print('Vector Store Already Exists. Skipping generation')
        return
    
    print("Loading PDF...")
    text = load_pdf_text(PDF_PATH)

    print('Checking text...')
    chunks = chunk_text(text)

    print("Generating embedding....")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode(chunks, show_progress_bar=True)

    print("Storing in FAISS....")
    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(embeddings)

    save_vectore_store(index, chunks)
    print("Vector store created.")

if __name__ == "__main__":
    main()