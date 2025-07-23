
import os
import pickle
import faiss
import torch
from transformers import pipelines, AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

VECTORE_STORE_DIR = "verctor_store"
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
Top_K = 3

embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

def load_vector_store():
    index = faiss.read_index(f"{VECTORE_STORE_DIR}/faiss.index")
    with open(f"{VECTORE_STORE_DIR}/chunks.pkl","rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def refine_answer_using_gpt(prompt):
    tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
    model = AutoModelForCausalLM.from_pretrained('distilgpt2')

    inputs = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=512)
    outputs = model.generate(inputs, max_length=512, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def query(question):

    question_embedding = embedder.encode([question])
    index, chunks = load_vector_store()
    D, I = index.search(question_embedding, Top_K)

    relevant_chunks = [chunks[i] for i in I[0]]
    context = "\n".join(relevant_chunks)
    prompt = f"Context:\n {context} \n\nQuestions:{question}\nAnswer:"
    refined_answer = refine_answer_using_gpt(prompt)

    print("Top Matches", I[0])
    print(refined_answer)

if __name__ == "__main__":
    while True:
        q = input("\nAsk something (or type 'exit'): ")
        if q.lower() == "exit":
            break
        query(q)
        