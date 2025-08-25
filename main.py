import json

import faiss
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

# -----------------------------
# 1. Load FAQ dari JSON
# -----------------------------
with open("faq.json", encoding="utf-8") as f:
    faq_data = json.load(f)

# -----------------------------
# 2. Load embedding model
# -----------------------------
embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

faq_questions = [item["q"] for item in faq_data]
faq_embeddings = embed_model.encode(faq_questions, convert_to_numpy=True)

dimension = faq_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(faq_embeddings)

# -----------------------------
# 3. Load LLaMA 3
# -----------------------------
llm = Llama.from_pretrained(
    repo_id="bartowski/Meta-Llama-3-8B-Instruct-GGUF",
    filename="Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
    n_ctx=4096,  # panjang konteks
    n_threads=4,  # sesuaikan CPU
    n_gpu_layers=0,
)


# -----------------------------
# 4. Chatbot function
# -----------------------------
def chatbot(query: str) -> str:
    query_emb = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_emb, k=1)

    best_match = faq_data[indices[0][0]]
    faq_answer = best_match["a"].strip()

    if distances[0][0] < 0.7:
        prompt = f"""
Anda adalah asisten Siakadu. Jawablah pertanyaan user hanya berdasarkan FAQ berikut:

FAQ:
{faq_answer}

Pertanyaan user:
{query}

Jawaban singkat, jelas, sopan dan sesuai FAQ:
"""
        output = llm(
            prompt, max_tokens=200, stop=["User:", "FAQ:", "Remember:", "Note:"]
        )
        return output["choices"][0]["text"].strip()
    else:
        return "Maaf, saya hanya bisa menjawab pertanyaan seputar FAQ yang tersedia."


# -----------------------------
# 5. Demo interaktif
# -----------------------------
if __name__ == "__main__":
    print("🤖 Chatbot FAQ Siakadu (ketik 'exit' untuk keluar)")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = chatbot(user_input)
        print("Bot :", response)
