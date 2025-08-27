import json
import os

import faiss
from dotenv import load_dotenv
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# -----------------------------
# 0. Load env
# -----------------------------
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# -----------------------------
# 1. Load FAQ dari JSON
# -----------------------------
with open("faq.json", encoding="utf-8") as f:
    faq_data = json.load(f)

# -----------------------------
# 2. Load embedding model
# -----------------------------
embed_model = SentenceTransformer("intfloat/multilingual-e5-large")

# E5 perlu prefix "passage: " untuk dokumen
faq_questions = [f"passage: {item['q']}" for item in faq_data]
faq_embeddings = embed_model.encode(
    faq_questions, convert_to_numpy=True, normalize_embeddings=True
)

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
    # E5 perlu prefix "query: " untuk pertanyaan user
    query_emb = embed_model.encode(
        [f"query: {query}"], convert_to_numpy=True, normalize_embeddings=True
    )
    distances, indices = index.search(query_emb, k=1)

    best_match = faq_data[indices[0][0]]
    faq_answer = best_match["a"].strip()
    distance = distances[0][0]

    # Thresholds (lebih ketat)
    low_threshold = 0.2  # sangat mirip → langsung FAQ
    high_threshold = 0.4  # di atas ini dianggap tidak relevan

    print(f"Distance: {distance:.3f}")
    print(f"Match: {best_match['q']}\nAnswer: {faq_answer}\nQuery: {query}")

    if distance < low_threshold:
        # ✅ langsung jawab dari FAQ tanpa LLaMA
        return faq_answer

    elif distance < high_threshold:
        # 🤖 mirip tapi tidak identik → biar LLaMA rapikan jawabannya
        prompt = f"""
Anda adalah asisten Siakadu. Jawablah pertanyaan user hanya berdasarkan FAQ berikut:

FAQ:
{faq_answer}

Pertanyaan user:
{query}

Jawaban singkat, jelas, sopan dan sesuai FAQ:
"""
        output = llm(
            prompt, max_tokens=200, stop=["User:", "FAQ:", "Note:", "Remember:"]
        )
        return output["choices"][0]["text"].strip()

    else:
        # ❌ tidak mirip → fallback
        return "Maaf, saya hanya bisa menjawab pertanyaan seputar FAQ yang tersedia."


# -----------------------------
# 5. Telegram Handlers
# -----------------------------
async def start(update: Update, _context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Halo 👋, saya adalah bot FAQ Siakadu. Silakan tanya saya di sini!"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text

    # Jika di grup → hanya jawab kalau mention @siakadu_bot
    if update.message.chat.type in ["group", "supergroup"]:
        if f"@{context.bot.username}" not in text:
            return  # tidak menjawab kalau tidak di-mention
        text = text.replace(f"@{context.bot.username}", "").strip()

    response = chatbot(text)
    await update.message.reply_text(response)


# -----------------------------
# 6. Main
# -----------------------------
def main():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🤖 Bot siap jalan...")
    app.run_polling()


if __name__ == "__main__":
    main()
