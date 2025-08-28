import json
import os

import faiss
import numpy as np
from dotenv import load_dotenv
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
embed_model = SentenceTransformer("BAAI/bge-m3")

# Encode FAQ questions dan normalisasi untuk cosine similarity
faq_questions = [item["q"] for item in faq_data]
faq_embeddings = embed_model.encode(faq_questions, convert_to_numpy=True)
faq_embeddings = faq_embeddings / np.linalg.norm(faq_embeddings, axis=1, keepdims=True)

dimension = faq_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(faq_embeddings)


# -----------------------------
# 3. Chatbot function
# -----------------------------
def chatbot(query: str) -> str:
    query_emb = embed_model.encode([query], convert_to_numpy=True)
    query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)

    scores, indices = index.search(query_emb, k=1)

    best_match = faq_data[indices[0][0]]
    faq_question = best_match["q"].strip()
    faq_answer = best_match["a"].strip()
    similarity = scores[0][0]  # makin tinggi makin mirip (0–1)

    # Thresholds
    high = 0.8  # sangat mirip → langsung jawab
    low = 0.6  # mirip sedang → kasih keterangan
    # < low dianggap tidak relevan

    print(f"Similarity: {similarity:.3f}")
    print(f"Match: {faq_question}\nAnswer: {faq_answer}\nQuery: {query}")

    if similarity > high:
        return faq_answer
    elif similarity > low:
        return (
            f'Berdasarkan pertanyaan Anda: "{query}"\n\n'
            f'Pertanyaan ini mirip dengan FAQ berikut:\n"{faq_question}"\n\n'
            f"Jawaban: {faq_answer}"
        )
    else:
        return "Maaf, saya tidak menemukan jawaban yang relevan di FAQ."


# -----------------------------
# 4. Telegram Handlers
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
# 5. Main
# -----------------------------
def main():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🤖 Bot siap jalan...")
    app.run_polling()


if __name__ == "__main__":
    main()
