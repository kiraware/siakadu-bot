import json
import os

import faiss
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

faq_questions = [item["q"] for item in faq_data]
faq_embeddings = embed_model.encode(
    faq_questions, convert_to_numpy=True, normalize_embeddings=True
)

dimension = faq_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(faq_embeddings)


# -----------------------------
# 3. Chatbot function
# -----------------------------
def chatbot(query: str) -> str:
    query_emb = embed_model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )
    distances, indices = index.search(query_emb, k=1)

    best_match = faq_data[indices[0][0]]
    faq_question = best_match["q"].strip()
    faq_answer = best_match["a"].strip()
    distance = distances[0][0]

    # Thresholds
    threshold = 0.9  # makin kecil makin ketat

    print(f"Distance: {distance:.3f}")
    print(f"Match: {faq_question}\nAnswer: {faq_answer}\nQuery: {query}")

    if distance < threshold:
        return (
            f'Berdasarkan pertanyaan Anda: "{query}"\n\n'
            f'Ini mirip dengan yang ada di FAQ untuk pertanyaan:\n"{faq_question}"\n\n'
            f"Sehingga jawaban untuk pertanyaan Anda adalah:\n{faq_answer}"
        )
    else:
        return "Maaf, saya hanya bisa menjawab pertanyaan seputar FAQ yang tersedia."


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
