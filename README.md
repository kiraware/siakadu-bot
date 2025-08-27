# 🤖 Siakadu FAQ Bot

Bot Telegram sederhana untuk menjawab pertanyaan mahasiswa terkait **Siakadu** berdasarkan daftar **FAQ (Frequently Asked Questions)**.  
Bot ini menggunakan **sentence-transformers (BAAI/bge-m3)** untuk pencocokan pertanyaan dan **FAISS** untuk pencarian vektor yang cepat.

---

## ✨ Fitur

- Jawaban otomatis berdasarkan file `faq.json`.
- Mendukung **bahasa Indonesia** dengan model embedding **bge-m3**.
- Bisa digunakan di **private chat** maupun **group**:
  - Private chat → menjawab semua pertanyaan.
  - Group chat → hanya menjawab jika di-mention (`@siakadu_bot`).
- Konfigurasi sederhana melalui **file `.env`**.
- Clean code dengan Python 3.12, dependency diatur lewat **uv (Astral)**.

---

## 📦 Persyaratan

- Python **>= 3.12**
- [uv (Astral)](https://docs.astral.sh/uv/) untuk manajemen dependency.
- Token bot Telegram dari [@BotFather](https://t.me/BotFather).

---

## 🚀 Instalasi

1. **Clone repository**

   ```bash
   git clone https://github.com/kiraware/siakadu-bot
   cd siakadu-bot
   ```

2. **Install uv (Astral)** jika belum ada

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh                                         # Linux / macOS
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"      # Windows PowerShell
   ```

3. **Install dependencies** dengan `uv`

   ```bash
   uv sync
   ```

4. **Konfigurasi environment**

   - Duplikat file `.env.example` menjadi `.env`
   - Isi token bot Telegram:

   ```env
   TELEGRAM_BOT_TOKEN=1234567890:your_telegram_bot_token
   ```

---

## ▶️ Menjalankan Bot

1. Aktifkan environment:

   ```bash
   source .venv/bin/activate   # Linux / macOS
   .venv\Scripts\activate      # Windows PowerShell
   ```

2. Jalankan bot dengan perintah:
   ```bash
   uv run python main.py
   ```

Jika berhasil, Anda akan melihat pesan:

```bash
🤖 Bot siap jalan...
```

---

## 🧹 Development

**Auto-fix**:

```bash
uv run ruff check --fix
```

**Format**:

```bash
uv run ruff format
```

---

## 📄 Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE).
